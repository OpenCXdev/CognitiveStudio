# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of RegNeRF.
Author: Patrick Huang
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.vanilla_nerf_field import NeRFField
from nerfstudio.model_components.losses import MSELoss, depth_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.models.base_model import Model
from nerfstudio.models.vanilla_nerf import VanillaModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class RegNerfModelConfig(VanillaModelConfig):
    """RegNerf config"""

    loss_coefficients: Dict[str, float] = to_immutable_dict({
        "rgb_loss_coarse": 1,
        "rgb_loss_fine": 1,
        "depth_smoothness": 1,
        "color_likelihood": 1,
    })
    """Overrides corresponding param from ModelConfig."""

    randpose_count: int = 10000
    """Number of random poses to generate for regularization."""
    randpose_radius: float = 4.03
    """Radius (distance to origin) of random poses."""
    randpose_only_up: bool = False
    """Whether to only sample random poses from upper hemisphere."""
    randpose_s_patch: int = 8
    """Random pose patch size."""
    #randpose_focal: float = 1000
    #"""Random pose patch focal length."""
    randpose_bs: int = 32
    """Batch size (number of poses) per step for regularization."""


class RegNerfModel(Model):
    """RegNeRF model

    Args:
        config: RegNerf configuration to instantiate model
    """

    config: RegNerfModelConfig

    def __init__(self, config: RegNerfModelConfig, **kwargs) -> None:
        self.focal = kwargs["focal"][0][0].item()
        """Focal length of dataparser."""

        self.field = None
        assert config.collider_params is not None, "RegNeRF requires bounding box collider parameters."
        super().__init__(config=config, **kwargs)
        assert self.config.collider_params is not None, "RegNeRF requires collider parameters to be set."

    def generate_random_poses(self) -> torch.Tensor:
        """
        Generate random poses to do regularization from.
        Returns:
            Torch tensor of shape (randpose_count, 3, 4)
            ret[i] gives SE(3) pose of camera i.
        """
        def normalize(x, eps=1e-8):
            """Normalizes ``x`` on last dimension."""
            return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

        # Sample from top hemisphere.
        origins = torch.randn((self.config.randpose_count, 3), dtype=torch.float32)
        if self.config.randpose_only_up:
            origins[:, 2] = torch.abs(origins[:, 2])
        origins = normalize(origins) * self.config.randpose_radius

        # Create SO(3) rotation matrices. Look at (0, 0, 0)+jitter from each ``origin[i]``.
        # From the paper: Add noise to target point.
        noise = 0.125 * torch.randn((1, 3), dtype=torch.float32)
        target = torch.tensor([[0, 0, 0]], dtype=torch.float32) + noise
        up = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        forward = normalize(target - origins)
        side = normalize(torch.cross(forward, up))
        up = normalize(torch.cross(side, forward))
        forward = -1 * forward
        rotations = torch.stack([side, up, forward], dim=-1)

        # Combine ``origins`` and ``rotations`` to SE(3) poses.
        # We use a 3x4 matrix (instead of 4x4) because the last row doesn't matter.
        poses = torch.cat([rotations, origins.unsqueeze(-1)], dim=-1)

        return poses

    def generate_patch_rays(self, poses) -> RayBundle:
        """
        For each pose, generate a square patch of rays.
        Patch size is config.randpose_s_patch
        Focal length is config.focal_length
        Return:
            RayBundle where origins and directions are the same shape.
            directions[n][i][j] gives ray dir for
                - random pose n,
                - pixel (i, j) in that pose's patch.
            Same for origins
        """
        s_patch = self.config.randpose_s_patch
        focal = self.focal

        x, y = torch.meshgrid(
            torch.linspace(-1, 1, s_patch),
            torch.linspace(-1, 1, s_patch),
            indexing="xy",
        )
        # Shape (s_patch, s_patch, 3)
        # Camera faces in -Z direction. We will rotate these rays in the next step.
        # ray_dirs[i, j] gives ray direction for pixel (i, j) for the -Z camera.
        ray_dirs = torch.stack([x, y, -focal * torch.ones_like(x)], dim=-1)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        # Now for each pose, apply the pose's rotation to ray_dirs
        # This creates a unique set of rays for each pose.
        # Shape (randpose_count, s_patch, s_patch, 3)
        # This line is matrix mult on last two dims; the operator @ doesn't work.
        directions = torch.sum(poses[:, None, None, :, :3] * ray_dirs[None, ..., None, :], dim=-1)
        # origins is same shape as pose_ray_dirs.
        origins = torch.empty_like(directions)
        origins[:] = poses[:, None, None, :, 3]

        # Compute pixel area
        # Squared distance between pose_ray_dirs[i, j] and pose_ray_dirs[i, j+1]
        # Shape (randpose_count, s_patch-1, s_patch)
        dx = torch.sum(
            (directions[:, :-1] - directions[:, 1:]) ** 2,
            dim=-1,
        )
        # Convert dim 1 back to original shape. In the previous step it was reduced by 1.
        dx = torch.cat([dx, dx[:, -2:-1]], dim=1)
        pixel_area = dx[..., None]

        # Create RayBundle
        origins.requires_grad = True
        directions.requires_grad = True
        rays = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            nears=torch.full_like(origins[..., :1], self.config.collider_params["near_plane"]),
            fars=torch.full_like(origins[..., :1], self.config.collider_params["far_plane"]),
        ).to("cuda")
        return rays

    def depth_smoothness_loss(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Compute depth smoothness loss on one patch.

        Args:
            depth: Predicted depth map. Shape (bs, h, w)
        """
        v00 = depth[:, :-1, :-1]
        v01 = depth[:, :-1, 1:]
        v10 = depth[:, 1:, :-1]
        loss = F.mse_loss(v00, v01) + F.mse_loss(v00, v10)
        return loss

    def forward_random_poses(self, rays: RayBundle, k: int) -> Tuple[Dict[str, torch.Tensor], RayBundle, List[int]]:
        """
        Chooses random poses, and pass them forward through the model.

        Why is there a separate function for this? Because there is a lot of messy code
        dealing with tensor shapes. This function is a wrapper that takes care of that.

        Args:
            rays: self.random_rays
            k: Number of random poses to choose

        Return:
            (outputs, rays, indices)
            outputs: Dict of model outputs. Same type as self.get_outputs, except reshaped.
            rays: Random indices of original rays. Flattened.
            indices: Indices of random poses. Shape (k,)
        """
        # Choose indices
        indices = random.choices(range(self.config.randpose_count), k=k)

        # From shape (pose, s_patch, s_patch, 3) to (flat, 3). Required by Nerfstudio.
        rays = RayBundle(
            origins=rays.origins[indices].view(-1, 3),
            directions=rays.directions[indices].view(-1, 3),
            pixel_area=rays.pixel_area[indices].view(-1, 1),
            nears=rays.nears[indices].view(-1, 1),
            fars=rays.fars[indices].view(-1, 1),
        )

        outputs = self.get_outputs(rays)
        # Reshape everything from flat to (pose, s_patch, s_patch, ...)
        for key in outputs:
            outputs[key] = outputs[key].view(k, self.config.randpose_s_patch, self.config.randpose_s_patch, *outputs[key].shape[1:])

        return outputs, rays, indices

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # Set up random poses.
        self.random_poses = self.generate_random_poses()
        self.random_rays = self.generate_patch_rays(self.random_poses)
        #plot_patch_rays(self.random_rays)

        # setting up fields
        position_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=16, min_freq_exp=0.0, max_freq_exp=16.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        self.field = NeRFField(
            position_encoding=position_encoding, direction_encoding=direction_encoding, use_integrated_encoding=True
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples, include_original=False)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)

        # First pass: coarse network
        field_outputs_coarse = self.field.forward(ray_samples_uniform)
        weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
        rgb_coarse = self.renderer_rgb(
            rgb=field_outputs_coarse[FieldHeadNames.RGB],
            weights=weights_coarse,
        )
        accumulation_coarse = self.renderer_accumulation(weights_coarse)
        depth_coarse = self.renderer_depth(weights_coarse, ray_samples_uniform)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)

        # Second pass: fine network
        field_outputs_fine = self.field.forward(ray_samples_pdf)
        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        rgb_fine = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        accumulation_fine = self.renderer_accumulation(weights_fine)
        depth_fine = self.renderer_depth(weights_fine, ray_samples_pdf)

        outputs = {
            "rgb_coarse": rgb_coarse,
            "rgb_fine": rgb_fine,
            "accumulation_coarse": accumulation_coarse,
            "accumulation_fine": accumulation_fine,
            "depth_coarse": depth_coarse,
            "depth_fine": depth_fine,
        }
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # RGB loss between render and ground truth.
        image = batch["image"].to(self.device)
        rgb_loss_coarse = self.rgb_loss(image, outputs["rgb_coarse"])
        rgb_loss_fine = self.rgb_loss(image, outputs["rgb_fine"])

        # Depth smoothness loss.
        patch_outputs, _, _ = self.forward_random_poses(self.random_rays, self.config.randpose_bs)
        depth_loss = 0
        depth_loss += self.depth_smoothness_loss(patch_outputs["depth_coarse"])
        depth_loss += self.depth_smoothness_loss(patch_outputs["depth_fine"])
        depth_loss /= 2

        loss_dict = {
            "rgb_loss_coarse": rgb_loss_coarse,
            "rgb_loss_fine": rgb_loss_fine,
            "depth_smoothness": depth_loss,
        }
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        assert self.config.collider_params is not None, "RegNeRF requires collider parameters to be set."
        image = batch["image"].to(outputs["rgb_coarse"].device)
        rgb_coarse = outputs["rgb_coarse"]
        rgb_fine = outputs["rgb_fine"]
        acc_coarse = colormaps.apply_colormap(outputs["accumulation_coarse"])
        acc_fine = colormaps.apply_colormap(outputs["accumulation_fine"])

        assert self.config.collider_params is not None
        depth_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )
        depth_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        combined_acc = torch.cat([acc_coarse, acc_fine], dim=1)
        combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb_coarse = torch.moveaxis(rgb_coarse, -1, 0)[None, ...]
        rgb_fine = torch.moveaxis(rgb_fine, -1, 0)[None, ...]
        rgb_coarse = torch.clip(rgb_coarse, min=0, max=1)
        rgb_fine = torch.clip(rgb_fine, min=0, max=1)

        coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb_fine)
        fine_ssim = self.ssim(image, rgb_fine)
        fine_lpips = self.lpips(image, rgb_fine)


        # Render examples of depth smoothness maps.
        bs = 3
        outputs, rays, indices = self.forward_random_poses(self.random_rays, bs)
        near = self.config.collider_params["near_plane"]
        far = self.config.collider_params["far_plane"]
        depth_sm_coarse = colormaps.apply_depth_colormap(
            outputs["depth_coarse"],
            accumulation=outputs["accumulation_coarse"],
            near_plane=near,
            far_plane=far,
        )
        depth_sm_fine = colormaps.apply_depth_colormap(
            outputs["depth_fine"],
            accumulation=outputs["accumulation_fine"],
            near_plane=near,
            far_plane=far,
        )
        depth_sm = torch.cat([depth_sm_coarse, depth_sm_fine], dim=0)
        # Shape (s_patch, s_patch, rgb_channels)
        depth_sm = depth_sm[0]

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "depth_smoothness": depth_sm,
        }
        return metrics_dict, images_dict


def plot_patch_rays(patch_rays: RayBundle):
    """
    3D matplotlib of a few patches.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(3):
        dirs = patch_rays.directions[i].view(-1, 3).detach().cpu().numpy()
        origin = patch_rays.origins[i].view(-1, 3).detach().cpu().numpy()
        for j in range(dirs.shape[0]):
            lc = art3d.Line3DCollection([[origin[j], origin[j]+dirs[j]]])
            ax.add_collection(lc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
