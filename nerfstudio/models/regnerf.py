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
from nerfstudio.model_components.losses import MSELoss
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

    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 10
    """How far along the ray to stop sampling."""

    randpose_count: int = 10000
    """Number of random poses to generate for regularization."""
    randpose_radius: float = 3
    """Random poses are sampled from a sphere of this radius."""
    randpose_only_up: bool = False
    """Whether to only sample random poses from upper hemisphere."""
    randpose_s_patch: int = 8
    """Random pose patch size."""
    randpose_focal: float = 16
    """Random pose patch focal length."""
    randpose_bs: int = 8
    """Batch size (number of poses) per step for regularization."""


class RegNerfModel(Model):
    """RegNeRF model

    Args:
        config: RegNerf configuration to instantiate model
    """

    config: RegNerfModelConfig

    def __init__(
        self,
        config: RegNerfModelConfig,
        **kwargs,
    ) -> None:
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
            """
            Normalizes ``x`` on last dimension.
            """
            return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

        # Sample from top hemisphere.
        origins = torch.randn((self.config.randpose_count, 3), dtype=torch.float32, requires_grad=True)
        if self.config.randpose_only_up:
            origins[:, 2] = torch.abs(origins[:, 2])
        origins = normalize(origins) * self.config.randpose_radius

        # Create SO(3) rotation matrices. Look at (0, 0, 0)+jitter from each ``origin[i]``.
        # From the paper: Add noise to target point.
        noise = 0.125 * torch.randn((1, 3), dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[0, 0, 0]], dtype=torch.float32, requires_grad=True) + noise
        up = torch.tensor([[0, 0, 1]], dtype=torch.float32, requires_grad=True)
        forward = normalize(target - origins)
        side = normalize(torch.cross(forward, up))
        up = normalize(torch.cross(side, forward))
        forward = -1 * forward
        rotations = torch.stack([side, up, forward], dim=-1)

        # Combine ``origins`` and ``rotations`` to SE(3) poses.
        # We use a 3x4 matrix because the last row doesn't matter.
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
        focal = self.config.randpose_focal

        x, y = torch.meshgrid(
            torch.linspace(-1, 1, s_patch, requires_grad=True),
            torch.linspace(-1, 1, s_patch, requires_grad=True),
            indexing="xy",
        )
        # Shape (s_patch, s_patch, 3)
        # Camera faces in -Z direction. We will rotate these rays in the next step.
        # ray_dirs[i, j] gives ray direction for pixel (i, j) for the -Z camera.
        ray_dirs = torch.stack([x, y, -focal * torch.ones_like(x, requires_grad=True)], dim=-1)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
        # Now for each pose, apply the pose's rotation to ray_dirs
        # This creates a unique set of rays for each pose.
        # Shape (randpose_count, s_patch, s_patch, 3)
        # This line is matrix mult on last two dims; the operator @ doesn't work.
        pose_ray_dirs = torch.sum(poses[:, None, None, :, :3] * ray_dirs[None, ..., None, :], dim=-1)
        # origins is same shape as pose_ray_dirs.
        origins = torch.empty_like(pose_ray_dirs)
        origins[:] = poses[:, None, None, :, 3]

        # Compute pixel area
        # Squared distance between pose_ray_dirs[i, j] and pose_ray_dirs[i, j+1]
        # Shape (randpose_count, s_patch-1, s_patch)
        dx = torch.sum(
            (pose_ray_dirs[:, :-1] - pose_ray_dirs[:, 1:]) ** 2,
            dim=-1,
        )
        # Convert dim 1 back to original shape. In the previous step it was reduced by 1.
        dx = torch.cat([dx, dx[:, -2:-1]], dim=1)
        pixel_area = dx[..., None]

        # Create RayBundle
        rays = RayBundle(
            origins=origins,
            directions=pose_ray_dirs,
            pixel_area=pixel_area,
            nears=torch.full_like(origins[..., :1], self.config.near_plane, requires_grad=True),
            fars=torch.full_like(origins[..., :1], self.config.far_plane, requires_grad=True),
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
        return F.mse_loss(v00, v01) + F.mse_loss(v00, v10)

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
        self.renderer_depth = DepthRenderer()

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
        # Choose a random batch of poses.
        patches = random.choices(range(self.config.randpose_count), k=self.config.randpose_bs)
        # Forward pass
        rays = RayBundle(
            origins=self.random_rays.origins[patches].view(-1, 3),
            directions=self.random_rays.directions[patches].view(-1, 3),
            pixel_area=self.random_rays.pixel_area[patches].view(-1, 1),
            nears=self.random_rays.nears[patches].view(-1, 1),
            fars=self.random_rays.fars[patches].view(-1, 1),
        )
        patch_outputs = self.get_outputs(rays)
        # Compute loss on coarse and fine.
        depth_loss = 0
        for depth in (patch_outputs["depth_coarse"], patch_outputs["depth_fine"]):
            depth = depth.view(len(patches), self.config.randpose_s_patch, self.config.randpose_s_patch)
            depth_loss += self.depth_smoothness_loss(depth)
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

        assert isinstance(fine_ssim, torch.Tensor)
        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            "coarse_psnr": float(coarse_psnr.item()),
            "fine_psnr": float(fine_psnr.item()),
            "fine_ssim": float(fine_ssim.item()),
            "fine_lpips": float(fine_lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
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
