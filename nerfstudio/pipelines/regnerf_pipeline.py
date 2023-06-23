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
Abstracts for the Pipeline class.
Author: Patrick Huang
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch

from nerfstudio.data.datamanagers.regnerf_datamanager import RegNerfDataManager, RegNerfDataManagerConfig
from nerfstudio.models.regnerf import RegNerfModel
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class RegNerfPipelineConfig(VanillaPipelineConfig):
    """Config for RegNerf pipeline."""

    _target: Type = field(default_factory=lambda: RegNerfPipeline)
    """Overrides superclass."""
    datamanager: RegNerfDataManagerConfig = RegNerfDataManagerConfig()
    """Overrides superclass."""


class RegNerfPipeline(VanillaPipeline):
    """
    Pipeline for RegNerf.
    """

    datamanager: RegNerfDataManager
    model: RegNerfModel

    def get_train_loss_dict(self, step: int):
        """
        Overrides super class.
        In addition to superclass functionality, adds `anneal_fac` to `loss_dict`.
        Passes the ``step`` argument to model.
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle, step)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, step)

        # Add anneal fac to metrics.
        anneal_fac = self.model.collider.get_anneal_fac(step)
        loss_dict["anneal_fac"] = torch.tensor(anneal_fac)

        return model_outputs, loss_dict, metrics_dict
