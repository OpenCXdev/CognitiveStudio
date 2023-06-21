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

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class RegNerfPipelineConfig(VanillaPipelineConfig):
    """Config for RegNerf pipeline."""

    _target: Type = field(default_factory=lambda: RegNerfPipeline)
    """Overrides superclass."""


class RegNerfPipeline(VanillaPipeline):
    """
    Pipeline for RegNerf.
    """
