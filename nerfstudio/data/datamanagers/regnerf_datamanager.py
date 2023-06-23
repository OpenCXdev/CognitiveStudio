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
RegNerf datamanager.
Author: Patrick Huang
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Tuple, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig, TDataset


@dataclass
class RegNerfDataManagerConfig(VanillaDataManagerConfig):
    """Data manager for RegNerf."""

    _target: Type = field(default_factory=lambda: RegNerfDataManager)
    """Target class to instantiate."""


class RegNerfDataManager(VanillaDataManager, Generic[TDataset]):
    """
    Data manager for RegNerf.

    Extended functionalities:
    - Generates random poses and rays (TODO).
    - Performs sample space annealing.
    """

    config: RegNerfDataManagerConfig
