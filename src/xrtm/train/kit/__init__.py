# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.
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

r"""
Researcher Kit for xrtm-train.

This module provides high-level, composable utilities for training.
Includes memory management, optimization strategies, sample building,
trainers, and neural projection utilities.

Note: Projectors require PyTorch. They are lazily imported.
"""

from xrtm.train.kit.builders import (
    BetaPriorSnapshot,
    NewsEvent,
    TrainingSample,
    TrainingSampleBuilder,
)
from xrtm.train.kit.trainers import (
    BaseTrainer,
    ReasoningTrace,
    SingleNodeTrainer,
    TrainResult,
)

__all__ = [
    # Builders (always available)
    "NewsEvent",
    "BetaPriorSnapshot",
    "TrainingSample",
    "TrainingSampleBuilder",
    # Trainers (always available)
    "BaseTrainer",
    "SingleNodeTrainer",
    "TrainResult",
    "ReasoningTrace",
    # Projectors (require torch)
    "SinusoidalEmbedding",
    "BetaPriorProjector",
    "DualHeadProjector",
]


def __getattr__(name: str):
    r"""Lazy import for torch-dependent projectors."""
    if name in ("SinusoidalEmbedding", "BetaPriorProjector", "DualHeadProjector"):
        from xrtm.train.kit.projectors import (
            BetaPriorProjector,
            DualHeadProjector,
            SinusoidalEmbedding,
        )
        return {
            "SinusoidalEmbedding": SinusoidalEmbedding,
            "BetaPriorProjector": BetaPriorProjector,
            "DualHeadProjector": DualHeadProjector,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
