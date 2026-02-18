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
Prompt optimization via Brier score minimization.

Uses an LLM meta-optimizer to rewrite agent instructions based on
observed prediction errors, minimizing the Brier score on a held-out dataset.
"""

import logging
from typing import Any, List, Tuple

# From xrtm-forecast
from xrtm.forecast.kit.agents.prompting import CompiledAgent, PromptTemplate

logger = logging.getLogger(__name__)


class BrierOptimizer:
    r"""Optimizes agent prompts to minimize Brier score via meta-learning."""
    def __init__(self, optimizer_model: Any):
        self.optimizer_model = optimizer_model

    async def optimize(self, agent: CompiledAgent, dataset: List[Tuple[Any, float, int]]) -> PromptTemplate:
        total_error = sum((p - g) ** 2 for _, p, g in dataset) / len(dataset)
        parts = [f"""
        You are a Prompt Optimizer for a forecasting engine.
        Current Goal: Minimize Brier Score (Current: {total_error:.4f}).
        Current Instruction: "{agent.template.instruction}"
        Performance Snippets (Input -> Pred vs Truth):
        """]
        for inp, pred, truth in dataset[:5]:
            parts.append(f"- {str(inp)[:50]}... -> {pred} (Actual: {truth})\n")
        parts.append("\nSuggest a NEW system instruction that corrects for these errors.\nReturn ONLY the new instruction string.")
        meta_prompt = "".join(parts)
        new_instruction = await self.optimizer_model.generate(meta_prompt)
        return PromptTemplate(
            instruction=new_instruction.strip(), examples=agent.template.examples, version=agent.template.version + 1
        )


__all__ = ["BrierOptimizer"]
