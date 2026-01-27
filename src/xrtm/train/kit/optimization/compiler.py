# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import logging
from typing import Any, List, Tuple

# From xrtm-forecast
from xrtm.forecast.kit.agents.prompting import CompiledAgent, PromptTemplate

logger = logging.getLogger(__name__)

class BrierOptimizer:
    def __init__(self, optimizer_model: Any):
        self.optimizer_model = optimizer_model

    async def optimize(self, agent: CompiledAgent, dataset: List[Tuple[Any, float, int]]) -> PromptTemplate:
        total_error = sum((p - g) ** 2 for _, p, g in dataset) / len(dataset)
        meta_prompt = f"""
        You are a Prompt Optimizer for a forecasting engine.
        Current Goal: Minimize Brier Score (Current: {total_error:.4f}).
        Current Instruction: "{agent.template.instruction}"
        Performance Snippets (Input -> Pred vs Truth):
        """
        for inp, pred, truth in dataset[:5]:
            meta_prompt += f"- {str(inp)[:50]}... -> {pred} (Actual: {truth})\n"
        meta_prompt += "\nSuggest a NEW system instruction that corrects for these errors.\nReturn ONLY the new instruction string."
        new_instruction = await self.optimizer_model.generate(meta_prompt)
        return PromptTemplate(
            instruction=new_instruction.strip(), examples=agent.template.examples, version=agent.template.version + 1
        )

__all__ = ["BrierOptimizer"]
