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
Backtesting orchestrator for temporal evaluation.

Runs an ``Agent`` against a dataset of (question, resolution) pairs,
computes scoring via an ``Evaluator``, and produces an ``EvaluationReport``.
Ensures temporal isolation to prevent look-ahead bias.
"""

import asyncio
import logging
from typing import List, Tuple

# From xrtm-data
from xrtm.data.core.schemas import ForecastQuestion

# From xrtm-eval
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult, Evaluator
from xrtm.eval.core.schemas import ForecastResolution

# From xrtm-forecast (Internal)
from xrtm.forecast.kit.agents.base import Agent

from xrtm.train.simulation.artifacts import (
    prediction_value_and_payload,
    resolution_payload,
    validate_resolution_for_question,
)

logger = logging.getLogger(__name__)


class Backtester:
    r"""Orchestrates backtesting for a given agent and evaluator.

    Args:
        agent: The forecasting agent to evaluate.
        evaluator: Scoring backend implementing the ``Evaluator`` protocol.
    """

    def __init__(self, agent: Agent, evaluator: Evaluator, concurrency: int = 5):
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.agent = agent
        self.evaluator = evaluator
        self.concurrency = concurrency

    async def run(self, dataset: List[Tuple[ForecastQuestion, ForecastResolution]]) -> EvaluationReport:
        r"""Run the full backtest and return an evaluation report.

        Args:
            dataset: List of (question, resolution) pairs.

        Returns:
            An ``EvaluationReport`` containing per-question results and aggregate score.
        """
        async def process_question(question: ForecastQuestion, resolution: ForecastResolution) -> EvaluationResult:
            try:
                logger.info("Backtesting question: %s", question.id)
                prediction = await self.agent.run(question)
                validated_resolution = validate_resolution_for_question(resolution, question.id)
                prediction_value, prediction_payload = prediction_value_and_payload(prediction)
                result = self.evaluator.evaluate(
                    prediction=prediction_value, ground_truth=validated_resolution.outcome, subject_id=question.id
                )
                result.metadata["resolution_payload"] = resolution_payload(validated_resolution)
                if prediction_payload is not None:
                    result.metadata["prediction_payload"] = prediction_payload
                return result
            except Exception as e:
                logger.error("Failed to evaluate question %s: %s", question.id, e)
                return EvaluationResult(
                    subject_id=question.id,
                    score=1.0,
                    ground_truth=None,
                    prediction=0.5,
                    metadata={"error": str(e), "resolution_payload": resolution_payload(resolution)},
                )

        queue: asyncio.Queue[tuple[int, ForecastQuestion, ForecastResolution]] = asyncio.Queue()
        processed_results: list[EvaluationResult | None] = [None] * len(dataset)
        for idx, (question, resolution) in enumerate(dataset):
            queue.put_nowait((idx, question, resolution))

        async def worker() -> None:
            while True:
                try:
                    idx, question, resolution = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    processed_results[idx] = await process_question(question, resolution)
                finally:
                    queue.task_done()

        if dataset:
            worker_count = min(self.concurrency, len(dataset))
            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            await queue.join()
            await asyncio.gather(*workers)

        results = [res for res in processed_results if res is not None]

        count = len(results)
        total_score = sum(res.score for res in results)
        mean_score = total_score / count if count > 0 else 0.0

        return EvaluationReport(
            metric_name="Brier Score", mean_score=mean_score, total_evaluations=count, results=results
        )


__all__ = ["Backtester"]
