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
from xrtm.eval.core.eval.definitions import EvaluationReport, Evaluator
from xrtm.eval.core.schemas import ForecastResolution

# From xrtm-forecast (Internal)
from xrtm.forecast.kit.agents.base import Agent

logger = logging.getLogger(__name__)


class Backtester:
    r"""Orchestrates backtesting for a given agent and evaluator.

    Args:
        agent: The forecasting agent to evaluate.
        evaluator: Scoring backend implementing the ``Evaluator`` protocol.
    """

    def __init__(self, agent: Agent, evaluator: Evaluator):
        self.agent = agent
        self.evaluator = evaluator

    async def run(self, dataset: List[Tuple[ForecastQuestion, ForecastResolution]]) -> EvaluationReport:
        r"""Run the full backtest and return an evaluation report.

        Args:
            dataset: List of (question, resolution) pairs.

        Returns:
            An ``EvaluationReport`` containing per-question results and aggregate score.
        """
        async def process_question(question, resolution):
            try:
                logger.info(f"Backtesting question: {question.id}")
                prediction = await self.agent.run(question)
                conf = getattr(prediction, "confidence", prediction)
                return self.evaluator.evaluate(prediction=conf, ground_truth=resolution.outcome, subject_id=question.id)
            except Exception as e:
                logger.error(f"Failed to evaluate question {question.id}: {e}")
                return None

        # Execute all questions concurrently
        tasks = [process_question(q, r) for q, r in dataset]
        processed_results = await asyncio.gather(*tasks)

        # Filter out failed evaluations
        results = [res for res in processed_results if res is not None]

        count = len(results)
        total_score = sum(res.score for res in results)
        mean_score = total_score / count if count > 0 else 0.0

        return EvaluationReport(
            metric_name="Brier Score", mean_score=mean_score, total_evaluations=count, results=results
        )


__all__ = ["Backtester"]
