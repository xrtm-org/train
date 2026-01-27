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

import logging
from typing import List, Tuple

# From xrtm-data
from xrtm.data.schemas.forecast import ForecastQuestion

# From xrtm-eval
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult, Evaluator
from xrtm.eval.schemas.forecast import ForecastResolution

# From xrtm-forecast (Internal)
from xrtm.forecast.kit.agents.base import Agent

logger = logging.getLogger(__name__)


class Backtester:
    """Orchestrates the backtesting process for a specific agent and evaluator."""

    def __init__(self, agent: Agent, evaluator: Evaluator):
        self.agent = agent
        self.evaluator = evaluator

    async def run(self, dataset: List[Tuple[ForecastQuestion, ForecastResolution]]) -> EvaluationReport:
        results: List[EvaluationResult] = []
        total_score = 0.0

        for question, resolution in dataset:
            try:
                logger.info(f"Backtesting question: {question.id}")
                prediction = await self.agent.run(question)
                conf = getattr(prediction, "confidence", prediction)
                eval_res = self.evaluator.evaluate(
                    prediction=conf, ground_truth=resolution.outcome, subject_id=question.id
                )
                results.append(eval_res)
                total_score += eval_res.score
            except Exception as e:
                logger.error(f"Failed to evaluate question {question.id}: {e}")

        count = len(results)
        mean_score = total_score / count if count > 0 else 0.0

        return EvaluationReport(
            metric_name="Brier Score", mean_score=mean_score, total_evaluations=count, results=results
        )


__all__ = ["Backtester"]
