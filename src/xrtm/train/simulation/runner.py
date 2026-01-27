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

import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

# From xrtm-data
from xrtm.data.schemas.forecast import ForecastOutput, ForecastQuestion

# From xrtm-eval
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult, Evaluator
from xrtm.eval.kit.eval.analytics import SliceAnalytics
from xrtm.eval.kit.eval.metrics import BrierScoreEvaluator, ExpectedCalibrationErrorEvaluator
from xrtm.eval.schemas.forecast import ForecastResolution

# From xrtm-forecast (Internal)
from xrtm.forecast.core.orchestrator import Orchestrator
from xrtm.forecast.core.schemas.graph import BaseGraphState, TemporalContext

logger = logging.getLogger(__name__)


class BacktestInstance(BaseModel):
    """Represents a single instance in a backtest dataset."""

    question: ForecastQuestion
    resolution: ForecastResolution
    reference_time: datetime
    tags: Optional[List[str]] = None


class BacktestDataset(BaseModel):
    """A collection of backtest instances representing an evaluation dataset."""

    name: str = "default_backtest"
    items: List[BacktestInstance]


class BacktestRunner:
    """Executes backtests on a dataset using a provided orchestrator."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        evaluator: Optional[Evaluator] = None,
        entry_node: str = "ingestion",
        concurrency: int = 5,
    ):
        self.orchestrator = orchestrator
        self.evaluator = evaluator or BrierScoreEvaluator()
        self.entry_node = entry_node
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _run_single(self, instance: BacktestInstance) -> EvaluationResult:
        async with self.semaphore:
            state = BaseGraphState(
                subject_id=instance.question.id,
                temporal_context=TemporalContext(reference_time=instance.reference_time, is_backtest=True),
            )
            state.context["question_title"] = instance.question.title
            if instance.question.content:
                state.context["question_content"] = instance.question.content

            try:
                await self.orchestrator.run(state, entry_node=self.entry_node)
                return self.evaluate_state(
                    state, instance.resolution, instance.question.id, instance.reference_time, instance.tags
                )
            except Exception as e:
                logger.error(f"Backtest error on {instance.question.id}: {e}")
                return EvaluationResult(
                    subject_id=instance.question.id,
                    score=1.0,
                    ground_truth=instance.resolution.outcome,
                    prediction=0.5,
                    metadata={"error": str(e)},
                )

    def evaluate_state(
        self,
        state: BaseGraphState,
        resolution: ForecastResolution,
        subject_id: str,
        reference_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
    ) -> EvaluationResult:
        prediction_val = 0.5
        for report in reversed(list(state.node_reports.values())):
            if isinstance(report, ForecastOutput):
                prediction_val = report.confidence
                break
            elif isinstance(report, dict) and "confidence" in report:
                prediction_val = float(report["confidence"])
                break
            elif isinstance(report, (int, float)):
                prediction_val = float(report)
                break

        outcome_raw = resolution.outcome
        if isinstance(outcome_raw, str):
            if outcome_raw.lower() in ["true", "yes", "1", "pass"]:
                gt_val = 1.0
            elif outcome_raw.lower() in ["false", "no", "0", "fail"]:
                gt_val = 0.0
            else:
                try:
                    gt_val = float(outcome_raw)
                except ValueError:
                    gt_val = 0.0
        else:
            gt_val = float(outcome_raw)

        eval_res = self.evaluator.evaluate(prediction=prediction_val, ground_truth=gt_val, subject_id=subject_id)
        if reference_time:
            eval_res.metadata["reference_time"] = reference_time.isoformat()
        eval_res.metadata["total_latency"] = sum(state.latencies.values())
        if tags:
            eval_res.metadata["tags"] = tags
        return eval_res

    async def run(self, dataset: BacktestDataset) -> EvaluationReport:
        tasks = [self._run_single(item) for item in dataset.items]
        results = await asyncio.gather(*tasks)
        total_score = sum(r.score for r in results)
        count = len(results)
        mean_score = total_score / count if count > 0 else 0.0
        ece_evaluator = ExpectedCalibrationErrorEvaluator()
        ece_score, ece_bins = ece_evaluator.compute_calibration_data(results)
        slices = SliceAnalytics.compute_slices(results)
        return EvaluationReport(
            metric_name=getattr(self.evaluator, "name", "Brier Score"),
            mean_score=mean_score,
            total_evaluations=count,
            results=results,
            reliability_bins=ece_bins,
            summary_statistics={"ece": ece_score},
            slices=slices,
        )


__all__ = ["BacktestInstance", "BacktestDataset", "BacktestRunner"]
