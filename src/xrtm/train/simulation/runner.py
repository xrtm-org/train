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

r"""Backtest orchestration with bounded concurrency.

Defines the ``BacktestRunner`` that executes a ``BacktestDataset`` against an
workflow ``Orchestrator``, evaluates each forecast result using a configurable ``Evaluator``,
and aggregates results into an ``EvaluationReport`` with calibration metrics
and tag-based slice analytics.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Self

from pydantic import BaseModel, model_validator

# From xrtm-data
from xrtm.data.core.schemas.forecast import ForecastOutput, ForecastQuestion

# From xrtm-eval
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult, Evaluator
from xrtm.eval.core.schemas import ForecastResolution
from xrtm.eval.kit.eval.analytics import SliceAnalytics
from xrtm.eval.kit.eval.metrics import BrierScoreEvaluator, ExpectedCalibrationErrorEvaluator

# From xrtm-forecast (Internal)
from xrtm.forecast.core.orchestrator import Orchestrator
from xrtm.forecast.core.schemas.graph import BaseGraphState, TemporalContext

from xrtm.train.simulation.artifacts import (
    excluded_from_aggregates,
    normalize_binary_outcome,
    prediction_value_and_payload,
    resolution_payload,
    validate_resolution_for_question,
)

logger = logging.getLogger(__name__)

class BacktestInstance(BaseModel):
    """Represents a single forecast-request / resolution pair in a backtest dataset."""

    question: ForecastQuestion
    resolution: ForecastResolution
    reference_time: datetime
    tags: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_resolution_matches_question(self) -> Self:
        validate_resolution_for_question(self.resolution, self.question.id)
        return self


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
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.orchestrator = orchestrator
        self.evaluator = evaluator or BrierScoreEvaluator()
        self.entry_node = entry_node
        self.concurrency = concurrency

    async def _run_single(self, instance: BacktestInstance) -> EvaluationResult:
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
            logger.error("Backtest error on %s: %s", instance.question.id, e)
            return EvaluationResult(
                subject_id=instance.question.id,
                score=1.0,
                ground_truth=instance.resolution.outcome,
                prediction=0.5,
                metadata={"error": str(e), "resolution_payload": resolution_payload(instance.resolution)},
            )

    def evaluate_state(
        self,
        state: BaseGraphState,
        resolution: ForecastResolution,
        subject_id: str,
        reference_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
    ) -> EvaluationResult:
        validated_resolution = validate_resolution_for_question(resolution, subject_id)
        prediction_val = 0.5
        prediction_payload = None
        prediction_node = None
        # Prefer an explicit designated output node; otherwise take the most
        # recent report. ``bool`` is not a numeric prediction.
        output_node = state.context.get("output_node") if isinstance(state.context, dict) else None
        if output_node and output_node in state.node_reports:
            candidate_names = [output_node]
        else:
            candidate_names = list(reversed(state.node_reports.keys()))
        for node_name in candidate_names:
            report = state.node_reports[node_name]
            if isinstance(report, ForecastOutput):
                prediction_val, prediction_payload = prediction_value_and_payload(report)
                prediction_node = node_name
                break
            elif isinstance(report, dict) and ("probability" in report or "confidence" in report):
                prediction_val, prediction_payload = prediction_value_and_payload(report)
                prediction_node = node_name
                break
            elif isinstance(report, (int, float)) and not isinstance(report, bool):
                prediction_val, prediction_payload = prediction_value_and_payload(report)
                prediction_node = node_name
                break

        gt_val = normalize_binary_outcome(validated_resolution.outcome)

        eval_res = self.evaluator.evaluate(prediction=prediction_val, ground_truth=gt_val, subject_id=subject_id)
        eval_res.metadata["resolution_payload"] = resolution_payload(validated_resolution)
        if prediction_payload is not None:
            eval_res.metadata["prediction_payload"] = prediction_payload
        if prediction_node is not None:
            eval_res.metadata["prediction_node"] = prediction_node
        if reference_time:
            eval_res.metadata["reference_time"] = reference_time.isoformat()
        eval_res.metadata["total_latency"] = sum(state.latencies.values())
        if tags:
            eval_res.metadata["tags"] = tags
        return eval_res

    async def run(self, dataset: BacktestDataset) -> EvaluationReport:
        results: list[EvaluationResult] = []
        if dataset.items:
            queue: asyncio.Queue[tuple[int, BacktestInstance]] = asyncio.Queue()
            ordered_results: list[Optional[EvaluationResult]] = [None] * len(dataset.items)
            for idx, item in enumerate(dataset.items):
                queue.put_nowait((idx, item))

            async def worker() -> None:
                while True:
                    try:
                        idx, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        ordered_results[idx] = await self._run_single(item)
                    finally:
                        queue.task_done()

            worker_count = min(self.concurrency, len(dataset.items))
            workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
            await queue.join()
            await asyncio.gather(*workers)
            results = [result for result in ordered_results if result is not None]

        # Failures and abstentions carry placeholder scores and must not
        # pollute aggregates; count both separately.
        scored_results = [result for result in results if not excluded_from_aggregates(result)]
        total_score = sum(r.score for r in scored_results)
        count = len(scored_results)
        error_count = sum(1 for result in results if (getattr(result, "metadata", None) or {}).get("error"))
        abstained_count = len(results) - count - error_count
        mean_score = total_score / count if count > 0 else 0.0
        ece_evaluator = ExpectedCalibrationErrorEvaluator()
        ece_score, ece_bins = ece_evaluator.compute_calibration_data(scored_results)
        slices = SliceAnalytics.compute_slices(scored_results)
        return EvaluationReport(
            metric_name=getattr(self.evaluator, "name", "Brier Score"),
            mean_score=mean_score,
            total_evaluations=len(results),
            results=results,
            reliability_bins=ece_bins,
            summary_statistics={"ece": ece_score, "errors": error_count, "abstained": abstained_count},
            slices=slices,
        )


__all__ = ["BacktestInstance", "BacktestDataset", "BacktestRunner"]
