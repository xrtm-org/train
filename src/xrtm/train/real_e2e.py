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

r"""Train/backtest utilities for deterministic real-question forecast artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from xrtm.data.corpora import load_real_binary_corpus, load_real_binary_questions
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult
from xrtm.eval.core.schemas import ForecastResolution
from xrtm.eval.kit.eval.analytics import SliceAnalytics
from xrtm.eval.kit.eval.metrics import BrierScoreEvaluator, ExpectedCalibrationErrorEvaluator
from xrtm.eval.real_e2e import coerce_forecast_outputs
from xrtm.forecast.core.orchestrator import Orchestrator
from xrtm.forecast.core.schemas.graph import BaseGraphState, TemporalContext

from xrtm.train.kit.builders import BetaPriorSnapshot, NewsEvent, TrainingSample, TrainingSampleBuilder
from xrtm.train.simulation.runner import BacktestDataset, BacktestInstance, BacktestRunner


def build_resolved_backtest_dataset(
    records: Iterable[Any],
    *,
    name: str = "real_binary_forecast_artifacts",
) -> BacktestDataset:
    r"""Build a train ``BacktestDataset`` for resolved real-question forecasts."""
    outputs = coerce_forecast_outputs(records)
    corpus_by_id = {record.id: record for record in load_real_binary_corpus()}
    questions_by_id = {question.id: question for question in load_real_binary_questions()}
    items: list[BacktestInstance] = []

    for output in outputs:
        corpus_record = corpus_by_id.get(output.question_id)
        question = questions_by_id.get(output.question_id)
        if corpus_record is None or question is None or corpus_record.resolved_outcome is None:
            continue

        outcome = "yes" if corpus_record.resolved_outcome else "no"
        resolved_at = corpus_record.resolution_time or output.metadata.snapshot_time
        items.append(
            BacktestInstance(
                question=question,
                resolution=ForecastResolution(
                    question_id=output.question_id,
                    outcome=outcome,
                    resolved_at=resolved_at,
                    metadata={
                        "corpus_source": corpus_record.source,
                        "resolution_notes": corpus_record.resolution_notes,
                    },
                ),
                reference_time=output.metadata.snapshot_time,
                tags=list(output.metadata.tags),
            )
        )

    return BacktestDataset(name=name, items=items)


def evaluate_forecast_records_with_backtest_runner(
    records: Iterable[Any],
    *,
    num_bins: int = 10,
) -> EvaluationReport:
    r"""Evaluate resolved forecast records through train's existing backtest runner path."""
    outputs = coerce_forecast_outputs(records)
    output_by_id = {output.question_id: output for output in outputs}
    dataset = build_resolved_backtest_dataset(outputs)
    runner = BacktestRunner(orchestrator=Orchestrator(), evaluator=BrierScoreEvaluator())
    results: list[EvaluationResult] = []

    for item in dataset.items:
        output = output_by_id[item.question.id]
        state = BaseGraphState(
            subject_id=item.question.id,
            temporal_context=TemporalContext(reference_time=item.reference_time, is_backtest=True),
        )
        state.node_reports["forecast_output"] = output
        results.append(
            runner.evaluate_state(
                state=state,
                resolution=item.resolution,
                subject_id=item.question.id,
                reference_time=item.reference_time,
                tags=item.tags,
            )
        )

    total = len(results)
    mean_brier = sum(result.score for result in results) / total if total else 0.0
    ece, bins = ExpectedCalibrationErrorEvaluator(num_bins=num_bins).compute_calibration_data(results)
    return EvaluationReport(
        metric_name="Real Binary Forecast Brier Score",
        mean_score=mean_brier,
        total_evaluations=total,
        results=results,
        reliability_bins=bins,
        summary_statistics={"brier_score": mean_brier, "ece": ece},
        slices=SliceAnalytics.compute_slices(results),
    )


def build_training_samples_from_resolved_forecasts(
    records: Iterable[Any],
    *,
    beta_strength: float = 20.0,
    target_probability: float = 0.99,
) -> list[TrainingSample]:
    r"""Turn resolved real-question forecast records into existing training samples."""
    outputs = coerce_forecast_outputs(records)
    corpus_by_id = {record.id: record for record in load_real_binary_corpus()}
    builder = TrainingSampleBuilder(context_window_size=2)
    samples: list[TrainingSample] = []

    for output in outputs:
        corpus_record = corpus_by_id.get(output.question_id)
        if corpus_record is None or corpus_record.resolved_outcome is None:
            continue

        snapshot_time = output.metadata.snapshot_time
        resolution_time = corpus_record.resolution_time or snapshot_time
        probability = _clamp_probability(output.probability)
        target_mean = target_probability if corpus_record.resolved_outcome else 1.0 - target_probability
        samples.extend(
            builder.build_sequence(
                question_id=output.question_id,
                news_events=[
                    NewsEvent(
                        content=output.reasoning,
                        timestamp=snapshot_time,
                        source="forecast_artifact",
                    ),
                    NewsEvent(
                        content=corpus_record.resolution_notes or f"Resolved outcome: {corpus_record.resolved_outcome}",
                        timestamp=resolution_time,
                        source=corpus_record.source,
                    ),
                ],
                prior_snapshots=[
                    _beta_snapshot_from_probability(probability, beta_strength, snapshot_time),
                    _beta_snapshot_from_probability(target_mean, beta_strength, resolution_time),
                ],
                deadline=resolution_time,
                start_time=snapshot_time,
            )
        )

    return samples


def _beta_snapshot_from_probability(probability: float, strength: float, timestamp: datetime) -> BetaPriorSnapshot:
    probability = _clamp_probability(probability)
    return BetaPriorSnapshot(
        alpha=max(0.01, probability * strength),
        beta=max(0.01, (1.0 - probability) * strength),
        timestamp=timestamp,
    )


def _clamp_probability(probability: float) -> float:
    return max(0.001, min(0.999, float(probability)))


__all__ = [
    "build_resolved_backtest_dataset",
    "build_training_samples_from_resolved_forecasts",
    "evaluate_forecast_records_with_backtest_runner",
]
