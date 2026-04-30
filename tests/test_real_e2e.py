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

import pytest
from xrtm.data.core.schemas.forecast import ForecastOutput, MetadataBase
from xrtm.data.corpora import REAL_BINARY_CORPUS_ID, load_real_binary_corpus

from xrtm.train.real_e2e import (
    build_resolved_backtest_dataset,
    build_training_samples_from_resolved_forecasts,
    evaluate_forecast_records_with_backtest_runner,
)


def _synthetic_forecast_artifacts(limit: int = 4) -> list[dict]:
    artifacts = []
    for record in load_real_binary_corpus()[:limit]:
        probability = 0.9 if record.resolved_outcome else 0.1
        output = ForecastOutput(
            question_id=record.id,
            probability=probability,
            reasoning=f"Synthetic deterministic forecast for {record.id}",
            metadata=MetadataBase(
                snapshot_time=record.snapshot_time,
                tags=["real-question-e2e", "synthetic-fixture"],
                subject_type="binary",
                source_version=REAL_BINARY_CORPUS_ID,
            ),
        )
        artifacts.append({"question_id": record.id, "output": output.model_dump(mode="json")})
    return artifacts


def test_forecast_artifacts_build_backtest_dataset_with_real_resolutions() -> None:
    dataset = build_resolved_backtest_dataset(_synthetic_forecast_artifacts(limit=3))

    assert dataset.name == "real_binary_forecast_artifacts"
    assert len(dataset.items) == 3
    assert [item.question.id for item in dataset.items] == [record.id for record in load_real_binary_corpus()[:3]]
    assert {item.resolution.outcome for item in dataset.items} <= {"yes", "no"}
    assert all("synthetic-fixture" in (item.tags or []) for item in dataset.items)


def test_train_backtest_runner_consumes_same_forecast_records_for_scoring() -> None:
    report = evaluate_forecast_records_with_backtest_runner(_synthetic_forecast_artifacts(), num_bins=2)

    assert report.total_evaluations == 4
    assert report.mean_score == pytest.approx(0.01)
    assert report.summary_statistics["brier_score"] == pytest.approx(0.01)
    assert report.summary_statistics["ece"] == pytest.approx(0.1)
    assert report.reliability_bins is not None
    assert sum(bin.count for bin in report.reliability_bins) == 4


def test_forecast_records_feed_existing_training_sample_builder() -> None:
    samples = build_training_samples_from_resolved_forecasts(_synthetic_forecast_artifacts(limit=2), beta_strength=20.0)

    assert len(samples) == 2
    assert [sample.question_id for sample in samples] == [record.id for record in load_real_binary_corpus()[:2]]
    assert samples[0].prior_mean == pytest.approx(0.9)
    assert samples[0].target_mean == pytest.approx(0.99)
    assert samples[1].prior_mean == pytest.approx(0.1)
    assert samples[1].target_mean == pytest.approx(0.01)
