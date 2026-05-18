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

from datetime import datetime, timezone
from pathlib import Path

from xrtm.eval.core.eval.benchmark_artifacts import (
    BenchmarkComparisonSnapshot,
    BenchmarkScoreSummary,
    ExternalComparisonRecord,
    ExternalLeaderboardEntry,
    ExternalLeaderboardSnapshot,
    InspectableOutputReference,
)

from xrtm.train.simulation.benchmark_artifacts import (
    BenchmarkRunResultBundle,
    BenchmarkRunSpec,
    BenchmarkSuiteArmResult,
    BenchmarkSuiteArmSpec,
    BenchmarkSuiteResult,
    BenchmarkSuiteSpec,
    ExternalBenchmarkLaneResult,
    ExternalBenchmarkLaneSpec,
    ExternalBenchmarkSourceSpec,
)


def test_benchmark_run_result_bundle_tracks_duration_and_paths() -> None:
    spec = BenchmarkRunSpec(
        benchmark_id="forecastbench-offline",
        benchmark_name="ForecastBench Offline Slice",
        corpus_id="forecast-v1",
        corpus_version="1.0",
        source_mode="preview",
        split="eval",
        provider="mock",
        run_limit=3,
        iterations=2,
        output_dir=Path("runs-bench"),
        tags=["offline", "preview"],
    )

    bundle = BenchmarkRunResultBundle(
        started_at=datetime(2026, 5, 7, 8, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 7, 8, 0, 5, tzinfo=timezone.utc),
        spec=spec,
        score_summary=BenchmarkScoreSummary(
            metric_name="Brier Score",
            primary_score_name="Brier Score",
            primary_score=0.22,
            sample_size=6,
        ),
        run_ids=["run-a", "run-b"],
        artifact_paths=[Path("artifacts/report.json")],
        warnings=["preview corpus"],
    )

    assert bundle.duration_seconds == 5.0
    assert bundle.spec.corpus_id == "forecast-v1"
    assert bundle.run_ids == ["run-a", "run-b"]
    assert bundle.artifact_paths == [Path("artifacts/report.json")]
    assert bundle.warnings == ["preview corpus"]


def test_benchmark_suite_result_tracks_arms_and_comparison_rows() -> None:
    arm = BenchmarkSuiteArmSpec(arm_id="baseline", display_name="Baseline", provider="mock", tags=["control"])
    spec = BenchmarkSuiteSpec(
        suite_id="suite-001",
        benchmark_id="xrtm-real-binary-v1",
        benchmark_name="XRTM Real Binary",
        corpus_id="xrtm-real-binary-v1",
        corpus_version="1.0",
        source_mode="builtin",
        split="held-out",
        split_signature="sig-123",
        run_limit=5,
        repeat_count=3,
        baseline_arm_id="baseline",
        runs_dir=Path("runs-benchmark"),
        output_dir=Path(".cache/benchmark"),
        arms=[arm],
    )
    arm_result = BenchmarkSuiteArmResult(
        arm=arm,
        score_summary=BenchmarkScoreSummary(
            metric_name="Brier Score",
            primary_score_name="eval_brier",
            primary_score=0.21,
            sample_size=15,
        ),
        systems_summary={"mean_duration_seconds": 1.2},
    )
    suite = BenchmarkSuiteResult(
        started_at=datetime(2026, 5, 7, 8, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 7, 8, 0, 9, tzinfo=timezone.utc),
        spec=spec,
        arm_results=[arm_result],
        comparison=BenchmarkComparisonSnapshot(
            benchmark_id="xrtm-real-binary-v1",
            benchmark_name="XRTM Real Binary",
        ),
    )

    assert suite.duration_seconds == 9.0
    assert suite.spec.repeat_count == 3
    assert suite.arm_results[0].arm.arm_id == "baseline"
    assert suite.comparison is not None


def test_external_benchmark_lane_result_builds_public_scorecard_snapshot() -> None:
    spec = ExternalBenchmarkLaneSpec(
        lane_id="forecastbench-public-20260507",
        benchmark_id="forecastbench",
        benchmark_name="ForecastBench",
        output_dir=Path("runs-benchmark-review"),
        sources=[
            ExternalBenchmarkSourceSpec(
                source_id="metaculus-community",
                display_name="Metaculus Community",
                evaluation_path="public-human-baseline",
                source_name="Metaculus",
                source_url="https://www.metaculus.com/questions/",
            ),
            ExternalBenchmarkSourceSpec(
                source_id="arena-output",
                display_name="Arena Output Review",
                evaluation_path="public-inspectable-output",
                source_name="Forecast Arena",
                source_url="https://arena.example/submissions/123",
                refresh_notes="manual weekly ingestion",
            ),
        ],
        metadata={"owner": "eval-team"},
    )
    result = ExternalBenchmarkLaneResult(
        started_at=datetime(2026, 5, 7, 8, 0, tzinfo=timezone.utc),
        completed_at=datetime(2026, 5, 7, 8, 0, 3, tzinfo=timezone.utc),
        spec=spec,
        comparisons=[
            ExternalComparisonRecord(
                benchmark_id="forecastbench",
                benchmark_name="ForecastBench",
                system_id="metaculus-community",
                display_name="Metaculus Community",
                evaluation_path="public-human-baseline",
                primary_score_name="brier",
                primary_score=0.17,
                captured_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
                source_name="Metaculus",
                source_id="metaculus-community",
            ),
            ExternalComparisonRecord(
                benchmark_id="forecastbench",
                benchmark_name="ForecastBench",
                system_id="arena-output",
                display_name="Forecast Arena Submission #123",
                evaluation_path="public-inspectable-output",
                primary_score_name="brier",
                primary_score=0.15,
                captured_at=datetime(2026, 5, 7, tzinfo=timezone.utc),
                source_name="Forecast Arena",
                source_id="arena-output",
                inspectable_output=InspectableOutputReference(
                    artifact_uri="https://arena.example/submissions/123/output.jsonl",
                    artifact_format="jsonl",
                ),
            ),
        ],
        leaderboards=[
            ExternalLeaderboardSnapshot(
                benchmark_id="forecastbench",
                benchmark_name="ForecastBench",
                source_name="ForecastBench",
                entries=[
                    ExternalLeaderboardEntry(
                        system_id="xrtm",
                        display_name="XRTM",
                        rank=2,
                        score_name="brier",
                        score=0.18,
                    )
                ],
            )
        ],
        warnings=["manual ingestion"],
        metadata={"review_window": "2026-W19"},
    )

    snapshot = result.to_public_scorecard_snapshot()

    assert result.duration_seconds == 3.0
    assert result.evaluation_paths() == ["public-human-baseline", "public-inspectable-output"]
    assert result.reporting_lanes() == ["public-human-baseline", "public-inspectable-output"]
    assert snapshot.evaluation_paths() == ["public-human-baseline", "public-inspectable-output"]
    assert snapshot.rows[0].metadata["source_id"] == "metaculus-community"
    assert snapshot.rows[1].inspectable_output is not None
    assert snapshot.metadata["lane_id"] == "forecastbench-public-20260507"
    assert result.leaderboards[0].source_name == "ForecastBench"
