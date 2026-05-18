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

r"""Typed benchmark run artifacts for xrtm-train orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from xrtm.eval.core.eval.benchmark_artifacts import (
    BenchmarkComparisonSnapshot,
    BenchmarkScoreSummary,
    ExternalBenchmarkReportingLane,
    ExternalComparisonRecord,
    ExternalLeaderboardSnapshot,
    PublicScorecardSnapshot,
)


class BenchmarkRunSpec(BaseModel):
    """Immutable specification for one benchmark execution plan."""

    benchmark_id: str
    benchmark_name: str
    corpus_id: str
    corpus_version: str
    source_mode: str
    split: str = "full"
    provider: str
    model: Optional[str] = None
    strategy_id: Optional[str] = None
    run_limit: int = Field(ge=1)
    iterations: int = Field(default=1, ge=1)
    release_gate_mode: bool = False
    output_dir: Optional[Path] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunResultBundle(BaseModel):
    """Stored result bundle for a completed benchmark run."""

    schema_version: str = "xrtm.benchmark-run.v1"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spec: BenchmarkRunSpec
    score_summary: BenchmarkScoreSummary
    run_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[Path] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    external_submission_ref: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the execution duration in seconds."""
        return max(0.0, (self.completed_at - self.started_at).total_seconds())


class BenchmarkSuiteArmSpec(BaseModel):
    """Typed definition for one arm inside a repeated stress suite."""

    arm_id: str
    display_name: str
    provider: str
    model: Optional[str] = None
    max_tokens: int = Field(default=768, ge=1)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuiteSpec(BaseModel):
    """Immutable specification for a repeated comparative benchmark suite."""

    schema_version: str = "xrtm.benchmark-suite-spec.v1"
    suite_id: str
    benchmark_id: str
    benchmark_name: str
    corpus_id: str
    corpus_version: str
    source_mode: str
    split: str = "full"
    split_signature: Optional[str] = None
    run_limit: int = Field(ge=1)
    repeat_count: int = Field(default=1, ge=1)
    release_gate_mode: bool = False
    baseline_arm_id: Optional[str] = None
    runs_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    arms: list[BenchmarkSuiteArmSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuiteArmResult(BaseModel):
    """Aggregated result for one arm across repeated benchmark executions."""

    arm: BenchmarkSuiteArmSpec
    score_summary: BenchmarkScoreSummary
    runs: list[BenchmarkRunResultBundle] = Field(default_factory=list)
    systems_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkSuiteResult(BaseModel):
    """Stored result bundle for a completed benchmark stress suite."""

    schema_version: str = "xrtm.benchmark-suite-result.v1"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spec: BenchmarkSuiteSpec
    arm_results: list[BenchmarkSuiteArmResult] = Field(default_factory=list)
    comparison: Optional[BenchmarkComparisonSnapshot] = None
    artifact_paths: list[Path] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the suite duration in seconds."""
        return max(0.0, (self.completed_at - self.started_at).total_seconds())


class ExternalBenchmarkSourceSpec(BaseModel):
    """Typed definition for one public benchmark source to ingest for reporting."""

    model_config = ConfigDict(populate_by_name=True)

    source_id: str
    display_name: str
    evaluation_path: ExternalBenchmarkReportingLane = Field(
        ...,
        validation_alias=AliasChoices("evaluation_path", "reporting_lane"),
    )
    source_name: str
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    scoring_rule: Optional[str] = None
    refresh_notes: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reporting_lane(self) -> ExternalBenchmarkReportingLane:
        """Backward compatibility alias for ``evaluation_path``."""
        return self.evaluation_path

    @reporting_lane.setter
    def reporting_lane(self, value: ExternalBenchmarkReportingLane) -> None:
        """Backward compatibility setter for ``evaluation_path``."""
        self.evaluation_path = value


class ExternalBenchmarkLaneSpec(BaseModel):
    """Immutable plan for the public external-comparison ingestion lane."""

    schema_version: str = "xrtm.external-benchmark-lane-spec.v1"
    lane_id: str
    benchmark_id: str
    benchmark_name: str
    output_dir: Optional[Path] = None
    sources: list[ExternalBenchmarkSourceSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExternalBenchmarkLaneResult(BaseModel):
    """Stored artifact bundle for public baselines and external benchmark references."""

    schema_version: str = "xrtm.external-benchmark-lane-result.v1"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spec: ExternalBenchmarkLaneSpec
    comparisons: list[ExternalComparisonRecord] = Field(default_factory=list)
    leaderboards: list[ExternalLeaderboardSnapshot] = Field(default_factory=list)
    artifact_paths: list[Path] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Return the ingestion duration in seconds."""
        return max(0.0, (self.completed_at - self.started_at).total_seconds())

    def reporting_lanes(self) -> list[str]:
        """Return the distinct public comparison lanes represented in this result."""
        return self.evaluation_paths()

    def evaluation_paths(self) -> list[str]:
        """Return the distinct public evaluation paths represented in this result."""
        return sorted({_comparison_evaluation_path(comparison) for comparison in self.comparisons})

    def to_public_scorecard_snapshot(
        self,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> PublicScorecardSnapshot:
        """Render imported public references into the public scorecard contract."""
        snapshot_metadata = {
            "benchmark_id": self.spec.benchmark_id,
            "benchmark_name": self.spec.benchmark_name,
            "lane_id": self.spec.lane_id,
            **self.spec.metadata,
            **self.metadata,
        }
        if metadata:
            snapshot_metadata.update(metadata)
        return PublicScorecardSnapshot(
            rows=[comparison.to_scorecard_row() for comparison in self.comparisons],
            metadata=snapshot_metadata,
        )


def _comparison_evaluation_path(comparison: ExternalComparisonRecord) -> str:
    path = getattr(comparison, "evaluation_path", None)
    if path is None:
        path = getattr(comparison, "reporting_lane")
    return path


__all__ = [
    "BenchmarkRunSpec",
    "BenchmarkRunResultBundle",
    "BenchmarkSuiteArmSpec",
    "BenchmarkSuiteSpec",
    "BenchmarkSuiteArmResult",
    "BenchmarkSuiteResult",
    "ExternalBenchmarkSourceSpec",
    "ExternalBenchmarkLaneSpec",
    "ExternalBenchmarkLaneResult",
]
