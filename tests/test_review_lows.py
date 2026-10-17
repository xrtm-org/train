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

r"""Regression tests for the train Low batch (validation + safe serialization)."""

import json
from datetime import datetime, timezone
from enum import Enum

import pytest

from xrtm.train.kit.builders import TrainingSampleBuilder
from xrtm.train.simulation.artifacts import _json_safe, normalize_binary_outcome


@pytest.mark.parametrize(
    "raw,expected",
    [("yes", 1.0), ("NO", 0.0), ("true", 1.0), ("false", 0.0), (True, 1.0), (False, 0.0), (0, 0.0), (1.0, 1.0), ("1.0", 1.0)],
)
def test_normalize_binary_outcome_accepts_binary_values(raw, expected):
    assert normalize_binary_outcome(raw) == expected


@pytest.mark.parametrize("raw", [0.7, "0.7", "maybe", None, object()])
def test_normalize_binary_outcome_rejects_non_binary_values(raw):
    with pytest.raises(ValueError):
        normalize_binary_outcome(raw)


def test_json_safe_converts_sets_enums_and_opaque_objects():
    class Colour(Enum):
        RED = "red"

    class Opaque:
        pass

    safe = _json_safe({"s": {2, 1}, "e": Colour.RED, "o": Opaque()})

    assert safe["s"] == [1, 2]
    assert safe["e"] == "red"
    assert isinstance(safe["o"], str)
    json.dumps(safe)  # must not raise


def test_build_sequence_length_mismatch_raises_even_for_tiny_inputs():
    builder = TrainingSampleBuilder()

    with pytest.raises(ValueError, match="same length"):
        builder.build_sequence(
            question_id="q1",
            news_events=[object(), object(), object()],
            prior_snapshots=[object()],
            deadline=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_excluded_from_aggregates_covers_failures_and_abstentions():
    from types import SimpleNamespace

    from xrtm.train.simulation.artifacts import excluded_from_aggregates

    assert excluded_from_aggregates(SimpleNamespace(metadata={"abstained": True})) is True
    assert excluded_from_aggregates(SimpleNamespace(metadata={"error": "boom"})) is True
    assert excluded_from_aggregates(SimpleNamespace(metadata={})) is False
    assert excluded_from_aggregates(SimpleNamespace(metadata=None)) is False


@pytest.mark.asyncio
async def test_backtester_excludes_abstained_rows_from_mean_score():
    from unittest.mock import AsyncMock, MagicMock

    from xrtm.data import ForecastOutput, ForecastQuestion
    from xrtm.eval import EvaluationResult
    from xrtm.eval.core.schemas import ForecastResolution

    from xrtm.train import Backtester

    agent = AsyncMock()
    evaluator = MagicMock()
    question_a = ForecastQuestion(id="qa", title="A")
    question_b = ForecastQuestion(id="qb", title="B")
    resolution_a = ForecastResolution(question_id="qa", outcome="yes")
    resolution_b = ForecastResolution(question_id="qb", outcome="no")
    agent.run.return_value = ForecastOutput.model_validate(
        {
            "forecast_request_id": "qa",
            "probability": 0.8,
            "reasoning_trace": {"narrative": "r"},
            "metadata": {"source_version": "test"},
        }
    )

    def evaluate(prediction, ground_truth, subject_id):
        if subject_id == "qb":
            # A row the evaluator could not use: 0.0 must not flatter the mean.
            return EvaluationResult(
                subject_id="qb",
                score=0.0,
                ground_truth=None,
                prediction=None,
                metadata={"metric": "Brier Score", "abstained": True, "reason": "unusable"},
            )
        return EvaluationResult(
            subject_id="qa",
            score=0.04,
            ground_truth=1.0,
            prediction=0.8,
            metadata={"metric": "Brier Score"},
        )

    evaluator.evaluate.side_effect = evaluate

    report = await Backtester(agent=agent, evaluator=evaluator).run(
        [(question_a, resolution_a), (question_b, resolution_b)]
    )

    assert report.mean_score == pytest.approx(0.04)
    assert report.summary_statistics["abstained"] == 1
    assert report.summary_statistics["errors"] == 0
