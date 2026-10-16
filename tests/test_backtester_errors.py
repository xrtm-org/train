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

r"""Regression tests: malformed resolutions must not hang the backtester."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from xrtm.data import ForecastOutput, ForecastQuestion
from xrtm.eval import EvaluationResult
from xrtm.eval.core.schemas import ForecastResolution

from xrtm.train import Backtester


@pytest.mark.asyncio
async def test_backtester_survives_non_model_resolution():
    mock_agent = AsyncMock()
    mock_evaluator = MagicMock()
    question = ForecastQuestion(id="q1", title="Question 1")
    bad_resolution = cast(Any, None)  # not a ForecastResolution

    async def _run(_question):
        return ForecastOutput(forecast_request_id="q1", probability=0.8, reasoning="narrative")

    mock_agent.run.side_effect = _run

    backtester = Backtester(agent=mock_agent, evaluator=mock_evaluator)
    # wait_for guards the old failure mode: a dead worker made queue.join() hang
    report = await asyncio.wait_for(backtester.run([(question, bad_resolution)]), timeout=5)

    assert report.total_evaluations == 1
    assert report.results[0].metadata["error"]
    assert report.results[0].metadata["resolution_payload"] == {"unavailable": True}
    # the placeholder score (1.0) is excluded from the aggregate
    assert report.mean_score == 0.0
    assert report.summary_statistics["errors"] == 1
    mock_evaluator.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_backtester_mixed_valid_and_invalid():
    mock_agent = AsyncMock()
    mock_evaluator = MagicMock()
    good_question = ForecastQuestion(id="q1", title="Q1")
    bad_question = ForecastQuestion(id="q2", title="Q2")
    good_resolution = ForecastResolution(question_id="q1", outcome="yes")
    bad_resolution = cast(Any, None)

    async def _run(_question):
        return ForecastOutput(forecast_request_id="q1", probability=0.8, reasoning="narrative")

    mock_agent.run.side_effect = _run
    mock_evaluator.evaluate.return_value = EvaluationResult(
        subject_id="q1", score=0.04, ground_truth="yes", prediction=0.8
    )

    backtester = Backtester(agent=mock_agent, evaluator=mock_evaluator)
    report = await asyncio.wait_for(
        backtester.run([(good_question, good_resolution), (bad_question, bad_resolution)]), timeout=5
    )

    assert report.total_evaluations == 2
    assert report.mean_score == pytest.approx(0.04)  # only the valid result counts
    assert report.summary_statistics["errors"] == 1
