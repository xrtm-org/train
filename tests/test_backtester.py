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
from unittest.mock import AsyncMock, MagicMock

import pytest
from xrtm.data import ForecastOutput, ForecastQuestion
from xrtm.eval import EvaluationResult
from xrtm.eval.core.schemas import ForecastResolution

from xrtm.train import Backtester


@pytest.mark.asyncio
async def test_backtester_flow():
    """Verify that Backtester correctly calls agent and evaluator."""

    # Mocks
    mock_agent = AsyncMock()
    mock_evaluator = MagicMock()

    # Setup Data
    question = ForecastQuestion(id="q1", title="Question 1")
    resolution = ForecastResolution(question_id="q1", outcome="yes")

    prediction = ForecastOutput(question_id="q1", probability=0.8, reasoning="full reasoning payload")
    mock_agent.run.return_value = prediction

    eval_result = EvaluationResult(subject_id="q1", score=0.04, ground_truth="yes", prediction=0.8)
    mock_evaluator.evaluate.return_value = eval_result

    # Run Backtester
    backtester = Backtester(agent=mock_agent, evaluator=mock_evaluator)
    report = await backtester.run([(question, resolution)])

    # Assertions
    assert report.total_evaluations == 1
    assert report.mean_score == 0.04
    assert report.results[0].metadata["prediction_payload"]["reasoning"] == "full reasoning payload"
    assert report.results[0].metadata["resolution_payload"]["question_id"] == "q1"

    mock_agent.run.assert_awaited_once_with(question)
    mock_evaluator.evaluate.assert_called_once_with(prediction=0.8, ground_truth="yes", subject_id="q1")


@pytest.mark.asyncio
async def test_backtester_limits_concurrency():
    max_active = 0
    active = 0

    class Agent:
        async def run(self, question):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return 0.8

    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda prediction, ground_truth, subject_id: EvaluationResult(
        subject_id=subject_id,
        score=0.04,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    dataset = []
    for i in range(10):
        question = ForecastQuestion(id=f"q{i}", title=f"Question {i}")
        resolution = ForecastResolution(question_id=f"q{i}", outcome="yes")
        dataset.append((question, resolution))

    report = await Backtester(agent=Agent(), evaluator=evaluator, concurrency=3).run(dataset)

    assert report.total_evaluations == 10
    assert max_active <= 3


@pytest.mark.asyncio
async def test_backtester_preserves_result_order_with_failures():
    class Agent:
        async def run(self, question):
            if question.id == "q1":
                raise RuntimeError("boom")
            await asyncio.sleep(0.01 * (3 - int(question.id[1:])))
            return 0.8

    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda prediction, ground_truth, subject_id: EvaluationResult(
        subject_id=subject_id,
        score=0.04,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    dataset = []
    for i in range(3):
        question = ForecastQuestion(id=f"q{i}", title=f"Question {i}")
        resolution = ForecastResolution(question_id=f"q{i}", outcome="yes")
        dataset.append((question, resolution))

    report = await Backtester(agent=Agent(), evaluator=evaluator, concurrency=2).run(dataset)

    assert report.total_evaluations == 3
    assert [result.subject_id for result in report.results] == ["q0", "q1", "q2"]
    assert report.results[1].metadata["error"] == "boom"


@pytest.mark.asyncio
async def test_backtester_validates_resolution_schema() -> None:
    class Agent:
        async def run(self, question):
            return 0.8

    question = ForecastQuestion(id="q-valid", title="Question")
    resolution = ForecastResolution(question_id="other", outcome="yes")
    evaluator = MagicMock()

    report = await Backtester(agent=Agent(), evaluator=evaluator).run([(question, resolution)])

    assert report.total_evaluations == 1
    assert report.results[0].subject_id == "q-valid"
    assert report.results[0].ground_truth is None
    assert "does not match forecast question" in report.results[0].metadata["error"]
    assert report.results[0].metadata["resolution_payload"]["question_id"] == "other"
    assert report.results[0].metadata["resolution_payload"]["outcome"] == "yes"
    evaluator.evaluate.assert_not_called()
