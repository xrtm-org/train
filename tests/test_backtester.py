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

from xrtm.train import Backtester


@pytest.mark.asyncio
async def test_backtester_flow():
    """Verify that Backtester correctly calls agent and evaluator."""

    # Mocks
    mock_agent = AsyncMock()
    mock_evaluator = MagicMock()

    # Setup Data
    question = MagicMock(spec=ForecastQuestion)
    question.id = "q1"
    resolution = MagicMock()
    resolution.outcome = 1

    prediction = MagicMock(spec=ForecastOutput)
    prediction.confidence = 0.8
    mock_agent.run.return_value = prediction

    eval_result = EvaluationResult(subject_id="q1", score=0.04, ground_truth=1, prediction=0.8)
    mock_evaluator.evaluate.return_value = eval_result

    # Run Backtester
    backtester = Backtester(agent=mock_agent, evaluator=mock_evaluator)
    report = await backtester.run([(question, resolution)])

    # Assertions
    assert report.total_evaluations == 1
    assert report.mean_score == 0.04

    mock_agent.run.assert_awaited_once_with(question)
    mock_evaluator.evaluate.assert_called_once()


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
        question = MagicMock(spec=ForecastQuestion)
        question.id = f"q{i}"
        resolution = MagicMock()
        resolution.outcome = 1
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
        question = MagicMock(spec=ForecastQuestion)
        question.id = f"q{i}"
        resolution = MagicMock()
        resolution.outcome = 1
        dataset.append((question, resolution))

    report = await Backtester(agent=Agent(), evaluator=evaluator, concurrency=2).run(dataset)

    assert [result.subject_id for result in report.results] == ["q0", "q2"]
