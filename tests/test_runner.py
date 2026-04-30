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
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from xrtm.data import ForecastOutput, ForecastQuestion
from xrtm.eval.core.schemas import ForecastResolution
from xrtm.forecast.core.schemas.graph import BaseGraphState

from xrtm.train.simulation.runner import BacktestDataset, BacktestInstance, BacktestRunner


class TrackingOrchestrator:
    def __init__(self, fail_id: str | None = None, delays: dict[str, float] | None = None) -> None:
        self.fail_id = fail_id
        self.delays = delays or {}
        self.active = 0
        self.max_active = 0

    async def run(self, state, entry_node="ingestion", **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delays.get(state.subject_id, 0.01))
        self.active -= 1
        if state.subject_id == self.fail_id:
            raise RuntimeError("orchestrator boom")
        state.node_reports["final"] = {"confidence": 0.8, "reasoning": f"reasoning for {state.subject_id}"}
        return state


def _dataset(size: int) -> BacktestDataset:
    reference_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return BacktestDataset(
        items=[
            BacktestInstance(
                question=ForecastQuestion(id=f"q{i}", title=f"Question {i}"),
                resolution=ForecastResolution(question_id=f"q{i}", outcome="yes"),
                reference_time=reference_time,
            )
            for i in range(size)
        ]
    )


@pytest.mark.asyncio
async def test_backtest_runner_limits_active_tasks_and_preserves_order():
    delays = {f"q{i}": 0.001 * (10 - i) for i in range(10)}
    orchestrator = TrackingOrchestrator(delays=delays)
    runner = BacktestRunner(orchestrator=orchestrator, concurrency=3)

    report = await runner.run(_dataset(10))

    assert report.total_evaluations == 10
    assert [result.subject_id for result in report.results] == [f"q{i}" for i in range(10)]
    assert orchestrator.max_active <= 3
    assert report.results[0].metadata["prediction_payload"]["reasoning"] == "reasoning for q0"


@pytest.mark.asyncio
async def test_backtest_runner_failed_items_do_not_leak_none_into_report():
    runner = BacktestRunner(orchestrator=TrackingOrchestrator(fail_id="q1"), concurrency=2)

    report = await runner.run(_dataset(3))

    assert report.total_evaluations == 3
    assert all(result is not None for result in report.results)
    assert report.results[1].subject_id == "q1"
    assert report.results[1].metadata["error"] == "orchestrator boom"
    assert report.results[1].metadata["resolution_payload"]["question_id"] == "q1"
    assert report.reliability_bins is not None


def test_backtest_instance_validates_resolution_question_id() -> None:
    with pytest.raises(ValidationError, match="does not match forecast question"):
        BacktestInstance(
            question=ForecastQuestion(id="q1", title="Question 1"),
            resolution=ForecastResolution(question_id="q2", outcome="yes"),
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_backtest_runner_preserves_forecast_output_payload() -> None:
    snapshot = datetime(2026, 1, 1, tzinfo=timezone.utc)
    state = BaseGraphState(
        subject_id="q1",
        node_reports={
            "final_forecast": ForecastOutput(
                question_id="q1",
                probability=0.7,
                reasoning="detailed forecast rationale",
                structural_trace=["ingestion", "forecast"],
            )
        },
    )
    runner = BacktestRunner(orchestrator=TrackingOrchestrator())

    result = runner.evaluate_state(
        state,
        ForecastResolution(question_id="q1", outcome="yes"),
        "q1",
        snapshot,
        tags=["payload"],
    )

    assert result.prediction == pytest.approx(0.7)
    assert result.metadata["prediction_node"] == "final_forecast"
    assert result.metadata["prediction_payload"]["reasoning"] == "detailed forecast rationale"
    assert result.metadata["prediction_payload"]["structural_trace"] == ["ingestion", "forecast"]
    assert result.metadata["resolution_payload"]["outcome"] == "yes"


def test_backtest_runner_rejects_unrecognized_outcome_strings() -> None:
    state = BaseGraphState(subject_id="q1", node_reports={"final": {"confidence": 0.7}})
    runner = BacktestRunner(orchestrator=TrackingOrchestrator())

    with pytest.raises(ValueError, match="Unsupported binary outcome string"):
        runner.evaluate_state(
            state,
            ForecastResolution(question_id="q1", outcome="pending"),
            "q1",
        )
