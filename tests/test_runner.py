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
from xrtm.data import ForecastQuestion
from xrtm.eval.core.schemas import ForecastResolution

from xrtm.train.simulation.runner import BacktestDataset, BacktestInstance, BacktestRunner


class TrackingOrchestrator:
    def __init__(self, fail_id: str | None = None) -> None:
        self.fail_id = fail_id
        self.active = 0
        self.max_active = 0

    async def run(self, state, entry_node="ingestion", **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        if state.subject_id == self.fail_id:
            raise RuntimeError("orchestrator boom")
        state.node_reports["final"] = {"confidence": 0.8}
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
    orchestrator = TrackingOrchestrator()
    runner = BacktestRunner(orchestrator=orchestrator, concurrency=3)

    report = await runner.run(_dataset(10))

    assert report.total_evaluations == 10
    assert [result.subject_id for result in report.results] == [f"q{i}" for i in range(10)]
    assert orchestrator.max_active <= 3


@pytest.mark.asyncio
async def test_backtest_runner_failed_items_do_not_leak_none_into_report():
    runner = BacktestRunner(orchestrator=TrackingOrchestrator(fail_id="q1"), concurrency=2)

    report = await runner.run(_dataset(3))

    assert report.total_evaluations == 3
    assert all(result is not None for result in report.results)
    assert report.results[1].subject_id == "q1"
    assert report.results[1].metadata["error"] == "orchestrator boom"
    assert report.reliability_bins is not None
