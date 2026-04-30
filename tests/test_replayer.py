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

import pytest
from xrtm.data import ForecastOutput
from xrtm.forecast.core.schemas.graph import BaseGraphState, TemporalContext

from xrtm.train.simulation.replayer import TraceReplayer


def _state() -> BaseGraphState:
    return BaseGraphState(
        subject_id="q1",
        temporal_context=TemporalContext(reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc), is_backtest=True),
        node_reports={
            "final": ForecastOutput(question_id="q1", probability=0.8, reasoning="test"),
        },
    )


@pytest.mark.asyncio
async def test_async_trace_save_load_and_replay_preserve_sync_api(tmp_path):
    path = tmp_path / "trace.json"

    TraceReplayer.save_trace(_state(), str(path))
    loaded_sync = TraceReplayer.load_trace(str(path))
    assert loaded_sync.subject_id == "q1"

    await TraceReplayer.save_trace_async(_state(), str(path))
    loaded_async = await TraceReplayer.load_trace_async(str(path))
    assert loaded_async.subject_id == "q1"

    replayer = TraceReplayer()
    sync_result = replayer.replay_evaluation(str(path), "yes")
    async_result = await replayer.replay_evaluation_async(str(path), "yes")

    assert sync_result.score == pytest.approx(0.04)
    assert async_result.score == pytest.approx(sync_result.score)
    assert async_result.metadata["is_replay"] is True
