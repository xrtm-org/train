import json
import os
import tempfile

import pytest
from xrtm.forecast.core.schemas.graph import BaseGraphState

from xrtm.train.simulation.replayer import TraceReplayer


@pytest.mark.asyncio
async def test_save_trace_async():
    # Create a small state
    state = BaseGraphState(
        subject_id="test_subject",
        node_reports={"foo": "bar"},
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "trace.json")
        await TraceReplayer.save_trace_async(state, path)

        assert os.path.exists(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["subject_id"] == "test_subject"
        assert data["node_reports"]["foo"] == "bar"

@pytest.mark.asyncio
async def test_save_trace_async_error():
    state = BaseGraphState(
        subject_id="test_subject",
        node_reports={"foo": "bar"},
    )
    # Test writing to an invalid path (e.g. directory that doesn't exist)
    path = "/non_existent_directory_xyz/trace.json"

    with pytest.raises(FileNotFoundError):
        await TraceReplayer.save_trace_async(state, path)
