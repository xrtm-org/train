import json

import pytest
from xrtm.forecast.core.schemas.graph import BaseGraphState

from xrtm.train.simulation.replayer import TraceReplayer


@pytest.mark.asyncio
async def test_save_trace_async(tmp_path):
    # Setup
    path = tmp_path / "test_trace.json"
    state = BaseGraphState(
        subject_id="test_subject",
        context={"foo": "bar"}
    )

    # Execute
    await TraceReplayer.save_trace(state, str(path))

    # Verify
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["subject_id"] == "test_subject"
    assert data["context"]["foo"] == "bar"

@pytest.mark.asyncio
async def test_save_trace_error_handling(tmp_path):
    # Setup - use a directory as file path to trigger error
    path = tmp_path / "test_dir"
    path.mkdir()
    state = BaseGraphState(subject_id="test")

    # Execute & Verify
    with pytest.raises(Exception):
        await TraceReplayer.save_trace(state, str(path))
