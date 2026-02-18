import os
import tempfile
from unittest.mock import MagicMock

import pytest
from xrtm.eval.core.eval.definitions import EvaluationResult, Evaluator
from xrtm.forecast.core.schemas.graph import BaseGraphState, TemporalContext

from xrtm.train.simulation.replayer import TraceReplayer


@pytest.fixture
def sample_state():
    return BaseGraphState(
        subject_id="test-1",
        temporal_context=TemporalContext(reference_time="2025-01-01T00:00:00", is_backtest=True),
        node_reports={"test_node": "test_value"}
    )

@pytest.fixture
def trace_file(sample_state):
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(sample_state.model_dump_json())
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)

@pytest.mark.asyncio
async def test_load_trace_async(trace_file):
    state = await TraceReplayer.load_trace(trace_file)
    assert isinstance(state, BaseGraphState)
    assert state.subject_id == "test-1"

@pytest.mark.asyncio
async def test_load_trace_not_found():
    with pytest.raises(Exception):
        await TraceReplayer.load_trace("non_existent_file.json")

@pytest.mark.asyncio
async def test_replay_evaluation(trace_file):
    # Mock evaluator
    evaluator = MagicMock(spec=Evaluator)
    evaluator.evaluate.return_value = EvaluationResult(
        subject_id="test-1", score=0.5, ground_truth=1.0, prediction=0.5
    )

    replayer = TraceReplayer()
    # resolution="YES" implies ground_truth=1.0
    result = await replayer.replay_evaluation(
        trace_path=trace_file,
        resolution="YES",
        evaluator=evaluator
    )

    assert isinstance(result, EvaluationResult)
    assert result.metadata["is_replay"] is True

@pytest.mark.asyncio
async def test_save_trace(sample_state):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    # Close the file so save_trace can write to it (on Windows this matters, on Linux less so but good practice)
    f.close()

    try:
        TraceReplayer.save_trace(sample_state, path)
        loaded = await TraceReplayer.load_trace(path)
        assert loaded.subject_id == sample_state.subject_id
    finally:
        if os.path.exists(path):
            os.remove(path)
