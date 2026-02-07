# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from xrtm.forecast.kit.memory.unified import Memory as UnifiedMemory

from xrtm.train.kit.memory.learner import EpisodicLearner


@pytest.mark.asyncio
async def test_get_lessons_for_subject_success():
    """Verify that get_lessons_for_subject correctly retrieves and formats lessons."""
    # Setup Mock Memory
    mock_memory = MagicMock(spec=UnifiedMemory)
    mock_memory.retrieve_similar.return_value = ["Lesson 1", "Lesson 2"]

    learner = EpisodicLearner(memory=mock_memory)

    # Execute
    result = await learner.get_lessons_for_subject("subject_1")

    # Verify
    assert "--- PAST LESSONS LEARNED ---" in result
    assert "Experience 1:" in result
    assert "Lesson 1" in result
    assert "Experience 2:" in result
    assert "Lesson 2" in result

    # Verify memory call
    mock_memory.retrieve_similar.assert_called_once()
    args, kwargs = mock_memory.retrieve_similar.call_args
    # The first arg is query, checking if subject_1 is in it
    assert "subject_1" in args[0]
    assert kwargs.get("n_results") == 3


@pytest.mark.asyncio
async def test_get_lessons_for_subject_empty():
    """Verify that get_lessons_for_subject handles empty results."""
    # Setup Mock Memory
    mock_memory = MagicMock(spec=UnifiedMemory)
    mock_memory.retrieve_similar.return_value = []

    learner = EpisodicLearner(memory=mock_memory)

    # Execute
    result = await learner.get_lessons_for_subject("subject_1")

    # Verify
    assert result == "No relevant past experiences found."


@pytest.mark.asyncio
async def test_get_lessons_for_subject_error():
    """Verify that get_lessons_for_subject handles exceptions."""
    # Setup Mock Memory
    mock_memory = MagicMock(spec=UnifiedMemory)
    mock_memory.retrieve_similar.side_effect = Exception("Database error")

    learner = EpisodicLearner(memory=mock_memory)

    # Execute
    result = await learner.get_lessons_for_subject("subject_1")

    # Verify
    assert result == "Error retrieving past lessons."

@pytest.mark.asyncio
async def test_get_lessons_for_subject_concurrency():
    """Verify that get_lessons_for_subject runs concurrently."""
    # Setup Mock Memory with delay
    mock_memory = MagicMock(spec=UnifiedMemory)
    def delayed_retrieve(*args, **kwargs):
        time.sleep(0.1)
        return ["Lesson"]
    mock_memory.retrieve_similar.side_effect = delayed_retrieve

    learner = EpisodicLearner(memory=mock_memory)

    start_time = time.time()
    # Run 5 calls concurrently
    tasks = [learner.get_lessons_for_subject(f"subject_{i}") for i in range(5)]
    await asyncio.gather(*tasks)
    end_time = time.time()

    duration = end_time - start_time
    # If sequential, it would be ~0.5s. If concurrent, ~0.1s + overhead.
    # Allowing some overhead, but it should be significantly less than 0.5s.
    assert duration < 0.3, f"Execution took {duration}s, expected < 0.3s (concurrent execution)"
