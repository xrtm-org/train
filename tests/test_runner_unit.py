import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from xrtm.data.schemas.forecast import ForecastOutput, ForecastQuestion
from xrtm.eval.core.eval.definitions import EvaluationReport, EvaluationResult
from xrtm.eval.schemas.forecast import ForecastResolution

from xrtm.train.simulation.runner import BacktestDataset, BacktestInstance, BacktestRunner


@pytest.mark.asyncio
async def test_backtest_runner_run_basic():
    """Verify that BacktestRunner correctly runs tasks and returns report."""

    # Mock Orchestrator
    orchestrator = MagicMock()

    async def mock_run(state, entry_node):
        # Return a simple confidence
        state.node_reports = {
            "final": ForecastOutput(
                confidence=0.7, explanation="test", question_id="test_q", reasoning="test_reasoning"
            )
        }
        state.latencies = {"node1": 0.1}

    orchestrator.run = AsyncMock(side_effect=mock_run)

    # Mock Evaluator
    evaluator = MagicMock()
    evaluator.evaluate.side_effect = lambda prediction, ground_truth, subject_id: EvaluationResult(
        subject_id=subject_id, score=(prediction - ground_truth) ** 2, ground_truth=ground_truth, prediction=prediction
    )
    evaluator.name = "TestEvaluator"

    runner = BacktestRunner(orchestrator=orchestrator, evaluator=evaluator, concurrency=2)

    # Create dataset
    items = []
    for i in range(5):
        q = MagicMock(spec=ForecastQuestion)
        q.id = f"q{i}"
        q.title = f"Question {i}"
        q.content = None

        r = MagicMock(spec=ForecastResolution)
        r.outcome = 1.0

        items.append(BacktestInstance(question=q, resolution=r, reference_time=datetime.now()))

    dataset = BacktestDataset(items=items)

    report = await runner.run(dataset)

    assert isinstance(report, EvaluationReport)
    assert report.total_evaluations == 5
    assert len(report.results) == 5
    # Verify results order matches
    for i, res in enumerate(report.results):
        assert res.subject_id == f"q{i}"


@pytest.mark.asyncio
async def test_backtest_runner_concurrency_limit():
    """Verify that BacktestRunner respects concurrency limit."""

    # Mock Orchestrator
    orchestrator = MagicMock()

    active_tasks = 0
    max_active_tasks = 0

    async def mock_run(state, entry_node):
        nonlocal active_tasks, max_active_tasks
        active_tasks += 1
        max_active_tasks = max(max_active_tasks, active_tasks)
        await asyncio.sleep(0.01)
        active_tasks -= 1

        state.node_reports = {
            "final": ForecastOutput(
                confidence=0.7, explanation="test", question_id="test_q", reasoning="test_reasoning"
            )
        }
        state.latencies = {"node1": 0.1}

    orchestrator.run = AsyncMock(side_effect=mock_run)

    concurrency = 3
    runner = BacktestRunner(orchestrator=orchestrator, concurrency=concurrency)

    # Create dataset with more items than concurrency
    items = []
    for i in range(10):
        q = MagicMock(spec=ForecastQuestion)
        q.id = f"q{i}"
        q.title = f"Question {i}"
        q.content = None
        r = MagicMock(spec=ForecastResolution)
        r.outcome = 1.0
        items.append(BacktestInstance(question=q, resolution=r, reference_time=datetime.now()))

    dataset = BacktestDataset(items=items)

    await runner.run(dataset)

    # Check that we didn't exceed concurrency
    # Note: _run_single uses a semaphore, so even without our fix,
    # execution concurrency is limited. But we want to ensure tasks are created.
    # The semaphore test here checks execution concurrency.
    assert max_active_tasks <= concurrency
