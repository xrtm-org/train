
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from xrtm.train.simulation.runner import BacktestRunner

@pytest.mark.asyncio
async def test_run_single_logging_deferred():
    # Setup
    mock_orchestrator = MagicMock()
    # Ensure run raises an exception
    test_exception = ValueError("Test error")
    mock_orchestrator.run = AsyncMock(side_effect=test_exception)

    # Mock instance
    instance = MagicMock()
    instance.question.id = "q1"
    instance.question.title = "test question"
    instance.question.content = "test content"
    instance.reference_time = datetime.now()
    instance.tags = None
    instance.resolution.outcome = 1

    runner = BacktestRunner(orchestrator=mock_orchestrator)

    # Patch logger
    with patch("xrtm.train.simulation.runner.logger") as mock_logger:
        # Act
        await runner._run_single(instance)

        # Assert
        # We expect the logging call to be deferred: logger.error("Backtest error on %s: %s", "q1", exception_object)
        mock_logger.error.assert_called_once()
        args, _ = mock_logger.error.call_args

        # Check if deferred formatting is used
        # args[0] should be the format string
        # args[1] should be the question id
        # args[2] should be the exception

        assert args[0] == "Backtest error on %s: %s", f"Expected deferred logging format, got: {args[0]}"
        assert args[1] == "q1"
        assert args[2] is test_exception
