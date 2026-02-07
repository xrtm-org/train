
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from xrtm.train.kit.optimization.compiler import BrierOptimizer
from xrtm.forecast.kit.agents.prompting import CompiledAgent, PromptTemplate

@pytest.fixture
def mock_agent():
    template = PromptTemplate(instruction="Current instruction", examples=[], version=1)
    return CompiledAgent(template=template, model=MagicMock())

@pytest.fixture
def mock_dataset():
    # dataset is List[Tuple[Any, float, int]] -> input, prediction, ground_truth
    # We create a dataset where Brier score is predictable.
    # (0.8 - 1)^2 = 0.04
    # (0.2 - 0)^2 = 0.04
    # Mean = 0.04
    return [
        ("input1", 0.8, 1),
        ("input2", 0.2, 0)
    ]

@pytest.mark.asyncio
async def test_optimize_calculation(mock_agent, mock_dataset):
    # Mock optimizer model
    optimizer_model = MagicMock()
    optimizer_model.generate = AsyncMock(return_value="New instruction")

    optimizer = BrierOptimizer(optimizer_model)

    new_template = await optimizer.optimize(mock_agent, mock_dataset)

    assert new_template.instruction == "New instruction"
    assert new_template.version == 2

    # Verify the prompt sent to the model contains the correct error
    # We can inspect the call args to verify calculation was done correctly
    # Total error = ((0.8-1)**2 + (0.2-0)**2) / 2 = (0.04 + 0.04) / 2 = 0.04

    call_args = optimizer_model.generate.call_args
    assert call_args is not None
    prompt = call_args[0][0]

    assert "Current Goal: Minimize Brier Score" in prompt
    assert "Current: 0.0400" in prompt

@pytest.mark.asyncio
async def test_optimize_empty_dataset(mock_agent):
    optimizer_model = MagicMock()
    optimizer_model.generate = AsyncMock(return_value="New instruction")
    optimizer = BrierOptimizer(optimizer_model)

    # Original implementation would raise ZeroDivisionError for empty dataset
    # We preserved this behavior.
    with pytest.raises(ZeroDivisionError):
        await optimizer.optimize(mock_agent, [])
