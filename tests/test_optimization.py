from unittest.mock import AsyncMock, MagicMock

import pytest
from xrtm.forecast.kit.agents.prompting import CompiledAgent, PromptTemplate

from xrtm.train.kit.optimization.compiler import BrierOptimizer


@pytest.mark.asyncio
async def test_brier_optimizer_mse_logic():
    # Setup mocks
    optimizer_model = AsyncMock()
    optimizer_model.generate.return_value = "New instruction"

    agent = MagicMock(spec=CompiledAgent)
    agent.template = MagicMock(spec=PromptTemplate)
    agent.template.instruction = "Old instruction"
    agent.template.examples = []
    agent.template.version = 1

    optimizer = BrierOptimizer(optimizer_model)

    # Create a dataset
    # (Any, float (pred), int (ground truth))
    dataset = [
        ("input1", 0.8, 1),
        ("input2", 0.2, 0),
        ("input3", 0.6, 1),
    ]

    # Calculate expected MSE manually
    # (0.8 - 1)^2 = (-0.2)^2 = 0.04
    # (0.2 - 0)^2 = (0.2)^2 = 0.04
    # (0.6 - 1)^2 = (-0.4)^2 = 0.16
    # Sum = 0.24, Mean = 0.24 / 3 = 0.08
    expected_mse = 0.08

    # We can't easily check the internal variable `total_error` unless we inspect the call to `generate`.
    # The `meta_prompt` passed to `generate` contains the error.

    result = await optimizer.optimize(agent, dataset)

    # Verify the prompt contains the correct error
    call_args = optimizer_model.generate.call_args
    assert call_args is not None
    prompt = call_args[0][0]

    # Check if the MSE is formatted correctly in the prompt
    expected_str = f"Current Goal: Minimize Brier Score (Current: {expected_mse:.4f})."
    assert expected_str in prompt

    # Verify return value
    assert result.instruction == "New instruction"
    assert result.version == 2

@pytest.mark.asyncio
async def test_brier_optimizer_empty_dataset():
    # What happens if dataset is empty?
    # The original code would divide by zero.
    # total_error = sum((p - g) ** 2 for _, p, g in dataset) / len(dataset)

    optimizer_model = AsyncMock()
    optimizer = BrierOptimizer(optimizer_model)
    agent = MagicMock(spec=CompiledAgent)

    dataset = []

    with pytest.raises(ZeroDivisionError):
        await optimizer.optimize(agent, dataset)
