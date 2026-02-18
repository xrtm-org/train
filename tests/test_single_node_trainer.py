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

r"""Unit tests for SingleNodeTrainer and related components."""

from datetime import datetime, timezone

import pytest

from xrtm.train.kit.builders import TrainingSample
from xrtm.train.kit.trainers import (
    ReasoningTrace,
    SingleNodeTrainer,
    TrainResult,
)


class TestTrainResult:
    r"""Tests for TrainResult dataclass."""

    def test_prediction_mean(self) -> None:
        r"""Verify prediction mean calculation."""
        result = TrainResult(
            loss=0.5,
            prediction_alpha=8.0,
            prediction_beta=2.0,
            target_alpha=7.0,
            target_beta=3.0,
        )
        assert result.prediction_mean == pytest.approx(0.8)
        assert result.target_mean == pytest.approx(0.7)


class TestReasoningTrace:
    r"""Tests for ReasoningTrace schema."""

    def test_creation(self) -> None:
        r"""Verify basic creation."""
        trace = ReasoningTrace(
            narrative="Test prediction",
            causal_factors=["factor1", "factor2"],
            confidence_signals={"high": 0.9},
        )
        assert trace.narrative == "Test prediction"
        assert len(trace.causal_factors) == 2


class TestSingleNodeTrainer:
    r"""Tests for SingleNodeTrainer."""

    @pytest.fixture
    def mock_model(self):
        r"""Create a mock model that returns Beta parameters."""
        def model(text: str, embedding) -> tuple[float, float]:
            # Simple mock: slight adjustment from prior
            return (8.0, 2.0)
        return model

    @pytest.fixture
    def mock_loss(self):
        r"""Create a mock loss function."""
        class MockLoss:
            def compute(self, alpha: float, beta: float, target: float) -> float:
                pred_mean = alpha / (alpha + beta)
                return abs(pred_mean - target)
        return MockLoss()

    @pytest.fixture
    def sample(self) -> TrainingSample:
        r"""Create a test training sample."""
        return TrainingSample(
            question_id="q123",
            step_index=1,
            snapshot_time=datetime.now(timezone.utc),
            prior_alpha=7.0,
            prior_beta=3.0,
            current_news="Test news headline",
            target_alpha=8.0,
            target_beta=2.0,
            news_context=["Previous headline 1", "Previous headline 2"],
        )

    def test_train_step_returns_result(self, mock_model, mock_loss, sample) -> None:
        r"""Verify train_step returns TrainResult."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        result = trainer.train_step(sample)

        assert isinstance(result, TrainResult)
        assert result.prediction_alpha == 8.0
        assert result.prediction_beta == 2.0
        assert result.metadata["question_id"] == "q123"

    def test_train_step_computes_loss(self, mock_model, mock_loss, sample) -> None:
        r"""Verify loss is computed correctly."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        result = trainer.train_step(sample)

        # Mock model returns (8, 2) -> mean = 0.8
        # Target is (8, 2) -> mean = 0.8
        # Loss = |0.8 - 0.8| = 0.0
        assert result.loss == pytest.approx(0.0)

    def test_captures_reasoning_traces(self, mock_model, mock_loss, sample) -> None:
        r"""Verify reasoning traces are captured."""
        trainer = SingleNodeTrainer(
            model=mock_model, loss_fn=mock_loss, capture_traces=True
        )
        trainer.train_step(sample)

        traces = trainer.get_traces()
        assert len(traces) == 1
        assert "q123" in traces[0].narrative

    def test_clear_traces(self, mock_model, mock_loss, sample) -> None:
        r"""Verify traces can be cleared."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        trainer.train_step(sample)
        assert len(trainer.get_traces()) == 1

        trainer.clear_traces()
        assert len(trainer.get_traces()) == 0

    def test_evaluate_batch(self, mock_model, mock_loss, sample) -> None:
        r"""Verify batch evaluation."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        samples = [sample, sample, sample]

        metrics = trainer.evaluate(samples)
        assert metrics["count"] == 3
        assert "mean_loss" in metrics
        assert "mean_calibration_error" in metrics

    def test_evaluate_empty_batch(self, mock_model, mock_loss) -> None:
        r"""Verify empty batch returns zeros."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        metrics = trainer.evaluate([])
        assert metrics["count"] == 0
        assert metrics["mean_loss"] == 0.0

    def test_build_input_with_context(self, mock_model, mock_loss, sample) -> None:
        r"""Verify input building includes context."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        input_text = trainer._build_input(sample)

        assert "Previous context" in input_text
        assert "Previous headline 1" in input_text
        assert "Current:" in input_text
        assert "Test news headline" in input_text

    def test_build_input_no_context(self, mock_model, mock_loss) -> None:
        r"""Verify input building handles empty context."""
        trainer = SingleNodeTrainer(model=mock_model, loss_fn=mock_loss)
        sample = TrainingSample(
            question_id="q123",
            step_index=0,
            snapshot_time=datetime.now(timezone.utc),
            prior_alpha=1.0,
            prior_beta=1.0,
            current_news="",
            target_alpha=2.0,
            target_beta=1.0,
        )
        input_text = trainer._build_input(sample)
        assert "[No news context]" in input_text
