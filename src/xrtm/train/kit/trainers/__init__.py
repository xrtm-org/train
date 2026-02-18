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

r"""
Trainer implementations for xrtm-train.

This module provides the BaseTrainer protocol and concrete implementations
for training LLM-based forecasting models with prior injection.

Example:
    >>> from xrtm.train.kit.trainers import SingleNodeTrainer
    >>> trainer = SingleNodeTrainer(model=my_model, loss_fn=BetaNLLLoss())
    >>> result = trainer.train_step(sample)
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

from xrtm.train.kit.builders import TrainingSample


@dataclass
class TrainResult:
    r"""Result of a single training step."""

    loss: float
    prediction_alpha: float
    prediction_beta: float
    target_alpha: float
    target_beta: float
    metadata: dict = field(default_factory=dict)

    @property
    def prediction_mean(self) -> float:
        r"""Mean of the predicted Beta distribution."""
        return self.prediction_alpha / (self.prediction_alpha + self.prediction_beta)

    @property
    def target_mean(self) -> float:
        r"""Mean of the target Beta distribution."""
        return self.target_alpha / (self.target_alpha + self.target_beta)


class BaseTrainer(Protocol):
    r"""
    Protocol for all trainer implementations.

    This defines the interface that all trainers must implement,
    enabling consistent usage across single-node and multi-agent training.
    """

    @abstractmethod
    def train_step(self, sample: TrainingSample) -> TrainResult:
        r"""
        Execute a single training step.

        Args:
            sample: A training sample with prior state and target.

        Returns:
            TrainResult containing loss and predictions.
        """
        ...

    @abstractmethod
    def evaluate(self, samples: list[TrainingSample]) -> dict[str, float]:
        r"""
        Evaluate the model on a batch of samples.

        Args:
            samples: List of training samples.

        Returns:
            Dictionary of evaluation metrics.
        """
        ...


class ReasoningTrace(BaseModel):
    r"""
    Captured reasoning trace during training (governance compliance).

    Implements the reasoning_trace requirement from forecast_object_v1.
    """

    narrative: str = Field(..., description="Text explanation of the prediction")
    causal_factors: list[str] = Field(
        default_factory=list, description="Key factors influencing the prediction"
    )
    confidence_signals: dict[str, float] = Field(
        default_factory=dict, description="Confidence indicators from the model"
    )


class SingleNodeTrainer:
    r"""
    Trainer for single LLM node with prior injection.

    This is the simplest trainer implementation, treating a single LLM
    as a degenerate 1-node orchestration graph. It implements prior
    injection via embedding projection.

    Attributes:
        model: The LLM or prediction function.
        loss_fn: Loss function for Beta distribution predictions.
        projector: Converts prior state to model-compatible embeddings.
        capture_traces: Whether to capture reasoning traces.

    Example:
        >>> trainer = SingleNodeTrainer(
        ...     model=my_llm_model,
        ...     loss_fn=BetaNLLLoss(),
        ...     projector=BetaPriorProjector(prior_dim=4, hidden_dim=64),
        ... )
        >>> result = trainer.train_step(sample)
        >>> print(f"Loss: {result.loss:.4f}")
    """

    def __init__(
        self,
        model: Callable[[str, Any], tuple[float, float]],
        loss_fn: Any,
        projector: Optional[Any] = None,
        capture_traces: bool = True,
    ) -> None:
        r"""
        Initialize the SingleNodeTrainer.

        Args:
            model: Callable that takes (text, embedding) and returns (alpha, beta).
            loss_fn: Loss function with compute(alpha, beta, target) method.
            projector: Optional prior projector for embedding injection.
            capture_traces: Whether to capture reasoning traces.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.projector = projector
        self.capture_traces = capture_traces
        self._traces: list[ReasoningTrace] = []

    def train_step(self, sample: TrainingSample) -> TrainResult:
        r"""
        Execute a single training step with prior injection.

        The prior state is projected to an embedding (if projector provided)
        and passed to the model along with the news context.

        Args:
            sample: TrainingSample containing prior, news, and target.

        Returns:
            TrainResult with loss and prediction details.
        """
        # Project prior state to embedding
        prior_embedding = None
        if self.projector is not None:
            prior_embedding = self.projector(
                alpha=sample.prior_alpha,
                beta=sample.prior_beta,
                silence_delta=sample.silence_delta,
                deadline_delta=sample.deadline_delta,
            )

        # Build input text with context
        input_text = self._build_input(sample)

        # Get model prediction
        pred_alpha, pred_beta = self.model(input_text, prior_embedding)

        # Compute loss
        target_mean = sample.target_mean
        loss = self.loss_fn.compute(pred_alpha, pred_beta, target_mean)

        # Capture trace if enabled
        if self.capture_traces:
            trace = ReasoningTrace(
                narrative=f"Prediction for {sample.question_id} at step {sample.step_index}",
                causal_factors=[sample.current_news] if sample.current_news else [],
                confidence_signals={
                    "prior_mean": sample.prior_mean,
                    "pred_mean": pred_alpha / (pred_alpha + pred_beta),
                },
            )
            self._traces.append(trace)

        return TrainResult(
            loss=loss,
            prediction_alpha=pred_alpha,
            prediction_beta=pred_beta,
            target_alpha=sample.target_alpha,
            target_beta=sample.target_beta,
            metadata={
                "question_id": sample.question_id,
                "step_index": sample.step_index,
            },
        )

    def evaluate(self, samples: list[TrainingSample]) -> dict[str, float]:
        r"""
        Evaluate the model on a batch of samples.

        Args:
            samples: List of training samples.

        Returns:
            Dictionary with mean_loss, mean_calibration_error, etc.
        """
        if not samples:
            return {"mean_loss": 0.0, "count": 0}

        total_loss = 0.0
        calibration_errors = []

        for sample in samples:
            result = self.train_step(sample)
            total_loss += result.loss
            calibration_errors.append(abs(result.prediction_mean - result.target_mean))

        return {
            "mean_loss": total_loss / len(samples),
            "mean_calibration_error": sum(calibration_errors) / len(calibration_errors),
            "count": len(samples),
        }

    def get_traces(self) -> list[ReasoningTrace]:
        r"""Return captured reasoning traces."""
        return self._traces.copy()

    def clear_traces(self) -> None:
        r"""Clear captured reasoning traces."""
        self._traces.clear()

    def _build_input(self, sample: TrainingSample) -> str:
        r"""Build the input text for the model."""
        parts = []

        # Add context from previous news
        if sample.news_context:
            parts.append("Previous context:")
            for ctx in sample.news_context[-3:]:  # Last 3 items
                parts.append(f"  - {ctx}")

        # Add current news
        if sample.current_news:
            parts.append(f"Current: {sample.current_news}")

        return "\n".join(parts) if parts else "[No news context]"


__all__ = [
    "BaseTrainer",
    "SingleNodeTrainer",
    "TrainResult",
    "ReasoningTrace",
]
