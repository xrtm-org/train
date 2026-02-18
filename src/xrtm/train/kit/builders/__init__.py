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
Training sample builder for LLM fine-tuning.

This module constructs training samples using Teacher Forcing (Decision 2),
building sequences where each step receives the ground truth from the
previous step rather than model predictions, enabling stable gradient flow.

Example:
    >>> from xrtm.train.kit.builders import TrainingSampleBuilder
    >>> builder = TrainingSampleBuilder(context_window_size=5)
    >>> samples = builder.build_sequence(
    ...     question_id="q123",
    ...     news_events=news_list,
    ...     prior_snapshots=prior_list,
    ...     deadline=deadline_dt,
    ... )
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NewsEvent(BaseModel):
    r"""
    A timestamped news item for context building.

    Attributes:
        content: The news headline or content text.
        timestamp: When this news was published (UTC).
        source: Optional source identifier.
    """

    content: str = Field(..., description="The news headline or content text")
    timestamp: datetime = Field(..., description="When this news was published (UTC)")
    source: Optional[str] = Field(default=None, description="Source identifier")


class BetaPriorSnapshot(BaseModel):
    r"""
    A snapshot of Beta distribution parameters at a point in time.

    Attributes:
        alpha: Shape parameter α.
        beta: Shape parameter β.
        timestamp: When this snapshot was captured (UTC).
    """

    alpha: float = Field(..., gt=0, description="Shape parameter α")
    beta: float = Field(..., gt=0, description="Shape parameter β")
    timestamp: datetime = Field(..., description="When this snapshot was captured (UTC)")

    @property
    def mean(self) -> float:
        r"""Expected value: α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)


class TrainingSample(BaseModel):
    r"""
    A complete training sample for the forecasting LLM.

    Implements Decisions 1-5 from the architecture document:
    - Decision 1: Beta injection via prior_alpha, prior_beta
    - Decision 2: Teacher Forcing (target from ground truth)
    - Decision 5: Rolling context via news_context
    - Decision 6: Target relaxation applied during loss computation

    Attributes:
        question_id: Identifier for the question being forecast.
        step_index: Position in the update sequence (0-indexed).
        snapshot_time: When this sample was created (UTC).
        prior_alpha: Current belief α parameter (input).
        prior_beta: Current belief β parameter (input).
        silence_delta: Normalized time since last information update.
        deadline_delta: Normalized time remaining until resolution.
        news_context: Rolling history of recent headlines.
        current_news: Primary news input for this step.
        target_alpha: Next prior α (teacher forcing target).
        target_beta: Next prior β (teacher forcing target).
        target_reasoning: Optional text explanation target.
    """

    question_id: str = Field(..., description="Identifier for the question")
    step_index: int = Field(..., ge=0, description="Position in update sequence")
    snapshot_time: datetime = Field(..., description="When this sample was created")

    # Input side (Decision 1)
    prior_alpha: float = Field(..., gt=0, description="Current belief α")
    prior_beta: float = Field(..., gt=0, description="Current belief β")
    silence_delta: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Normalized time since last update",
    )
    deadline_delta: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Normalized time remaining",
    )

    # Context (Decision 5)
    news_context: list[str] = Field(
        default_factory=list,
        description="Rolling history of recent headlines",
    )
    current_news: str = Field(..., description="Primary news input")

    # Target side (Decision 2: Teacher Forcing)
    target_alpha: float = Field(..., gt=0, description="Next prior α")
    target_beta: float = Field(..., gt=0, description="Next prior β")
    target_reasoning: Optional[str] = Field(
        default=None,
        description="Optional text explanation target",
    )

    @property
    def prior_mean(self) -> float:
        r"""Input prior mean: α / (α + β)."""
        return self.prior_alpha / (self.prior_alpha + self.prior_beta)

    @property
    def target_mean(self) -> float:
        r"""Target prior mean: α / (α + β)."""
        return self.target_alpha / (self.target_alpha + self.target_beta)


class TrainingSampleBuilder:
    r"""
    Constructs training samples using Teacher Forcing (Decision 2).

    At each step t, we feed the GROUND TRUTH from t-1 (not model prediction).
    This trains the model to learn the transition function δ, not accumulate errors.

    Attributes:
        context_window_size: Maximum number of headlines in rolling context.
        target_clamp: Tuple of (min, max) for target mean clamping.

    Example:
        >>> builder = TrainingSampleBuilder(context_window_size=5)
        >>> samples = builder.build_sequence(
        ...     question_id="q123",
        ...     news_events=news_list,
        ...     prior_snapshots=prior_list,
        ...     deadline=deadline_dt,
        ... )
        >>> # Sample[t].prior == GroundTruth[t-1] (Teacher Forcing)
    """

    def __init__(
        self,
        context_window_size: int = 5,
        target_clamp: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        r"""
        Initialize the sample builder.

        Args:
            context_window_size: Maximum headlines in rolling context.
            target_clamp: (min, max) for target mean clamping.
        """
        self.context_window_size = context_window_size
        self.target_clamp = target_clamp

    def _compute_time_delta(
        self,
        current: datetime,
        reference: datetime,
        total_duration: float,
    ) -> float:
        r"""Compute normalized time delta in [0, 1]."""
        if total_duration <= 0:
            return 0.0
        delta_seconds = (reference - current).total_seconds()
        normalized = delta_seconds / total_duration
        return max(0.0, min(1.0, normalized))

    def _compute_silence_delta(
        self,
        current_time: datetime,
        last_update_time: datetime,
        max_silence_hours: float = 72.0,
    ) -> float:
        r"""Compute normalized silence period in [0, 1]."""
        delta_seconds = (current_time - last_update_time).total_seconds()
        max_silence_seconds = max_silence_hours * 3600
        normalized = delta_seconds / max_silence_seconds
        return max(0.0, min(1.0, normalized))

    def build_sequence(
        self,
        question_id: str,
        news_events: list[NewsEvent],
        prior_snapshots: list[BetaPriorSnapshot],
        deadline: datetime,
        start_time: Optional[datetime] = None,
    ) -> list[TrainingSample]:
        r"""
        Build a full training sequence from resolved history.

        Teacher Forcing: Each sample's prior is the GROUND TRUTH from the
        previous step, not a model prediction.

        Args:
            question_id: Identifier for the question.
            news_events: Timestamped news items in chronological order.
            prior_snapshots: Beta snapshots corresponding to news events.
            deadline: When the question resolves.
            start_time: Optional start time (defaults to first news timestamp).

        Returns:
            List of TrainingSample objects for training.

        Example:
            >>> # Sample 1: Prior(1,1) + "Rumor..." → Target(7,3)
            >>> # Sample 2: Prior(7,3) + "Confirmed..." → Target(95,5)
            >>> # Note: Sample 2.prior == Sample 1.target (Teacher Forcing)
        """
        if len(news_events) < 2 or len(prior_snapshots) < 2:
            return []

        if len(news_events) != len(prior_snapshots):
            msg = "news_events and prior_snapshots must have same length"
            raise ValueError(msg)

        # Sort by timestamp
        paired = sorted(
            zip(news_events, prior_snapshots, strict=True),
            key=lambda x: x[0].timestamp,
        )
        news_events = [p[0] for p in paired]
        prior_snapshots = [p[1] for p in paired]

        # Compute total duration
        if start_time is None:
            start_time = news_events[0].timestamp
        total_duration = (deadline - start_time).total_seconds()

        samples: list[TrainingSample] = []
        rolling_context: list[str] = []

        for i in range(len(news_events) - 1):
            current_news = news_events[i]
            current_prior = prior_snapshots[i]
            target_prior = prior_snapshots[i + 1]

            # Compute time deltas
            deadline_delta = self._compute_time_delta(
                current_news.timestamp, deadline, total_duration
            )

            if i > 0:
                silence_delta = self._compute_silence_delta(
                    current_news.timestamp,
                    news_events[i - 1].timestamp,
                )
            else:
                silence_delta = 0.0

            # Build sample with Teacher Forcing
            sample = TrainingSample(
                question_id=question_id,
                step_index=i,
                snapshot_time=current_news.timestamp,
                prior_alpha=current_prior.alpha,
                prior_beta=current_prior.beta,
                silence_delta=silence_delta,
                deadline_delta=deadline_delta,
                news_context=list(rolling_context),
                current_news=current_news.content,
                target_alpha=target_prior.alpha,
                target_beta=target_prior.beta,
            )
            samples.append(sample)

            # Update rolling context (FIFO)
            rolling_context.append(current_news.content)
            if len(rolling_context) > self.context_window_size:
                rolling_context.pop(0)

        return samples


__all__ = [
    "NewsEvent",
    "BetaPriorSnapshot",
    "TrainingSample",
    "TrainingSampleBuilder",
]
