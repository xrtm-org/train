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

r"""Unit tests for TrainingSampleBuilder."""

from datetime import datetime, timedelta, timezone

import pytest

from xrtm.train.kit.builders import (
    BetaPriorSnapshot,
    NewsEvent,
    TrainingSample,
    TrainingSampleBuilder,
)


class TestNewsEvent:
    r"""Tests for NewsEvent schema."""

    def test_creation(self) -> None:
        r"""Verify basic creation."""
        event = NewsEvent(
            content="Breaking news",
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        assert event.content == "Breaking news"


class TestBetaPriorSnapshot:
    r"""Tests for BetaPriorSnapshot schema."""

    def test_mean_property(self) -> None:
        r"""Verify mean calculation."""
        snapshot = BetaPriorSnapshot(
            alpha=7.0,
            beta=3.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert snapshot.mean == pytest.approx(0.7)


class TestTrainingSample:
    r"""Tests for TrainingSample schema."""

    def test_prior_mean(self) -> None:
        r"""Verify prior mean calculation."""
        sample = TrainingSample(
            question_id="q123",
            step_index=0,
            snapshot_time=datetime.now(timezone.utc),
            prior_alpha=7.0,
            prior_beta=3.0,
            current_news="Test news",
            target_alpha=8.0,
            target_beta=2.0,
        )
        assert sample.prior_mean == pytest.approx(0.7)
        assert sample.target_mean == pytest.approx(0.8)


class TestTrainingSampleBuilder:
    r"""Tests for TrainingSampleBuilder."""

    def _make_news_events(self, count: int = 5) -> list[NewsEvent]:
        r"""Helper to create news events."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [
            NewsEvent(
                content=f"News {i}",
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(count)
        ]

    def _make_prior_snapshots(self, count: int = 5) -> list[BetaPriorSnapshot]:
        r"""Helper to create prior snapshots with increasing conviction."""
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [
            BetaPriorSnapshot(
                alpha=1.0 + i * 2,
                beta=1.0,
                timestamp=base_time + timedelta(hours=i),
            )
            for i in range(count)
        ]

    def test_empty_events_returns_empty(self) -> None:
        r"""Empty or single event should return empty list."""
        builder = TrainingSampleBuilder()
        assert builder.build_sequence("q1", [], [], datetime.now(timezone.utc)) == []

    def test_single_event_returns_empty(self) -> None:
        r"""Single event can't form a training pair."""
        builder = TrainingSampleBuilder()
        news = self._make_news_events(1)
        priors = self._make_prior_snapshots(1)
        deadline = datetime(2026, 1, 10, tzinfo=timezone.utc)
        assert builder.build_sequence("q1", news, priors, deadline) == []

    def test_sequence_length(self) -> None:
        r"""N events should produce N-1 samples (Teacher Forcing pairs)."""
        builder = TrainingSampleBuilder()
        news = self._make_news_events(5)
        priors = self._make_prior_snapshots(5)
        deadline = datetime(2026, 1, 10, tzinfo=timezone.utc)
        samples = builder.build_sequence("q1", news, priors, deadline)
        # 5 events → 4 training pairs
        assert len(samples) == 4

    def test_teacher_forcing_propagation(self) -> None:
        r"""Each sample's target should become next sample's prior."""
        builder = TrainingSampleBuilder()
        news = self._make_news_events(4)
        priors = self._make_prior_snapshots(4)
        deadline = datetime(2026, 1, 10, tzinfo=timezone.utc)
        samples = builder.build_sequence("q1", news, priors, deadline)

        # Verify Teacher Forcing: sample[i].target == sample[i+1].prior
        for i in range(len(samples) - 1):
            assert samples[i].target_alpha == samples[i + 1].prior_alpha
            assert samples[i].target_beta == samples[i + 1].prior_beta

    def test_rolling_context_grows(self) -> None:
        r"""Context should accumulate up to window size."""
        builder = TrainingSampleBuilder(context_window_size=3)
        news = self._make_news_events(5)
        priors = self._make_prior_snapshots(5)
        deadline = datetime(2026, 1, 10, tzinfo=timezone.utc)
        samples = builder.build_sequence("q1", news, priors, deadline)

        # First sample has no context (started empty)
        assert len(samples[0].news_context) == 0
        # Context grows with each step
        assert len(samples[1].news_context) == 1
        assert len(samples[2].news_context) == 2
        # Capped at window size
        assert len(samples[3].news_context) == 3

    def test_mismatched_lengths_raises(self) -> None:
        r"""Different length lists should raise."""
        builder = TrainingSampleBuilder()
        news = self._make_news_events(5)
        priors = self._make_prior_snapshots(3)
        deadline = datetime(2026, 1, 10, tzinfo=timezone.utc)
        with pytest.raises(ValueError):
            builder.build_sequence("q1", news, priors, deadline)
