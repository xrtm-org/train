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

import os
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
from pydantic import SecretStr
from xrtm.data import ForecastOutput, ForecastQuestion, MetadataBase
from xrtm.eval.core.schemas import ForecastResolution
from xrtm.eval.kit.eval.metrics import BrierScoreEvaluator
from xrtm.forecast.core.config.inference import OpenAIConfig
from xrtm.forecast.core.orchestrator import Orchestrator
from xrtm.forecast.core.schemas.graph import BaseGraphState
from xrtm.forecast.providers.inference.factory import ModelFactory

from xrtm.train.kit.builders import BetaPriorSnapshot, NewsEvent, TrainingSampleBuilder
from xrtm.train.simulation.runner import BacktestRunner


def _local_base_url() -> str:
    return os.getenv("XRTM_LOCAL_LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")


def _health_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[:-3] + "/health"
    return base_url.rstrip("/") + "/health"


def _require_local_llm(base_url: str) -> None:
    request = Request(_health_url(base_url), method="GET")
    try:
        with urlopen(request, timeout=2) as response:
            if response.status != 200:
                pytest.fail(f"Local LLM endpoint returned HTTP {response.status}")
    except URLError as exc:
        pytest.fail(f"Local LLM endpoint is not reachable at {base_url}: {exc}")


@pytest.mark.local_llm
def test_offline_stack_e2e_with_local_llamacpp() -> None:
    r"""Exercise local inference, schemas, eval scoring, and train utilities together."""
    base_url = _local_base_url()
    _require_local_llm(base_url)

    config = OpenAIConfig(
        model_id=os.getenv("XRTM_LOCAL_LLM_MODEL", "Qwen3.5-27B-Q4_K_M.gguf"),
        api_key=SecretStr(os.getenv("XRTM_LOCAL_LLM_API_KEY", "test")),
        base_url=base_url,
    )
    provider = ModelFactory.get_provider(config)
    response = provider.generate_content(
        "Reply with exactly OFFLINE_STACK_OK and no other text.",
        max_tokens=int(os.getenv("XRTM_LOCAL_LLM_MAX_TOKENS", "512")),
        temperature=0,
    )
    assert "OFFLINE_STACK_OK" in response.text

    snapshot_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    question = ForecastQuestion(
        id="offline-q1",
        title="Will the local offline XRTM stack complete its smoke test?",
        metadata=MetadataBase(snapshot_time=snapshot_time),
    )
    forecast = ForecastOutput(
        question_id=question.id,
        probability=0.7,
        reasoning=response.text,
        structural_trace=["local_llm", "schema_validation"],
        metadata=MetadataBase(snapshot_time=snapshot_time),
    )
    resolution = ForecastResolution(question_id=question.id, outcome="yes", resolved_at=snapshot_time + timedelta(days=1))

    evaluator = BrierScoreEvaluator()
    score = evaluator.evaluate(forecast.probability, resolution.outcome, question.id)
    assert score.score == pytest.approx(0.09)

    builder = TrainingSampleBuilder(context_window_size=2)
    samples = builder.build_sequence(
        question_id=question.id,
        news_events=[
            NewsEvent(content="Initial local smoke signal", timestamp=snapshot_time),
            NewsEvent(content="Local provider returned sentinel", timestamp=snapshot_time + timedelta(hours=1)),
        ],
        prior_snapshots=[
            BetaPriorSnapshot(alpha=3.0, beta=2.0, timestamp=snapshot_time),
            BetaPriorSnapshot(alpha=7.0, beta=3.0, timestamp=snapshot_time + timedelta(hours=1)),
        ],
        deadline=snapshot_time + timedelta(days=1),
    )
    assert len(samples) == 1
    assert samples[0].target_mean == pytest.approx(0.7)

    state = BaseGraphState(subject_id=question.id, node_reports={"final_forecast": forecast})
    runner = BacktestRunner(orchestrator=Orchestrator(), evaluator=evaluator)
    backtest_result = runner.evaluate_state(state, resolution, question.id, snapshot_time, tags=["offline", "local_llm"])
    assert backtest_result.score == pytest.approx(0.09)
    assert backtest_result.metadata["tags"] == ["offline", "local_llm"]
