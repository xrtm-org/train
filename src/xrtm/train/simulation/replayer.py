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

import logging
from typing import Optional

# From xrtm-eval
from xrtm.eval.core.eval.definitions import EvaluationResult, Evaluator
from xrtm.eval.schemas.forecast import ForecastResolution

# From xrtm-data
# From xrtm-forecast (Internal)
from xrtm.forecast.core.orchestrator import Orchestrator
from xrtm.forecast.core.schemas.graph import BaseGraphState

from xrtm.train.simulation.runner import BacktestRunner

__all__ = ["TraceReplayer"]
logger = logging.getLogger(__name__)


class TraceReplayer:
    """Utility class for saving, loading, and replaying evaluation traces."""

    @staticmethod
    def save_trace(state: BaseGraphState, path: str) -> None:
        try:
            json_str = state.model_dump_json(indent=2)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Trace saved to %s", path)
        except Exception as e:
            logger.error("Failed to save trace to %s: %s", path, e)
            raise

    @staticmethod
    def load_trace(path: str) -> BaseGraphState:
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_str = f.read()
            state = BaseGraphState.model_validate_json(json_str)
            return state
        except Exception as e:
            logger.error("Failed to load trace from %s: %s", path, e)
            raise

    def replay_evaluation(
        self,
        trace_path: str,
        resolution: ForecastResolution | float | str,
        evaluator: Optional[Evaluator] = None,
        subject_id_override: Optional[str] = None,
    ) -> EvaluationResult:
        state = self.load_trace(trace_path)
        subject_id = subject_id_override or state.subject_id
        if not isinstance(resolution, ForecastResolution):
            resolution = ForecastResolution(
                question_id=subject_id, outcome=str(resolution), metadata={"source": "replay_override"}
            )
        dummy_orch: Orchestrator[BaseGraphState] = Orchestrator()
        runner = BacktestRunner(orchestrator=dummy_orch, evaluator=evaluator)
        result = runner.evaluate_state(
            state=state,
            resolution=resolution,
            subject_id=subject_id,
            reference_time=state.temporal_context.reference_time if state.temporal_context else None,
        )
        result.metadata["is_replay"] = True
        return result
