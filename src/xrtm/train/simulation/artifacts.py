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

r"""Shared schema and artifact helpers for simulation backtests."""

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel
from xrtm.data.core.schemas.forecast import ForecastOutput
from xrtm.eval.core.schemas import ForecastResolution

TRUE_VALUES = {"true", "yes", "1", "pass"}
FALSE_VALUES = {"false", "no", "0", "fail"}


def validate_resolution_for_question(resolution: Any, question_id: str) -> ForecastResolution:
    r"""Validate a resolution against the canonical schema and question id."""
    if isinstance(resolution, ForecastResolution):
        validated = resolution
    else:
        validated = ForecastResolution.model_validate(resolution)

    if validated.question_id != question_id:
        raise ValueError(
            f"Resolution question_id {validated.question_id!r} does not match forecast question {question_id!r}"
        )
    return validated


def resolution_payload(resolution: ForecastResolution) -> dict[str, Any]:
    r"""Return a JSON-safe payload for a validated resolution."""
    return resolution.model_dump(mode="json")


def normalize_binary_outcome(outcome_raw: Any) -> float:
    r"""Normalize common binary resolution values for evaluators."""
    if isinstance(outcome_raw, bool):
        return 1.0 if outcome_raw else 0.0
    if isinstance(outcome_raw, str):
        lowered = outcome_raw.strip().lower()
        if lowered in TRUE_VALUES:
            return 1.0
        if lowered in FALSE_VALUES:
            return 0.0
        try:
            return float(outcome_raw)
        except ValueError:
            raise ValueError(f"Unsupported binary outcome string: {outcome_raw!r}") from None
    return float(outcome_raw)


def prediction_value_and_payload(prediction: Any) -> tuple[float, dict[str, Any] | None]:
    r"""Extract a scoring probability and full payload from a prediction object."""
    payload = serialize_payload(prediction)
    if isinstance(prediction, ForecastOutput):
        return float(prediction.confidence), payload
    if isinstance(prediction, Mapping):
        if "confidence" in prediction:
            return float(prediction["confidence"]), payload
        if "probability" in prediction:
            return float(prediction["probability"]), payload
    if isinstance(prediction, (int, float)):
        return float(prediction), payload
    return float(getattr(prediction, "confidence", prediction)), payload


def serialize_payload(value: Any) -> dict[str, Any] | None:
    r"""Serialize rich prediction artifacts without discarding useful fields."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value
