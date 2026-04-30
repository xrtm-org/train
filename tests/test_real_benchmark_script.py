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

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_real_benchmark_provider_free_output_shape() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    script_path = workspace_root / "scripts" / "bench_real.py"
    spec = importlib.util.spec_from_file_location("bench_real", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_benchmark(
        provider_name="mock",
        limit=2,
        iterations=2,
        artifact_dir=Path(".cache/real-benchmark-tests"),
        write_artifact=False,
    )

    assert result["benchmark"] == "real-benchmark-tier"
    assert result["provider"] == "mock"
    assert result["corpus"]["attempted_records"] == 4
    assert result["corpus"]["successful_records"] == 4
    assert result["latency_seconds"]["total"] >= 0
    assert result["throughput"]["end_to_end_records_per_second"] > 0
    assert result["token_usage"]["available"] is True
    assert result["token_usage"]["total_tokens"] > 0
    assert result["cache"]["enabled"] is True
    assert result["cache"]["hits"] == 2
    assert result["cache"]["misses"] == 2
    assert result["failure"]["rate"] == 0
    assert "rss_mb" in result["memory"]["start"]
    assert result["eval"]["total_evaluations"] == 4
    assert result["train"]["backtest_evaluations"] == 4
    assert result["train"]["training_samples"] == 4
