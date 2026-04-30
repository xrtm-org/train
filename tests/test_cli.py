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

import json
from pathlib import Path

from click.testing import CliRunner

from xrtm.train.cli import main


def test_data_prepare_builds_jsonl_samples() -> None:
    r"""CLI data preparation should use the current TrainingSampleBuilder API."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        trades_path = Path("trades.json")
        news_path = Path("news.json")
        output_path = Path("samples")

        trades_path.write_text(json.dumps([
            {"question_id": "q-cli", "timestamp": "2026-01-01T00:00:00Z", "alpha": 1.0, "beta": 1.0},
            {"question_id": "q-cli", "timestamp": "2026-01-02T00:00:00Z", "alpha": 3.0, "beta": 2.0},
            {"question_id": "q-cli", "timestamp": "2026-01-03T00:00:00Z", "alpha": 4.0, "beta": 1.0},
        ]))
        news_path.write_text(json.dumps([
            {"timestamp": "2026-01-01T00:00:00Z", "content": "Initial signal"},
            {"timestamp": "2026-01-02T00:00:00Z", "content": "Evidence improves"},
            {"timestamp": "2026-01-03T00:00:00Z", "content": "Final update"},
        ]))

        result = runner.invoke(
            main,
            [
                "data",
                "prepare",
                "--trades",
                str(trades_path),
                "--news",
                str(news_path),
                "--output",
                str(output_path),
                "--split",
                "0.5",
            ],
        )

        assert result.exit_code == 0, result.output
        train_rows = [json.loads(line) for line in (output_path / "train.jsonl").read_text().splitlines()]
        val_rows = [json.loads(line) for line in (output_path / "val.jsonl").read_text().splitlines()]
        assert len(train_rows) == 1
        assert len(val_rows) == 1
        assert train_rows[0]["question_id"] == "q-cli"
        assert train_rows[0]["current_news"] == "Initial signal"
        assert train_rows[0]["target_alpha"] == 3.0
        assert isinstance(train_rows[0]["snapshot_time"], str)


def test_data_prepare_can_derive_news_and_priors_from_trade_records() -> None:
    r"""Trade records with probabilities are enough for a small local smoke dataset."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        trades_path = Path("trades.json")
        output_path = Path("samples")

        trades_path.write_text(json.dumps([
            {"market_id": "m-cli", "timestamp": "2026-01-01T00:00:00Z", "price": 0.4, "amount": 10},
            {"market_id": "m-cli", "timestamp": "2026-01-02T00:00:00Z", "price": 0.6, "amount": 10},
        ]))

        result = runner.invoke(
            main,
            [
                "data",
                "prepare",
                "--trades",
                str(trades_path),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0, result.output
        train_rows = [json.loads(line) for line in (output_path / "train.jsonl").read_text().splitlines()]
        val_rows = [json.loads(line) for line in (output_path / "val.jsonl").read_text().splitlines()]
        rows = train_rows + val_rows
        assert len(rows) == 1
        assert rows[0]["question_id"] == "m-cli"
        assert rows[0]["current_news"] == "Market probability update: 0.400"
        assert rows[0]["prior_alpha"] == 4.0
        assert rows[0]["target_alpha"] == 6.0
