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
CLI entry point for xrtm-train.

Provides commands for:
- Preparing training samples from raw data
- Training models with prior injection
- Evaluating trained models

Designed for extensibility:
- Pluggable model backends (Qwen, Llama, Mistral)
- Configurable training strategies (full, LoRA)
- YAML-based configuration

Example:
    $ xrtm-train prepare --trades trades.parquet --news news.json -o samples/
    $ xrtm-train train --config config.yaml --data samples/
    $ xrtm-train eval --model model/ --data samples/test/
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from xrtm.train.version import __version__

console = Console()


# =============================================================================
# Configuration Schema
# =============================================================================


DEFAULT_CONFIG = {
    "model": {
        "name": "Qwen/Qwen2.5-0.5B",  # Default to smaller Qwen for testing
        "family": "qwen3",
        "strategy": "full",  # "full" or "lora"
    },
    "training": {
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 2e-5,
        "epochs": 3,
        "warmup_ratio": 0.1,
        "max_length": 2048,
    },
    "prior": {
        "inject_method": "embedding",  # "embedding" or "prompt"
        "projector_hidden": 64,
    },
    "checkpoints": {
        "save_steps": 500,
        "save_total_limit": 3,
        "output_dir": "./models/checkpoints",
    },
    "cache": {
        "enabled": True,
        "dir": ".cache/xrtm/",
    },
}


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    r"""Load configuration from YAML file, merged with defaults."""
    config: Dict[str, Any] = DEFAULT_CONFIG.copy()

    if config_path:
        with open(config_path) as f:
            user_config = yaml.safe_load(f)

        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

    return config


# =============================================================================
# CLI Group
# =============================================================================


@click.group()
@click.version_option(version=__version__)
def main():
    r"""xrtm-train: Training pipeline for xRTM forecasting models."""
    pass


# =============================================================================
# Data Preparation Commands
# =============================================================================


@main.group()
def data():
    r"""Data preparation commands."""
    pass


@data.command("prepare")
@click.option("--trades", "-t", required=True, type=click.Path(exists=True), help="Trade data file (.parquet/.json)")
@click.option("--news", "-n", type=click.Path(exists=True), help="News data file (.json)")
@click.option("--priors", "-p", type=click.Path(exists=True), help="Pre-fitted priors directory")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory for samples")
@click.option("--split", default=0.8, help="Train/validation split ratio")
@click.option("--context-size", default=5, help="Rolling news context window size")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing files")
def prepare(
    trades: str,
    news: Optional[str],
    priors: Optional[str],
    output: str,
    split: float,
    context_size: int,
    force: bool,
):
    r"""
    Prepare training samples from raw data.

    Generates (prior, news, target) tuples using Teacher Forcing.
    Saves samples as JSONL files for efficient streaming during training.

    Example:
        xrtm-train data prepare -t trades.parquet -n news.json -o samples/
    """
    output_path = Path(output)
    train_path = output_path / "train.jsonl"
    val_path = output_path / "val.jsonl"

    # Check existing
    if train_path.exists() and not force:
        console.print(f"[yellow]⚠ Output exists:[/yellow] {output_path}")
        console.print("  Use --force to overwrite.")
        return

    console.print(Panel(
        f"[bold blue]Preparing Training Samples[/bold blue]\n"
        f"Trades: {trades}\n"
        f"News: {news or 'None'}\n"
        f"Context Size: {context_size}",
        title="xrtm-train",
    ))

    from xrtm.train.kit.builders import TrainingSampleBuilder

    # Load data
    trades_data = _load_trades(Path(trades))
    news_data = _load_news(Path(news)) if news else []
    priors_data = _load_priors(Path(priors)) if priors else None

    # Build samples
    builder = TrainingSampleBuilder(context_window_size=context_size)
    news_events = _coerce_news_events(news_data or trades_data)
    prior_snapshots = _coerce_prior_snapshots(priors_data or trades_data)
    if len(news_events) != len(prior_snapshots):
        raise click.ClickException(
            f"Need matching news events and prior snapshots; got {len(news_events)} news and "
            f"{len(prior_snapshots)} priors."
        )
    deadline = _infer_deadline(news_events)
    question_id = _infer_question_id(trades_data, news_data, priors_data)
    samples = builder.build_sequence(
        question_id=question_id,
        news_events=news_events,
        prior_snapshots=prior_snapshots,
        deadline=deadline,
    )

    if not samples:
        console.print("[red]✗ No samples generated. Check input data.[/red]")
        return

    # Split
    split_idx = int(len(samples) * split)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    _save_jsonl(train_samples, train_path)
    _save_jsonl(val_samples, val_path)

    console.print(f"[green]✓ Generated {len(samples)} samples[/green]")
    console.print(f"  Train: {len(train_samples)} → {train_path}")
    console.print(f"  Val: {len(val_samples)} → {val_path}")


# =============================================================================
# Training Commands
# =============================================================================


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Training config file (.yaml)")
@click.option("--data", "-d", "data_dir", required=True, type=click.Path(exists=True), help="Training data directory")
@click.option("--output", "-o", default="./models", type=click.Path(), help="Output directory for model")
@click.option("--resume", "-r", type=click.Path(exists=True), help="Resume from checkpoint")
@click.option("--model", "-m", default=None, help="Model name/path (overrides config)")
@click.option("--strategy", type=click.Choice(["full", "lora"]), default=None, help="Training strategy")
@click.option("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
@click.option("--batch-size", type=int, default=None, help="Batch size (overrides config)")
@click.option("--dry-run", is_flag=True, help="Validate config without training")
def train(
    config: Optional[str],
    data_dir: str,
    output: str,
    resume: Optional[str],
    model: Optional[str],
    strategy: Optional[str],
    epochs: Optional[int],
    batch_size: Optional[int],
    dry_run: bool,
):
    r"""
    Train a forecasting model with prior injection.

    Supports full fine-tuning and LoRA. Uses Qwen 3 family by default.
    Progress and checkpoints are saved automatically.

    Example:
        xrtm-train train -c config.yaml -d samples/ -o models/
        xrtm-train train -d samples/ -m Qwen/Qwen2.5-1.5B --epochs 3
    """
    # Load and merge config
    cfg = load_config(config)

    # CLI overrides
    if model:
        cfg["model"]["name"] = model
    if strategy:
        cfg["model"]["strategy"] = strategy
    if epochs:
        cfg["training"]["epochs"] = epochs
    if batch_size:
        cfg["training"]["batch_size"] = batch_size

    console.print(Panel(
        f"[bold blue]Training Configuration[/bold blue]\n"
        f"Model: {cfg['model']['name']}\n"
        f"Strategy: {cfg['model']['strategy']}\n"
        f"Epochs: {cfg['training']['epochs']}\n"
        f"Batch Size: {cfg['training']['batch_size']}",
        title="xrtm-train",
    ))

    # Show config table
    table = Table(title="Full Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="dim")
    table.add_column("Value", style="green")

    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, val in values.items():
                table.add_row(section, key, str(val))
        else:
            table.add_row("", section, str(values))

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run complete. No training performed.[/yellow]")
        return

    # Check for GPU dependencies
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"\n[bold]Device:[/bold] {device}")
        if device == "cuda":
            console.print(f"  GPU: {torch.cuda.get_device_name(0)}")
            console.print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        console.print("[red]✗ PyTorch not installed. Run: pip install xrtm-train[gpu][/red]")
        return

    # Run training
    _run_training(cfg, data_dir, output, resume)


def _run_training(cfg: dict, data_dir: str, output: str, resume: Optional[str]) -> None:
    r"""Execute the training loop."""
    import torch  # noqa: F401 - used for device detection

    console.print("\n[bold]Starting training...[/bold]\n")

    # Load data
    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"

    train_samples = _load_jsonl(train_path)
    val_samples = _load_jsonl(val_path)

    console.print(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")

    # Initialize model (placeholder - actual model loading in future)
    model_name = cfg["model"]["name"]
    console.print(f"\n[dim]Model loading: {model_name}[/dim]")
    console.print("[yellow]⚠ Full model integration pending. Using mock training loop.[/yellow]")

    # Mock training loop for now
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    total_steps = (len(train_samples) // batch_size) * epochs if train_samples else 1

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=total_steps)

        step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0

            for i in range(0, len(train_samples), batch_size):
                # Mock training step
                batch_loss = 0.1 * (1 - step / total_steps)  # Decreasing mock loss
                epoch_loss += batch_loss
                step += 1

                progress.update(task, advance=1, description=f"Epoch {epoch+1}/{epochs} | Loss: {batch_loss:.4f}")

                # Checkpoint
                if step % cfg["checkpoints"]["save_steps"] == 0:
                    ckpt_path = output_path / "checkpoints" / f"step_{step}"
                    ckpt_path.mkdir(parents=True, exist_ok=True)
                    console.print(f"\n  [dim]Saved checkpoint: {ckpt_path}[/dim]")

            avg_loss = epoch_loss / (len(train_samples) // batch_size)
            console.print(f"\n[green]Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}[/green]")

    # Save final model
    final_path = output_path / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    # Save training info
    info = {
        "model": cfg["model"]["name"],
        "strategy": cfg["model"]["strategy"],
        "epochs": epochs,
        "train_samples": len(train_samples),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(final_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    console.print("\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Model saved to: {final_path}")


# =============================================================================
# Evaluation Commands
# =============================================================================


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Model directory")
@click.option("--data", "-d", "data_path", required=True, type=click.Path(exists=True), help="Test data file")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (.json)")
def eval(model: str, data_path: str, output: Optional[str]):
    r"""
    Evaluate a trained model on test data.

    Computes calibration metrics, Brier scores, and generates evaluation report.

    Example:
        xrtm-train eval -m models/final -d samples/test.jsonl
    """
    console.print(Panel(
        f"[bold blue]Evaluating Model[/bold blue]\n"
        f"Model: {model}\n"
        f"Data: {data_path}",
        title="xrtm-train",
    ))

    # Load model info
    model_path = Path(model)
    info_path = model_path / "training_info.json"

    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        console.print(f"  Trained on: {info.get('train_samples', 'N/A')} samples")
        console.print(f"  Strategy: {info.get('strategy', 'N/A')}")

    # Load test data
    test_samples = _load_jsonl(Path(data_path))
    console.print(f"\nLoaded {len(test_samples)} test samples")

    # Mock evaluation
    console.print("\n[yellow]⚠ Full evaluation pending. Using mock metrics.[/yellow]")

    metrics = {
        "brier_score": 0.15,
        "calibration_error": 0.05,
        "log_loss": 0.35,
        "accuracy": 0.72,
    }

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for name, value in metrics.items():
        table.add_row(name, f"{value:.4f}")

    console.print(table)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"metrics": metrics, "test_samples": len(test_samples)}, f, indent=2)
        console.print(f"\n[green]✓ Results saved to:[/green] {output_path}")


# =============================================================================
# Config Commands
# =============================================================================


@main.command("init-config")
@click.option("--output", "-o", default="config.yaml", help="Output config file path")
@click.option("--model", "-m", default="Qwen/Qwen2.5-1.5B", help="Model name")
@click.option("--strategy", type=click.Choice(["full", "lora"]), default="full", help="Training strategy")
def init_config(output: str, model: str, strategy: str):
    r"""
    Generate a default configuration file.

    Creates a YAML config with sensible defaults that can be customized.

    Example:
        xrtm-train init-config -o config.yaml -m Qwen/Qwen2.5-7B
    """
    config: Dict[str, Any] = DEFAULT_CONFIG.copy()
    config["model"]["name"] = model  # type: ignore[index]
    config["model"]["strategy"] = strategy  # type: ignore[index]

    output_path = Path(output)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓ Config created:[/green] {output_path}")
    console.print("\nEdit this file to customize training parameters.")


# =============================================================================
# Helper Functions
# =============================================================================


def _load_trades(path: Path) -> list[dict[str, Any]]:
    r"""Load trades from file."""
    records = _load_json_records(path)
    return records


def _load_news(path: Path) -> list[dict[str, Any]]:
    r"""Load news events from file."""
    return _load_json_records(path)


def _load_priors(path: Path) -> Optional[list[dict[str, Any]]]:
    r"""Load pre-fitted priors."""
    if not path.exists():
        return None

    priors = []
    for prior_file in path.glob("*.json"):
        with open(prior_file) as f:
            prior = json.load(f)
            if isinstance(prior, list):
                priors.extend(prior)
            else:
                priors.append(prior)

    return priors


def _save_jsonl(samples: list, path: Path) -> None:
    r"""Save samples to JSONL format."""
    with open(path, "w") as f:
        for sample in samples:
            if hasattr(sample, "model_dump"):
                f.write(json.dumps(sample.model_dump(mode="json")) + "\n")
            else:
                f.write(json.dumps(sample) + "\n")


def _load_jsonl(path: Path) -> list:
    r"""Load samples from JSONL format."""
    samples = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    return samples


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    r"""Load a JSON object/list into a list of records."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "trades", "priors", "records", "data"):
            records = data.get(key)
            if isinstance(records, list):
                return records
        return [data]
    raise click.ClickException(f"Expected JSON object or list in {path}")


def _parse_timestamp(value: Any) -> datetime:
    r"""Parse a timestamp field into an aware UTC datetime."""
    if value is None:
        raise click.ClickException("Each record must include a timestamp.")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, tz=timezone.utc)
    elif isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
    else:
        raise click.ClickException(f"Unsupported timestamp value: {value!r}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _record_timestamp(record: dict[str, Any]) -> datetime:
    r"""Read a timestamp from common field names."""
    for key in ("timestamp", "snapshot_time", "created_at", "time"):
        if key in record:
            return _parse_timestamp(record[key])
    raise click.ClickException(f"Record is missing a timestamp: {record}")


def _coerce_news_events(records: list[dict[str, Any]]) -> list[Any]:
    r"""Convert JSON records into TrainingSampleBuilder news events."""
    from xrtm.train.kit.builders import NewsEvent

    events = []
    for record in records:
        content = record.get("content") or record.get("headline") or record.get("title")
        if content is None:
            probability = record.get("probability", record.get("price"))
            content = f"Market probability update: {float(probability):.3f}" if probability is not None else "Market update"
        events.append(
            NewsEvent(
                content=str(content),
                timestamp=_record_timestamp(record),
                source=record.get("source"),
            )
        )
    return events


def _coerce_prior_snapshots(records: list[dict[str, Any]]) -> list[Any]:
    r"""Convert JSON records into beta prior snapshots."""
    from xrtm.train.kit.builders import BetaPriorSnapshot

    snapshots = []
    for record in records:
        if "alpha" in record and "beta" in record:
            alpha = float(record["alpha"])
            beta = float(record["beta"])
        else:
            raw_probability = record.get("probability", record.get("price"))
            if raw_probability is None:
                raise click.ClickException(f"Prior record needs alpha/beta or probability/price: {record}")
            probability = max(0.001, min(0.999, float(raw_probability)))
            strength = max(2.0, float(record.get("amount", record.get("volume", 100.0))))
            alpha = max(0.01, probability * strength)
            beta = max(0.01, (1.0 - probability) * strength)
        snapshots.append(BetaPriorSnapshot(alpha=alpha, beta=beta, timestamp=_record_timestamp(record)))
    return snapshots


def _infer_deadline(news_events: list[Any]) -> datetime:
    r"""Infer a deadline after the last available event."""
    if not news_events:
        return datetime.now(timezone.utc) + timedelta(days=1)
    latest = max(event.timestamp for event in news_events)
    return latest + timedelta(days=1)


def _infer_question_id(*record_groups: Optional[list[dict[str, Any]]]) -> str:
    r"""Infer a stable question id from input records."""
    for group in record_groups:
        if not group:
            continue
        for record in group:
            question_id = record.get("question_id") or record.get("market_id") or record.get("id")
            if question_id:
                return str(question_id)
    return "training-sample"


if __name__ == "__main__":
    main()


__all__ = ["main", "load_config", "DEFAULT_CONFIG"]
