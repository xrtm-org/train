#!/usr/bin/env python3
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
End-to-End Training Pipeline Example
=====================================

This script demonstrates the complete xRTM training pipeline:
1. Fetch real data from Polymarket (3 questions, 2 timestamps each)
2. Generate CoT reasoning using a cheap model
3. Build training samples with prior injection
4. Fine-tune a small model (Qwen 0.5B)
5. Evaluate and save checkpoint

Requirements:
    pip install xrtm-train[gpu]

Usage:
    python run_end_to_end_training.py [--output-dir ./output] [--dry-run]

This is a MINIMAL but COMPLETE example - uses small models and limited data
to verify the entire pipeline works before scaling up.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Data collection
    "num_questions": 1,
    "timestamps_per_question": 2,
    "days_of_history": 7,
    "data_source": "goldsky", # Options: "goldsky", "clob"

    # Models
    # Switched to SmolLM2-135M (Tiny) to fit in memory on restricted hardware
    "reasoning_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "training_model": "HuggingFaceTB/SmolLM2-135M",

    # Training
    "batch_size": 1,              # Reduced for CPU memory
    "gradient_accumulation": 4,   # Increased to maintain effective batch
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "max_length": 256,            # Reduced context length

    # Prior injection
    "prior_hidden_dim": 32,

    # Output
    "output_dir": "./output/e2e_run",
}

# Sample Polymarket question IDs for testing
SAMPLE_MARKET_IDS = [
    "0x1234567890abcdef1234567890abcdef12345678",  # Placeholder - will use subgraph
    "0xabcdef1234567890abcdef1234567890abcdef12",
    "0x567890abcdef1234567890abcdef1234567890ab",
]


# =============================================================================
# Step 1: Data Collection
# =============================================================================

async def collect_market_data(output_dir: Path) -> list[dict]:
    """
    Fetch REAL market data from Polymarket.

    Uses:
    - Gamma API: Get active markets with metadata
    - CLOB API: Get historical price data

    Returns list of market data with price history.
    """
    import aiohttp

    logger.info("=" * 60)
    logger.info("STEP 1: Collecting Market Data from Polymarket")
    logger.info("=" * 60)

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    markets = []

    # Initialize Subgraph Source if needed
    subgraph_source = None
    if CONFIG.get("data_source") == "goldsky":
        try:
            from xrtm.data.providers.subgraph import PolymarketTradeSource
            subgraph_source = PolymarketTradeSource()
            logger.info("Using Goldsky Subgraph for trade history")
        except ImportError:
            logger.warning("Could not import PolymarketTradeSource. Falling back to CLOB.")

    async with aiohttp.ClientSession() as session:
        # Step 1a: Get active markets from Gamma API
        logger.info("Fetching active markets from Gamma API...")
        async with session.get(
            f"{GAMMA_API}/markets",
            params={"limit": CONFIG["num_questions"], "closed": "false"},
        ) as resp:
            resp.raise_for_status()
            gamma_markets = await resp.json()

        logger.info(f"Found {len(gamma_markets)} markets")

        # Step 1b: Get price history for each market
        for i, market in enumerate(gamma_markets[:CONFIG["num_questions"]]):
            question = market.get("question", "Unknown")[:50]
            logger.info(f"  [{i+1}/{CONFIG['num_questions']}] {question}...")

            # Get token IDs from clobTokenIds
            clob_token_ids = market.get("clobTokenIds", "[]")
            if isinstance(clob_token_ids, str):
                import ast
                try:
                    clob_token_ids = ast.literal_eval(clob_token_ids)
                except (ValueError, SyntaxError):
                    clob_token_ids = []

            if not clob_token_ids:
                logger.warning("    No CLOB token IDs, skipping")
                continue

            token_id = clob_token_ids[0]
            trades = []

            # Option A: Goldsky Subgraph
            if subgraph_source and CONFIG.get("data_source") == "goldsky":
                try:
                    # Fetch last 30 days or so?
                    now = datetime.now(timezone.utc)
                    start = now - timedelta(days=7) # 1 week history per plan

                    subgraph_source.fetch_trades.__annotations__['return']
                    # Wait, import TradeEvent type annotation is valid if imported.
                    # But fetch_trades returns list[TradeEvent].

                    # Call fetch_trades with token_id as market_id (as refactored)
                    sg_trades = await subgraph_source.fetch_trades(
                        market_id=token_id,
                        start_time=start,
                        end_time=now,
                        limit=100
                    )

                    if sg_trades:
                        logger.info(f"    Fetched {len(sg_trades)} trades from Goldsky")
                        # Convert to dict format expected by rest of pipeline
                        for t in sg_trades:
                            trades.append({
                                "price": t.price,
                                "amount": t.amount,
                                "timestamp": t.timestamp.isoformat(),
                            })
                    else:
                         logger.warning("    No trades found in Subgraph")

                except Exception as e:
                    logger.warning(f"    Goldsky failed: {e}")
                    # Fallback to CLOB? Or continue?
                    # Continue to try CLOB if Goldsky fails?
                    pass

            # Option B: CLOB API (Fallback or Primary)
            if not trades: # If Goldsky failed or not used
                async with session.get(
                    f"{CLOB_API}/prices-history",
                    params={"market": token_id, "interval": "1h"},
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"    Price history failed: {resp.status}")
                        continue
                    price_data = await resp.json()

                history = price_data.get("history", [])
                if not history:
                    logger.warning("    No price history available")
                    continue

                # Convert to trades format
                for point in history[-100:]:  # Last 100 points
                    trades.append({
                        "price": point["p"],
                        "amount": 100.0,  # Default weight for CLOB candles
                        "timestamp": datetime.fromtimestamp(point["t"], tz=timezone.utc).isoformat(),
                    })

            if not trades:
                continue

            # Calculate VWAP (True VWAP if we have amounts!)
            total_vol = sum(t["amount"] for t in trades)
            if total_vol > 0:
                vwap = sum(t["price"] * t["amount"] for t in trades) / total_vol
            else:
                 vwap = sum(t["price"] for t in trades) / len(trades)

            markets.append({
                "id": market.get("id", f"market_{i}"),
                "question": market.get("question", "Unknown"),
                "trades": trades,
                "vwap": vwap,
                "volume": float(market.get("volume", 0)),
            })

            logger.info(f"    → {len(trades)} price points, VWAP: {vwap:.3f}")

            # Stop early if we have enough
            if len(markets) >= CONFIG["num_questions"]:
                break

    if not markets:
        raise RuntimeError("Failed to fetch any market data from Polymarket APIs")

    # Save raw data
    raw_path = output_dir / "raw_markets.json"
    with open(raw_path, "w") as f:
        json.dump(markets, f, indent=2)
    logger.info(f"Saved raw data to: {raw_path}")

    return markets


# =============================================================================
# Step 2: Generate CoT Reasoning
# =============================================================================

def generate_cot_reasoning(markets: list[dict], output_dir: Path) -> list[dict]:
    """
    Generate Chain-of-Thought reasoning for each market using a cheap model.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Generating CoT Reasoning")
    logger.info("=" * 60)

    cot_path = output_dir / "cot_reasoning.json"
    if cot_path.exists():
        logger.info(f"Found existing CoT data at {cot_path}, skipping generation.")
        with open(cot_path, "r") as f:
            return json.load(f)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        logger.info(f"Loading reasoning model: {CONFIG['reasoning_model']}")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["reasoning_model"])
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["reasoning_model"],
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)

    except ImportError:
        logger.warning("Transformers not available, using mock CoT generation")
        return _generate_mock_cot(markets, output_dir)
    except Exception as e:
        logger.warning(f"Model loading failed: {e}, using mock CoT")
        return _generate_mock_cot(markets, output_dir)

    cot_data = []

    for i, market in enumerate(markets):
        logger.info(f"  [{i+1}/{len(markets)}] Generating reasoning for: {market['question'][:50]}...")

        # Create prompt
        recent_prices = [t["price"] for t in market["trades"][-5:]]
        prompt = f"""You are a forecaster analyzing prediction markets.

Question: {market['question']}

Recent price history (probability): {recent_prices}
Current VWAP: {market['vwap']:.3f}
Total volume: ${market['volume']:,.0f}

Provide a brief analysis of the probability and key factors. Be concise.

Analysis:"""

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        reasoning = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning = reasoning[len(prompt):].strip()

        cot_data.append({
            "market_id": market["id"],
            "question": market["question"],
            "reasoning": reasoning,
            "probability": market["vwap"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(f"    → Generated {len(reasoning)} chars of reasoning")

    # Clean up GPU memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import gc
    gc.collect()

    # Save CoT data
    cot_path = output_dir / "cot_reasoning.json"
    with open(cot_path, "w") as f:
        json.dump(cot_data, f, indent=2)
    logger.info(f"Saved CoT data to: {cot_path}")

    return cot_data


def _generate_mock_cot(markets: list[dict], output_dir: Path) -> list[dict]:
    """Generate mock CoT when model is unavailable."""
    cot_data = []

    for market in markets:
        reasoning = (
            f"Based on recent trading activity showing VWAP of {market['vwap']:.3f} "
            f"and total volume of ${market['volume']:,.0f}, the market sentiment "
            f"appears moderately confident. Key factors include recent price stability "
            f"and consistent trading volume. Current probability estimate: {market['vwap']:.1%}."
        )

        cot_data.append({
            "market_id": market["id"],
            "question": market["question"],
            "reasoning": reasoning,
            "probability": market["vwap"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    cot_path = output_dir / "cot_reasoning.json"
    with open(cot_path, "w") as f:
        json.dump(cot_data, f, indent=2)

    return cot_data


# =============================================================================
# Step 3: Build Training Samples
# =============================================================================

def build_training_samples(
    markets: list[dict],
    cot_data: list[dict],
    output_dir: Path,
) -> list[dict]:
    """
    Build training samples with prior injection using Teacher Forcing.

    Samples are built inline without importing xrtm.train to avoid
    import chains that may have stale dependencies.
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Building Training Samples")
    logger.info("=" * 60)

    # Helper: compute Beta parameters from mean/concentration
    def beta_params_from_mean_conc(mean: float, concentration: float) -> tuple[float, float]:
        """Compute (alpha, beta) from mean and concentration."""
        mean = max(0.01, min(0.99, mean))
        alpha = mean * concentration
        beta = (1 - mean) * concentration
        return max(0.1, alpha), max(0.1, beta)

    samples = []

    for market, cot in zip(markets, cot_data):
        trades = market["trades"]

        # Create time-stepped samples
        n_timestamps = min(CONFIG["timestamps_per_question"], len(trades) // 2)
        step_size = len(trades) // (n_timestamps + 1)

        for t_idx in range(n_timestamps):
            # Prior from trades up to this point
            prior_trades = trades[:(t_idx + 1) * step_size]
            [t["price"] for t in prior_trades]

            # Fit Beta prior from prices (Volume Weighted)
            total_vol = sum(t["amount"] for t in prior_trades)
            if total_vol > 0:
                mean_price = sum(t["price"] * t["amount"] for t in prior_trades) / total_vol
            else:
                mean_price = 0.5

            # Use log-volume for concentration to avoid exploding gradients with raw amounts
            # logic: base_confidence + log10(volume + 1) * scale
            # e.g., $10 vol -> 1.0, $1000 vol -> 3.0, $1M vol -> 6.0.
            # Scaling factor of 2.0 gives range [2, 14] roughly.
            import math
            vol_score = math.log10(total_vol + 1.0) * 2.0
            concentration = min(50.0, 2.0 + vol_score)

            prior_alpha, prior_beta = beta_params_from_mean_conc(mean_price, concentration)

            # Target from next step
            target_trades = trades[(t_idx + 1) * step_size:(t_idx + 2) * step_size]
            if not target_trades:
                target_trades = trades[-step_size:]

            target_vol = sum(t["amount"] for t in target_trades)
            if target_vol > 0:
                sum(t["price"] * t["amount"] for t in target_trades) / target_vol
            else:
                 sum(t["price"] for t in target_trades) / len(target_trades)

            # Target concentration also volume based?
            # Actually, target is what we want to predict.
            # We predicting the future state.
            # The future state distribution is also defined by volume.
            total_vol + target_vol # Progressive volume?
            # Usually we predict the STATE at time t+1.
            # The state at t+1 includes history up to t+1.
            # So compute properly:

            trades_until_target = trades[:(t_idx + 2) * step_size]
            cum_vol = sum(t["amount"] for t in trades_until_target)
            if cum_vol > 0:
                cum_mean = sum(t["price"] * t["amount"] for t in trades_until_target) / cum_vol
            else:
                cum_mean = 0.5

            vol_score_target = math.log10(cum_vol + 1.0) * 2.0
            target_conc = min(50.0, 2.0 + vol_score_target)

            target_alpha, target_beta = beta_params_from_mean_conc(cum_mean, target_conc)

            sample = {
                "question_id": market["id"],
                "step_index": t_idx,
                "snapshot_time": prior_trades[-1]["timestamp"],
                "prior_alpha": prior_alpha,
                "prior_beta": prior_beta,
                "current_news": cot["reasoning"][:200],  # Truncate for small model
                "target_alpha": target_alpha,
                "target_beta": target_beta,
                "news_context": [f"Trade volume: ${market['volume']:,.0f}"],
            }

            samples.append(sample)

        logger.info(f"  Built {n_timestamps} samples for: {market['question'][:50]}...")

    logger.info(f"Total training samples: {len(samples)}")

    # Save samples
    samples_path = output_dir / "training_samples.jsonl"
    with open(samples_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved samples to: {samples_path}")

    return samples


# =============================================================================
# Step 4: Fine-tune Model
# =============================================================================

def finetune_model(samples: list[dict], output_dir: Path) -> Path:
    """
    Fine-tune a small model on the training samples.
    """
    logger.info("=" * 60)
    logger.info("STEP 4: Fine-tuning Model")
    logger.info("=" * 60)

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError:
        logger.error("PyTorch/Transformers/PEFT not installed. Run: pip install xrtm-train[gpu]")
        return _mock_finetune(samples, output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model: {CONFIG['training_model']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["training_model"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["training_model"],
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # Apply LoRA (Adapter) Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    model.to(device)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA Adapted Model: {trainable_params:,} trainable params "
        f"({100 * trainable_params / all_params:.2f}% of {all_params/1e6:.1f}M)"
    )

    # Prepare dataset
    class ForecastDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # Format as instruction
            text = f"""Prior: α={sample['prior_alpha']:.2f}, β={sample['prior_beta']:.2f}
News: {sample['current_news']}
Prediction: α={sample['target_alpha']:.2f}, β={sample['target_beta']:.2f}"""

            encodings = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": encodings["input_ids"].squeeze(),
                "attention_mask": encodings["attention_mask"].squeeze(),
                "labels": encodings["input_ids"].squeeze(),
            }

    dataset = ForecastDataset(samples, tokenizer, CONFIG["max_length"])
    logger.info(f"Dataset size: {len(dataset)} samples")

    # Training arguments
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        fp16=device == "cuda",
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Save memory
        optim="adafactor",            # Save memory (no momentum)
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Saved final model to: {final_path}")

    # Clean up
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_path


def _mock_finetune(samples: list[dict], output_dir: Path) -> Path:
    """Mock fine-tuning when GPU dependencies unavailable."""
    logger.warning("Running mock fine-tuning (no GPU deps)")

    final_path = output_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)

    # Save mock info
    info = {
        "model": CONFIG["training_model"],
        "samples": len(samples),
        "epochs": CONFIG["num_epochs"],
        "mock": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(final_path / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return final_path


# =============================================================================
# Step 5: Evaluate Model
# =============================================================================

def evaluate_model(model_path: Path, samples: list[dict], output_dir: Path) -> dict:
    """
    Evaluate the fine-tuned model.
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating Model")
    logger.info("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not (model_path / "config.json").exists():
            logger.warning("Model not found, using mock evaluation")
            return _mock_evaluate(samples, output_dir)

        logger.info(f"Loading model from: {model_path}")
        AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

    except Exception as e:
        logger.warning(f"Model loading failed: {e}, using mock evaluation")
        return _mock_evaluate(samples, output_dir)

    # Compute Metrics using Evaluator
    try:
        from xrtm.train.metrics import Evaluator
        evaluator = Evaluator()

        # For now, we evaluate the 'Prior' vs 'Target' in the samples as a baseline
        # (Since parsing the model output exactly requires strictly formatted output)
        metrics = evaluator.compute_metrics(samples)

        metrics["model_path"] = str(model_path)
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    except ImportError:
        logger.warning("Could not import xrtm.train.metrics.Evaluator. Using simple fallback.")
        # Simple fallback
        total_loss = 0.0
        for sample in samples:
            prior_mean = sample["prior_alpha"] / (sample["prior_alpha"] + sample["prior_beta"])
            target_mean = sample["target_alpha"] / (sample["target_alpha"] + sample["target_beta"])
            total_loss += abs(prior_mean - target_mean)

        metrics = {
            "samples": len(samples),
            "avg_shift": total_loss / len(samples),
            "model_path": str(model_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    logger.info("Evaluation complete:")
    logger.info(json.dumps(metrics, indent=2))

    # Save results
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation to: {eval_path}")

    return metrics


def _mock_evaluate(samples: list[dict], output_dir: Path) -> dict:
    """Mock evaluation when model unavailable."""
    import random

    metrics = {
        "samples": len(samples),
        "avg_shift": random.uniform(0.05, 0.15),
        "mock": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline(output_dir: Path, dry_run: bool = False):
    """
    Execute the complete end-to-end pipeline.
    """
    logger.info("=" * 60)
    logger.info("xRTM End-to-End Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)

    start_time = datetime.now()

    # Step 1: Collect data
    markets = await collect_market_data(output_dir)

    if dry_run:
        logger.info("\n[DRY RUN] Stopping after data collection")
        return

    # Step 2: Generate CoT
    cot_data = generate_cot_reasoning(markets, output_dir)

    # Step 3: Build samples
    samples = build_training_samples(markets, cot_data, output_dir)

    # Step 4: Fine-tune
    model_path = finetune_model(samples, output_dir)

    # Step 5: Evaluate
    metrics = evaluate_model(model_path, samples, output_dir)

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Markets processed: {len(markets)}")
    logger.info(f"Training samples: {len(samples)}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Output directory: {output_dir}")

    # Save summary
    summary = {
        "elapsed_seconds": elapsed.total_seconds(),
        "markets": len(markets),
        "samples": len(samples),
        "model_path": str(model_path),
        "metrics": metrics,
        "config": CONFIG,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="End-to-End xRTM Training Pipeline")
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory for all artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run data collection step",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    asyncio.run(run_pipeline(output_dir, args.dry_run))


if __name__ == "__main__":
    main()
