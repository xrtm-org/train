# End-to-End Training Examples

This directory contains examples demonstrating the complete xRTM training pipeline.

## Quick Start

```bash
# Dry run (data collection only)
./run_e2e.sh --dry-run

# Full pipeline (requires GPU)
./run_e2e.sh --output-dir ./my_run
```

## run_end_to_end_training.py

A complete, minimal example that demonstrates:

1. **Data Collection** — Fetches REAL trades from Polymarket Gamma/CLOB APIs
2. **CoT Generation** — Generates reasoning traces using Qwen 0.5B
3. **Sample Building** — Creates training samples with prior injection
4. **Fine-tuning** — Trains Qwen model (requires ~16GB RAM for CPU, or small GPU)
5. **Evaluation** — Computes metrics on the trained model

### Requirements

```bash
# CPU only (synthetic data, mock training)
pip install xrtm-train

# Full GPU training
pip install xrtm-train[gpu]
```

### Configuration

Edit the `CONFIG` dict in the script to adjust:

- `num_questions`: Number of markets to process (default: 3)
- `timestamps_per_question`: Time steps per market (default: 2)
- `reasoning_model`: Model for CoT generation (default: Qwen 0.5B)
- `training_model`: Model to fine-tune (default: Qwen 0.5B)
- `num_epochs`: Training epochs (default: 1)

### Output

```
output/e2e_run/
├── config.json           # Configuration used
├── raw_markets.json      # Collected market data
├── cot_reasoning.json    # Generated reasoning traces
├── training_samples.jsonl # Training data
├── checkpoints/          # Training checkpoints
├── final_model/          # Final trained model
├── evaluation.json       # Evaluation metrics
└── summary.json          # Run summary
```

## Other Examples

- `kit/run_calibration_demo.py` — Calibration evaluation demo
- `kit/run_evaluation_harness.py` — Full evaluation harness
- `kit/run_trace_replay.py` — Replay and analyze traces
