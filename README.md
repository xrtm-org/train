# xrtm-train

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/xrtm-train.svg)](https://pypi.org/project/xrtm-train/)

**The Optimization Layer for XRTM.**

`xrtm-train` is the engine that closes the loop. It simulates history by replaying agents against past "Ground Truth" snapshots stored in `xrtm-data`, scoring them with `xrtm-eval`, and optimizing their reasoning parameters.

## Part of the XRTM Ecosystem

```
Layer 4: xrtm-train    → (imports all) ← YOU ARE HERE
Layer 3: xrtm-forecast → (imports eval, data)
Layer 2: xrtm-eval     → (imports data)
Layer 1: xrtm-data     → (zero dependencies)
```

`xrtm-train` sits at the top of the stack and can import from ALL other packages. **Installing `xrtm-train` gives you the full XRTM stack.**

## Installation

```bash
pip install xrtm-train
```

> This automatically installs `xrtm-forecast`, `xrtm-eval`, and `xrtm-data`.

## Core Primitives

### The Simulation Loop
The `Backtester` orchestrates the simulation. It ensures strict temporal isolation—agents are never exposed to data from the future.

```python
from xrtm.train import Backtester

# Initialize components
backtester = Backtester(agent=my_agent, evaluator=my_evaluator)

# Run simulation
results = await backtester.run(dataset=historical_questions)
print(f"Mean Brier Score: {results.mean_score}")
```

### Examples (v0.1.2+)
With the v0.6.0 architecture split, calibration and replay examples now live here:

*   **[Calibration Demo](examples/kit/run_calibration_demo.py)**: Adjusting confidence intervals to match reality.
*   **[Trace Replay](examples/kit/run_trace_replay.py)**: Re-running a saved execution for debugging.
*   **[Evaluation Harness](examples/kit/run_evaluation_harness.py)**: End-to-end backtest with metrics.

## Project Structure

```
src/xrtm/train/
├── core/            # Interfaces & Schemas
│   └── eval/            # Calibration (PlattScaler, BetaScaler)
├── kit/             # Training utilities
│   ├── memory/          # Replay buffers
│   └── optimization/    # Training strategies
├── simulation/      # Backtester, TraceReplayer
└── providers/       # Remote training backends (future)
```

## Development

Prerequisites:
- [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```
