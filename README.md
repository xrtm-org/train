# xrtm-train

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**The Optimization Layer for XRTM.**

`xrtm-train` is the engine that closes the loop. It simulates history by replaying agents against past "Ground Truth" snapshots stored in `xrtm-data`, scoring them with `xrtm-eval`, and optimizing their reasoning parameters.

## Installation

```bash
uv pip install xrtm-train
```

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

## Development

Prerequisites:
- [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest
```
