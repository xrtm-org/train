# xrtm-train v0.3.0

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**The Optimization Layer for XRTM.**

`xrtm-train` provides backtesting, trace replay, and training sample construction for forecast evaluation.

## Installation

```bash
pip install xrtm-train
```

## Components

### Backtester
Run an Agent against (question, resolution) pairs and compute Brier scores.

```python
from xrtm.train import Backtester
```

### BacktestRunner
Run a full Orchestrator workflow graph against a dataset with ECE and slice analytics.

```python
from xrtm.train import BacktestRunner
```

### TraceReplayer
Save and replay execution traces for deterministic evaluation.

```python
from xrtm.train import TraceReplayer
```

### TrainingSampleBuilder
Build training samples using Teacher Forcing (current prior = previous ground truth).

```python
from xrtm.train.kit.builders import TrainingSampleBuilder
```

## Dependencies

- `pydantic`
- `xrtm-data`
- `xrtm-eval`
- `xrtm-forecast`

## License

Apache 2.0
