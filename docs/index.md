# xrtm-train Documentation

**The Optimizer.**

## Quick Links
- **[API Reference](api.md)**
- **[Concepts](concepts/)**
    - [Benchmark orchestration](concepts/benchmark_orchestration.md)
    - [Simulation Loop](concepts/simulation_loop.md)

## Overview
`xrtm-train` orchestrates backtesting, execution-trace replay, and parameter
optimization loops.

That same layer boundary is where benchmark execution belongs: replay over
corpora, repeated comparisons, and live-benchmark submission workflows should
all be coordinated here instead of in the product shell.
