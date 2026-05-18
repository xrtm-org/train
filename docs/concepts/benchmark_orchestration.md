# Benchmark orchestration

`xrtm-train` owns the **benchmark execution loop** for the XRTM stack.

Benchmarking is a specialized simulation loop: load an admissible corpus, run a
forecasting configuration under a fixed history boundary, score the result, and
compare it against alternatives.

## What belongs here

- offline benchmark sweeps
- historical replay and backtesting
- repeated experiment orchestration
- live competition submission workflows
- benchmark artifact generation for downstream reporting

Concrete orchestration artifacts include:

- `BenchmarkRunSpec`
- `BenchmarkRunResultBundle`
- `BenchmarkSuiteSpec`
- `BenchmarkSuiteResult`
- `ExternalBenchmarkSourceSpec`
- `ExternalBenchmarkLaneSpec`
- `ExternalBenchmarkLaneResult`

## Internal stress suites vs. public external ingestion

`BenchmarkSuiteSpec` and `BenchmarkSuiteResult` remain the contract for
**reproducible internal stress suites**. Their `arms` are systems XRTM can
actually run repeatedly on one frozen corpus slice.

Public comparison material belongs in the separate external lane:

- public human baselines
- public leaderboard snapshots
- inspectable competitor outputs

That lane is modeled by `ExternalBenchmarkLaneSpec` and
`ExternalBenchmarkLaneResult`. Its canonical serialized field is
`evaluation_path` (`reporting_lane` remains a compatibility alias), and it is
intentionally ingestion/reporting-only:

- no repeated-arm runner semantics
- no claim that a third-party system is locally reproducible
- explicit source provenance for imported public evidence

This keeps `xrtm-train` honest: execution artifacts stay reproducible, while
public comparison artifacts stay inspectable and separately labeled.

## Why it belongs here

The governance rule for XRTM is that logic which **iterates over time or over a
corpus at scale** belongs in `xrtm-train`.

That keeps:

- `xrtm-data` focused on corpora and provenance
- `xrtm-eval` focused on scoring
- `xrtm` focused on thin product surfaces

## Design rule

If a benchmark feature needs to coordinate many runs, many questions, or many
model variants, it should be implemented in `xrtm-train` and consume the lower
layers rather than rebuilding them.
