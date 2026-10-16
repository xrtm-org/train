# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-10-16

### Fixed
- Malformed resolutions no longer hang the backtester: the error path cannot raise
  (`_safe_resolution_payload`), and failure placeholders are excluded from mean
  score / ECE / slices (`summary_statistics["errors"]` reports the count).
- `Backtester` normalizes ground truth through `normalize_binary_outcome`, matching
  `BacktestRunner` (both APIs now hand evaluators floats, not strings).
- CI workspace maps `xrtm-train` as a source (install fix).

## [0.3.2] - 2026-09-26

### Changed
- Subpackage marker release (packaging alignment with the 0.3.x line).

## [0.3.0] - 2026-09-10

### Added
- Real-question E2E evaluation path (`real_e2e.evaluate_forecast_records_with_backtest_runner`).
- `SliceAnalytics` per-slice summaries in evaluation reports.

## [0.2.9] - 2026-06-20

### Fixed
- Trace replay determinism and coarse clock handling.

## [0.2.8] - 2026-05-18

### Changed
- Standardize train artifact terminology around evaluation paths, reasoning traces, and reporting lanes.
- Require the coordinated release floors for `xrtm-data>=0.2.7`, `xrtm-eval>=0.2.7`, and `xrtm-forecast>=0.6.11`.

### Fixed
- Normalize artifact aliases so released train remains compatible with current sibling package schemas.

## [0.2.7] - 2026-05-10

### Added
- Export benchmark orchestration artifacts from the public package surface and document the orchestration workflow.

### Changed
- Require the coordinated release floors for `xrtm-data`, `xrtm-eval`, and `xrtm-forecast`.

## [0.2.6] - 2026-04-30

### Fixed
- Hardened backtest/replay resolution validation, payload preservation, deterministic failure entries, and invalid binary outcome handling.

## [0.2.1] - 2026-02-17

### Fixed
- **CI/CD**: Fixed dependency resolution in CI by switching `xrtm-data`, `xrtm-eval`, and `xrtm-forecast` to git dependencies.
- **Type Hints**: Resolved `mypy` error in projectors module (`Unsupported operand types`).
- **Audit Remediation**: Applied strict linter fixes and license header enforcement.

## [0.2.0] - 2026-02-04

### Changed
- **Architecture**: Restructured to `core/kit/providers` hierarchy for consistency with xrtm-forecast
- **kit/**: Added `__init__.py` for proper module discovery
- **providers/**: Added empty directory for future training backends
- **README**: Added Project Structure section

## [0.1.2] - 2026-01-28

### Added
- Calibration examples moved from xrtm-forecast
- Trace replay utilities

## [0.1.1] - 2026-01-27

### Added
- Beta calibration scalers

## [0.1.0] - 2026-01-27

### Added
- Initial release
- `Backtester` for temporal simulation
- `TraceReplayer` for offline debugging
- Memory management utilities
