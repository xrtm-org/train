# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
