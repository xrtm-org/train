#!/bin/bash
# Run the end-to-end training pipeline.
#
# This script sets up the correct PYTHONPATH to resolve local dev packages.
# Usage:
#   ./run_e2e.sh [--dry-run] [--output-dir ./output]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${TRAIN_ROOT}/../data"

export PYTHONPATH="${DATA_ROOT}/src:${TRAIN_ROOT}/src:${PYTHONPATH}"

cd "$TRAIN_ROOT"
python examples/run_end_to_end_training.py "$@"
