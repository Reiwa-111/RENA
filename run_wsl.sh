#!/usr/bin/env bash
# Run RENA-RUNE using the Python interpreter selected in PATH or via PY.
# Usage: PY=/path/to/python bash run_wsl.sh train --steps 50000
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python3}"

exec "$PY" RENA_RUNE.py "$@"
