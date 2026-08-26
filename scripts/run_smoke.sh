#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/retainrec-pycache"

PYTHONPATH=src python3 -m retainrec.cli smoke --config configs/experiments/smoke.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
