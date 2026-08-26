#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/retainrec-pycache"

PYTHONPATH=src python3 -m compileall -q src tests
PYTHONPATH=src python3 -m unittest discover -s tests -v

if command -v ruff >/dev/null 2>&1; then
  ruff check src tests
fi
