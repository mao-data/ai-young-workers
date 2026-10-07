#!/usr/bin/env bash
# Rebuild every number, table, and figure from a cold start (~790 MB download).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 ai_young_workers.ipynb
