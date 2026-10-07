#!/usr/bin/env bash
# Rebuild every number, table, and figure from a cold start.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python src/01_download.py   # CPS Jan 2020 - Aug 2026 (~790 MB) + exposure scores
.venv/bin/python src/02_exposure.py   # O*NET -> SOC -> Census occupation exposure
.venv/bin/python src/03_clean.py      # fixed-width CPS -> typed Parquet + cleaning waterfall
.venv/bin/python src/04_analysis.py   # figures, event study, robustness table
