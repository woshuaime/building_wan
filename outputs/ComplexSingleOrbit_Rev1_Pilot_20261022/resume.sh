#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/mnt/windowsE/chuanjun/building_wan/outputs/ComplexSingleOrbit_Rev1_Pilot_20261022
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR="$RUN_ROOT/tmp"
exec /home/shi/miniconda3/envs/scj/bin/python -u "$RUN_ROOT/scripts/run_pilot.py" --resume
