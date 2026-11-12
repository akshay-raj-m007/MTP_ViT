#!/usr/bin/env bash
# Final train+val fit and one test run for every cross-validated setup without a final yet. Safe to rerun after a stop.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
"$YOLO" scripts/final_test.py "$@" 2>&1 | tee -a final_test.log
