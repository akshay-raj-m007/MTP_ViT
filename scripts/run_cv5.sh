#!/usr/bin/env bash
# Run the five-fold cross-validation, one training at a time. Safe to rerun after a stop: it resumes.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
"$YOLO" scripts/cv5.py "$@" 2>&1 | tee -a cv5.log
