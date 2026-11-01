#!/usr/bin/env bash
# Train the final YOLOS on all train+val images, then score the test images once. Safe to rerun after a stop.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
"$YOLO" scripts/final_yolos.py "$@" 2>&1 | tee -a final_yolos.log
