#!/usr/bin/env bash
# Train YOLOS and SSD (natural and weighted) on the pitch-normalised data, one after another.
# Safe to rerun: a run with a saved checkpoint resumes, and a finished run stops straight away.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
SSD=outputs/mobilevitv2_ssd/env/bin/python
ENTRY=experiments/mobilevitv2_ssd/scripts/experiment.py
SSD_CONFIGS=experiments/mobilevitv2_ssd/configs

yolos() {  # sampling config
  out="outputs/runs/yolos_pitch32_$1"
  resume=""; [ -f "$out/last/training_state.pt" ] && resume="--resume"
  "$YOLO" scripts/train_yolos.py --config "configs/$2" --sampling "$1" --device cuda --output "$out" $resume 2>&1 | tee -a "yolos_pitch32_$1.log"
}

ssd() {  # sampling
  out="outputs/mobilevitv2_ssd/pitch32_$1_seed42"
  resume=""; [ -f "$out/last.pt" ] && resume="--resume"
  "$SSD" "$ENTRY" train --config "$SSD_CONFIGS/pitch32_$1.json" --seed 42 --device cuda --precision fp32 --output "$out" $resume 2>&1 | tee -a "ssd_pitch32_$1.log"
}

yolos natural pipeline_config_pitch32.json
yolos weighted pipeline_config_pitch32_weighted.json
ssd natural
ssd weighted
echo "ALL FOUR PITCH32 RUNS ENDED"
