#!/usr/bin/env bash
# Evaluate the four pitch-normalised runs, draw predictions and print a summary table.
# Safe to rerun: finished evaluation and drawing folders are skipped.
cd "$(dirname "$0")/.." || exit 1
YOLO="${YOLO:-$HOME/miniconda3/envs/mtp_yolos/bin/python}"
SSD=outputs/mobilevitv2_ssd/env/bin/python
ENTRY=experiments/mobilevitv2_ssd/scripts/experiment.py
PREPARED=data/prepared/label_studio_v3_pitch32
REVIEW=outputs/prediction_review_pitch32

draw() {  # predictions split output [threshold]
  [ -d "$3" ] && return
  "$YOLO" scripts/draw_predictions.py --predictions "$1" --split "$2" --prepared "$PREPARED" --output "$3" ${4:+--threshold "$4"} | tail -1
}

for s in natural weighted; do
  run="outputs/runs/yolos_pitch32_$s"
  [ -d "$run/eval_val" ] || "$YOLO" scripts/evaluate.py --model "$run/best" --split val --select-threshold --device cuda --output "$run/eval_val" > /dev/null
  [ -d "$run/eval_test" ] || "$YOLO" scripts/evaluate.py --model "$run/best" --split test --device cuda --output "$run/eval_test" > /dev/null
  threshold=$("$YOLO" -c "import json,sys;print(json.load(open(sys.argv[1]))['threshold'])" "$run/best/threshold.json")
  draw "$run/eval_val/predictions.json" val "$REVIEW/yolos_$s/val" "$threshold"
  draw "$run/eval_test/predictions.json" test "$REVIEW/yolos_$s/test" "$threshold"

  run="outputs/mobilevitv2_ssd/pitch32_${s}_seed42"
  [ -d "$run/eval_test" ] || "$SSD" "$ENTRY" test --model "$run/best.pt" --device cuda --precision fp32 --output "$run/eval_test" > /dev/null
  draw "$run/best_validation_predictions.json" val "$REVIEW/ssd_$s/val"
  draw "$run/eval_test/predictions.json" test "$REVIEW/ssd_$s/test"
done

"$YOLO" - <<'PY'
import json
def row(name, path):
    m = json.load(open(path))
    per = '  '.join(f"{k} {v['ap']:.3f}" for k, v in m['per_class_ap'].items())
    print(f"{name:26} mAP {m['map']:.3f}  mAP50 {m['map50']:.3f}  {per}")
for s in ('natural', 'weighted'):
    yolos, ssd = f'outputs/runs/yolos_pitch32_{s}', f'outputs/mobilevitv2_ssd/pitch32_{s}_seed42'
    row(f'yolos {s} val', f'{yolos}/eval_val/metrics.json'); row(f'yolos {s} test', f'{yolos}/eval_test/metrics.json')
    row(f'ssd {s} val', f'{ssd}/best_validation_metrics.json'); row(f'ssd {s} test', f'{ssd}/eval_test/metrics.json')
    print('thresholds:', json.load(open(f'{yolos}/best/threshold.json'))['threshold'])
PY
echo "Drawings: $REVIEW"
