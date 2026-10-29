"""Draw ground truth beside low-threshold predictions for manual review (YOLOS or SSD predictions.json).

Writes one PNG per image plus gt_review.csv, which says for every ground-truth box whether the model
found it (same class, IoU>=0.5), found it with the wrong class, localised it poorly, or missed it.
Needs only Pillow, so it runs in either environment.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sem_yolos.common import ROOT, CLASSES, read, write, prepared_dir, resolve_image

COLORS = ["#ff3b30", "#00c7ff", "#ffcc00", "#d65cff"]  # bridge, gap, line_collapse, multi_bridge
PANEL = 640


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0., min(ax+aw, bx+bw)-max(ax, bx)); iy = max(0., min(ay+ah, by+bh)-max(ay, by))
    inter = ix*iy; union = aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.


def font(size):
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def review_box(gt, predictions):
    """Classify one ground-truth box against all predictions on its image (no score cut-off)."""
    same = [(iou(gt['bbox'], p['bbox']), p['score']) for p in predictions if p['category_id'] == gt['category_id']]
    other = [(iou(gt['bbox'], p['bbox']), p['score'], p['category_id']) for p in predictions if p['category_id'] != gt['category_id']]
    hits = [s for v, s in same if v >= .5]
    best_iou, best_iou_score = max(same, default=(0., 0.))
    wrong = max(((s, c) for v, s, c in other if v >= .5), default=None)
    if hits: verdict = 'found'
    elif wrong and wrong[0] >= .05: verdict = f'wrong_class:{CLASSES[wrong[1]]}'
    elif best_iou >= .1: verdict = 'poorly_localised'
    else: verdict = 'missed'
    return dict(verdict=verdict, best_score_iou50=round(max(hits, default=0.), 4), best_same_class_iou=round(best_iou, 3),
                score_at_best_iou=round(best_iou_score, 4),
                wrong_class_score_iou50=round(wrong[0], 4) if wrong else '', wrong_class=CLASSES[wrong[1]] if wrong else '')


def draw_panel(image, boxes, scale, title, fnt, small):
    panel = image.resize((round(image.width*scale), round(image.height*scale)), Image.BICUBIC).convert('RGB')
    canvas = Image.new('RGB', (panel.width, panel.height+28), 'white'); canvas.paste(panel, (0, 28))
    draw = ImageDraw.Draw(canvas); draw.text((4, 4), title, fill='black', font=fnt)
    for b, cls, text, width in boxes:
        x, y, w, h = [v*scale for v in b]
        draw.rectangle([x, y+28, x+w, y+h+28], outline=COLORS[cls], width=width)
        if text:
            tx, ty = min(x+2, canvas.width-draw.textlength(text, font=small)-2), max(28, y+28-16)
            bb = draw.textbbox((tx, ty), text, font=small); draw.rectangle(bb, fill='black'); draw.text((tx, ty), text, fill=COLORS[cls], font=small)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True, help='predictions.json (category_id 0-3, bbox xywh in source pixels)')
    p.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', default='data/prepared/label_studio_v2')
    p.add_argument('--min-score', type=float, default=.05, help='Predictions drawn at or above this score')
    p.add_argument('--max-per-image', type=int, default=12)
    p.add_argument('--threshold', type=float, help='Deployment threshold, shown in the title and CSV')
    args = p.parse_args()
    data = read(prepared_dir(ROOT, args.prepared)/f'{args.split}.json')
    predictions = read(args.predictions)
    by_image = defaultdict(list)
    for pred in predictions: by_image[pred['image_id']].append(pred)
    gts = defaultdict(list)
    for a in data['annotations']: gts[a['image_id']].append(a)
    if not set(by_image) <= {im['id'] for im in data['images']}:
        raise ValueError('Predictions refer to images outside this split; check --split and --prepared')
    args.output.mkdir(parents=True, exist_ok=False)
    fnt, small = font(16), font(14)
    rows = []
    for im in sorted(data['images'], key=lambda i: i['file_name']):
        name = Path(im['file_name']).name
        with Image.open(resolve_image(ROOT, im['file_name'])) as opened: image = opened.convert('RGB')
        scale = PANEL/max(image.size)
        preds = sorted(by_image[im['id']], key=lambda q: -q['score'])
        shown = [q for q in preds if q['score'] >= args.min_score][:args.max_per_image]
        gt_boxes = [(a['bbox'], a['category_id'], CLASSES[a['category_id']], 3) for a in gts[im['id']]]
        pred_boxes = [(q['bbox'], q['category_id'], f"{CLASSES[q['category_id']]} {q['score']:.2f}",
                       3 if args.threshold is not None and q['score'] >= args.threshold else 1) for q in shown]
        verdicts = []
        for a in gts[im['id']]:
            r = review_box(a, preds); verdicts.append(r['verdict'].split(':')[0])
            rows.append(dict(file=name, gt_class=CLASSES[a['category_id']], gt_bbox=[round(v, 1) for v in a['bbox']], **r))
        above = sum(q['score'] >= args.threshold for q in preds) if args.threshold is not None else None
        left = draw_panel(image, gt_boxes, scale, f'{name}  GROUND TRUTH ({len(gt_boxes) or "clean"})', fnt, small)
        right = draw_panel(image, pred_boxes, scale, f'PREDICTIONS score>={args.min_score}'
                           + (f'  thick=above thr {args.threshold} ({above})' if args.threshold is not None else ''), fnt, small)
        sheet = Image.new('RGB', (left.width+right.width+10, max(left.height, right.height)+26), 'white')
        sheet.paste(left, (0, 0)); sheet.paste(right, (left.width+10, 0))
        legend = '   '.join(f'{c}' for c in CLASSES)
        d = ImageDraw.Draw(sheet); x = 4
        for c, colour in zip(CLASSES, COLORS):
            d.text((x, sheet.height-22), c, fill=colour, font=fnt); x += d.textlength(c, font=fnt)+20
        d.text((x, sheet.height-22), '| GT: ' + (', '.join(verdicts) or 'clean'), fill='black', font=fnt)
        prefix = 'clean' if not gt_boxes else '_'.join(sorted({CLASSES[a['category_id']] for a in gts[im['id']]}))
        sheet.save(args.output/f'{prefix}__{Path(name).stem}.png')
    with open(args.output/'gt_review.csv', 'w', newline='') as f:
        fields = ['file', 'gt_class', 'gt_bbox', 'verdict', 'best_score_iou50', 'best_same_class_iou', 'score_at_best_iou', 'wrong_class', 'wrong_class_score_iou50']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows: summary[r['gt_class']][r['verdict'].split(':')[0]] += 1
    summary = {k: dict(v) for k, v in summary.items()}
    write(args.output/'summary.json', dict(predictions=str(args.predictions), split=args.split, min_score=args.min_score,
                                           threshold=args.threshold, images=len(data['images']), gt_verdicts=summary))
    print(args.output); print(summary)


if __name__ == '__main__':
    main()
