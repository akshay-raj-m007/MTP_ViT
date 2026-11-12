"""Paired grouped bootstrap of two cross-validated setups, e.g. pitch32 against unscaled v2 for one detector.

Uses the pooled out-of-fold predictions that `cv5.py summary` saved for each setup (its best pooled-mAP epoch).
Each resample draws the independent units of train+val with replacement (a defect group from split_v2 counts as
one unit, as in the fold assignment), puts the same images into both setups, and scores both. Reports the observed
difference, a 95% percentile interval and the share of resamples in which the first setup wins.

Both setups keep their own best epoch, picked on these same predictions, so the interval is slightly optimistic for
both sides; the difference is affected less than either score.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import numpy as np
from cv5 import SETUPS, Runs, truth
from sem_yolos.common import ROOT, read, write
from sem_yolos.metrics import coco_ap


def by_name(setup, runs):
    """Ground truth and out-of-fold predictions for one setup, keyed by image file name."""
    oof = read(runs.base/'summary'/f'{setup}_oof.json')
    coco = truth(runs, SETUPS[setup]['data'], oof['folds'])
    names = {im['id']: Path(im['file_name']).name for im in coco['images']}
    if len(set(names.values())) != len(names):
        raise ValueError(f'{setup}: duplicate image names in the pooled folds')
    images = {names[im['id']]: im for im in coco['images']}
    annotations, predictions = {n: [] for n in images}, {n: [] for n in images}
    for a in coco['annotations']:
        annotations[names[a['image_id']]].append(a)
    for p in oof['predictions']:
        predictions[names[p['image_id']]].append(p)
    return dict(epoch=oof['epoch'], folds=oof['folds'], categories=coco['categories'], images=images, annotations=annotations, predictions=predictions)


def units(names):
    plan = read(ROOT/'data/splits/split_v2.json')
    grouped, result = set(), []
    for members in plan.get('groups', {}).values():
        inside = [m for m in members if m in names]
        if inside:
            result.append(inside); grouped.update(inside)
    return result+[[n] for n in sorted(names) if n not in grouped]


def score(data, drawn):
    """mAP and mAP50 on the drawn units; a unit drawn twice gets fresh image ids for its second copy."""
    images, annotations, predictions, next_id = [], [], [], 1
    for unit in drawn:
        for name in unit:
            image_id, next_id = next_id, next_id+1
            images.append(dict(data['images'][name], id=image_id))
            annotations += [dict(a, image_id=image_id) for a in data['annotations'][name]]
            predictions += [dict(p, image_id=image_id) for p in data['predictions'][name]]
    metrics = coco_ap(dict(images=images, annotations=annotations, categories=data['categories']), predictions)
    return metrics['map'], metrics['map50']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('first', choices=list(SETUPS)); p.add_argument('second', choices=list(SETUPS))
    p.add_argument('--resamples', type=int, default=2000); p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(); runs = Runs(False)
    a, b = by_name(args.first, runs), by_name(args.second, runs)
    if set(a['images']) != set(b['images']) or a['folds'] != b['folds']:
        raise SystemExit('The two setups were not scored on the same images and folds')
    pool = units(set(a['images']))
    observed = np.array(score(a, pool))-np.array(score(b, pool))
    rng = np.random.default_rng(args.seed); differences = []
    for k in range(args.resamples):
        drawn = [pool[i] for i in rng.integers(len(pool), size=len(pool))]
        first, second = score(a, drawn), score(b, drawn)
        if None in first+second:
            continue
        differences.append(np.array(first)-np.array(second))
        if (k+1) % 200 == 0:
            print(f'{k+1} of {args.resamples} resamples', flush=True)
    differences = np.array(differences)
    rows = read(runs.base/'summary/summary.json')
    per_fold = {r['setup']: r['per_fold_map'] for r in rows}
    wins = sum(x > y for x, y in zip(per_fold[args.first], per_fold[args.second])) if args.first in per_fold and args.second in per_fold else None
    result = dict(first=args.first, second=args.second, epochs=[a['epoch'], b['epoch']], units=len(pool), images=len(a['images']),
                  resamples_used=len(differences), seed=args.seed, folds_won_by_first=wins, folds=len(a['folds']))
    print(f"\n{args.first} ({a['epoch']}) minus {args.second} ({b['epoch']}): {len(pool)} units, {len(a['images'])} images, {len(differences)} resamples")
    for i, metric in enumerate(['map', 'map50']):
        low, high = np.percentile(differences[:, i], [2.5, 97.5])
        share = float((differences[:, i] > 0).mean())
        result[metric] = dict(difference=float(observed[i]), interval_95=[float(low), float(high)], share_first_better=share)
        print(f'  {metric:6} difference {observed[i]:+.3f}   95% interval [{low:+.3f}, {high:+.3f}]   first better in {share:.1%} of resamples')
    if wins is not None:
        print(f'  first setup better in {wins} of {len(a["folds"])} folds')
    write(runs.base/'summary'/f'compare_{args.first}_vs_{args.second}.json', result)


if __name__ == '__main__': main()
