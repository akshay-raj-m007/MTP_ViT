"""Split train+val of split_v2 into grouped folds; test stays held out. Writes one split plan per fold."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import random
from collections import Counter
from pathlib import Path
from sem_yolos.common import ROOT, CLASSES, read, write

FEATURES = ['gap_defects', 'multi_bridge', 'line_collapse', 'bridge', 'clean_images', 'images']
WEIGHTS = dict(gap_defects=8., multi_bridge=4., line_collapse=2., bridge=2., clean_images=1., images=1.)


def box_counts(prepared):
    counts = {}
    for split in ('train', 'val', 'test'):
        data = read(prepared/f'{split}.json')
        names = {im['id']: Path(im['file_name']).name for im in data['images']}
        counts.update({name: Counter() for name in names.values()})
        for ann in data['annotations']:
            counts[names[ann['image_id']]][CLASSES[ann['category_id']]] += 1
    return counts


def units(plan, counts):
    """Each independent defect or single image, with what it would add to a fold."""
    assignments, groups = plan['assignments'], plan.get('groups', {})
    grouped = {m for members in groups.values() for m in members}
    pool = [(name, members) for name, members in groups.items() if assignments[members[0]] != 'test']
    pool += [(name, [name]) for name, split in assignments.items() if split != 'test' and name not in grouped]
    result = []
    for name, members in pool:
        boxes = sum((counts[m] for m in members), Counter())
        result.append(dict(name=name, members=members, gap_defects=int(boxes['gap'] > 0), multi_bridge=boxes['multi_bridge'],
                           line_collapse=boxes['line_collapse'], bridge=boxes['bridge'],
                           clean_images=sum(not counts[m] for m in members), images=len(members)))
    return result


def assign(pool, folds, seed):
    random.Random(seed).shuffle(pool)
    pool.sort(key=lambda u: tuple(-u[f] for f in FEATURES))
    totals = [{f: 0 for f in FEATURES} for _ in range(folds)]
    members = [[] for _ in range(folds)]
    for unit in pool:
        cost = lambda k: sum(WEIGHTS[f]*(totals[k][f]+unit[f])**2 for f in FEATURES)
        best = min(range(folds), key=lambda k: (cost(k), k))
        members[best].extend(unit['members'])
        for f in FEATURES:
            totals[best][f] += unit[f]
    return members, totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', default='data/splits/split_v2.json'); p.add_argument('--prepared', default='data/prepared/label_studio_v3_pitch32')
    p.add_argument('--output', default='data/splits/cv5'); p.add_argument('--folds', type=int, default=5); p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    plan = read(ROOT/args.plan)
    members, totals = assign(units(plan, box_counts(ROOT/args.prepared)), args.folds, args.seed)
    for fold, held_out in enumerate(members):
        assignments = {name: 'test' if split == 'test' else 'val' if name in held_out else 'train' for name, split in plan['assignments'].items()}
        write(ROOT/args.output/f'fold{fold}.json', dict(description=f'Fold {fold} of {args.folds} over train+val of {args.plan}. val is the held-out fold; test is unchanged.',
                                                        groups=plan['groups'], assignments=assignments))
    for fold, total in enumerate(totals):
        print(f'fold{fold}', total)


if __name__ == '__main__': main()
