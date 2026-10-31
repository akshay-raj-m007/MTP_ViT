"""Five-fold grouped cross-validation over train+val, on normalised and v2 images. Test is never read here.

Every setup runs a fixed schedule, one epoch count is chosen on the pooled held-out predictions, and all results
stay under outputs/cv5 and outputs/mobilevitv2_ssd/cv5. Safe to rerun: finished folds are skipped, others resume.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import os
import re
import shutil
import subprocess
import time
from draw_predictions import review_box
from sem_yolos.common import ROOT, CATEGORIES, read, write
from sem_yolos.metrics import coco_ap

FOLDS = 5
SOURCES = dict(pitch32=['--source', 'data/pitch_norm/coco.json', '--images', 'data/pitch_norm/images'], v2=[])
SSD_CONFIGS = 'experiments/mobilevitv2_ssd/configs'
SETUPS = dict(
    ssd_pitch32=dict(model='ssd', data='pitch32', config=f'{SSD_CONFIGS}/pitch32_natural.json'),
    ssd_v2=dict(model='ssd', data='v2', config=f'{SSD_CONFIGS}/natural.json'),
    yolos_pitch32=dict(model='yolos', data='pitch32', config='configs/pipeline_config_pitch32.json'),
    yolos_v2=dict(model='yolos', data='v2', config='configs/pipeline_config_20_views.json'),
)
# YOLOS peaked by stage B epoch 13 in every earlier run, and no fold may stop early.
YOLOS_SCHEDULE = dict(stage_b_epochs=15, patience=999)
QUICK = dict(stage_a_epochs=1, stage_b_epochs=1, training_view_mode='random_augmented')


def ssd_python():
    folder = ROOT/'outputs/mobilevitv2_ssd/env'
    return os.environ.get('SSD_PYTHON') or str(folder/('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))


class Runs:
    def __init__(self, quick):
        self.quick, name = quick, 'cv5_quick' if quick else 'cv5'
        self.base, self.ssd = ROOT/'outputs'/name, ROOT/'outputs/mobilevitv2_ssd'/name

    def prepared(self, data, fold):
        return f'{self.base.relative_to(ROOT).as_posix()}/prepared/{data}/fold{fold}'

    def folder(self, setup, fold):
        return (self.ssd if SETUPS[setup]['model'] == 'ssd' else self.base/'runs')/f'{setup}_fold{fold}'

    def config(self, setup, fold):
        path = self.base/'configs'/f'{setup}_fold{fold}.json'
        config = read(ROOT/SETUPS[setup]['config'])
        config.update(prepared=self.prepared(SETUPS[setup]['data'], fold), fixed_schedule=True)
        if SETUPS[setup]['model'] == 'yolos':
            config.update(YOLOS_SCHEDULE)
        if self.quick:
            config.update(QUICK)
        write(path, config)
        return path


def prepare(runs, folds):
    for data, source in SOURCES.items():
        for fold in folds:
            output = runs.prepared(data, fold)
            if (ROOT/output/'split_manifest.json').exists() and (ROOT/output/'test.json').exists():
                continue
            subprocess.run([sys.executable, 'scripts/prepare_data.py', '--assign', f'data/splits/cv5/fold{fold}.json', '--output', output, *source],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            print('prepared', output, flush=True)


def train_one(runs, setup, fold, device):
    folder, model = runs.folder(setup, fold), SETUPS[setup]['model']
    if (folder/'DONE').exists():
        print(setup, 'fold', fold, 'already finished', flush=True); return True
    state = folder/('last/training_state.pt' if model == 'yolos' else 'last.pt')
    resume = state.exists()
    if folder.exists() and not resume:
        shutil.rmtree(folder)  # stopped before its first checkpoint; nothing to keep
    config = runs.config(setup, fold)
    if model == 'yolos':
        command = [sys.executable, 'scripts/train_yolos.py', '--config', config, '--sampling', 'natural', '--device', device, '--output', folder]
    else:
        command = [ssd_python(), 'experiments/mobilevitv2_ssd/scripts/experiment.py', 'train', '--config', config, '--seed', '42',
                   '--device', device, '--precision', 'fp32', '--output', folder]
    command = [str(part) for part in command]+(['--resume'] if resume else [])
    print(f'=== {setup} fold {fold}: {"resuming" if resume else "starting"}', flush=True)
    logs = runs.base/'logs'; logs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    tail = []
    with open(logs/f'{setup}_fold{fold}.log', 'a', encoding='utf-8') as log:
        for line in process.stdout:
            log.write(line); log.flush(); tail = (tail+[line])[-15:]
            row = re.search(r"'stage': '(\w)', 'epoch': (\d+).*?'map': (?:np\.float64\()?([\d.eE-]+)", line)
            if row:
                print(f'{setup} fold {fold} stage {row[1]} epoch {int(row[2]):2d}  mAP {float(row[3]):.3f}  {(time.time()-started)/60:.0f} min', flush=True)
    if process.wait():
        print(''.join(tail)); print(f'FAILED: {setup} fold {fold}. Log: {logs}/{setup}_fold{fold}.log', flush=True)
        return False
    (folder/'DONE').write_text('finished')
    return True


def truth(runs, data, folds):
    images, annotations = [], []
    for fold in folds:
        part = read(ROOT/runs.prepared(data, fold)/'val.json')
        images += part['images']; annotations += part['annotations']
    return dict(images=images, annotations=annotations, categories=CATEGORIES)


def score(runs, setup, folds):
    data = SETUPS[setup]['data']
    pooled = truth(runs, data, folds)
    folders = [runs.folder(setup, fold) for fold in folds]
    names = sorted(set.intersection(*[{p.stem for p in (f/'epoch_predictions').glob('*.json')} for f in folders]))
    rows, best_predictions = [], None
    for name in names:
        by_fold = [read(f/'epoch_predictions'/f'{name}.json') for f in folders]
        predictions = [p for part in by_fold for p in part]
        metrics = coco_ap(pooled, predictions)
        rows.append(dict(epoch=name, map=metrics['map'], map50=metrics['map50']))
        if best_predictions is None or (metrics['map'] or -1) > (best['map'] or -1):
            best, best_predictions, best_by_fold = metrics, predictions, by_fold
            best_name = name
    per_fold = [coco_ap(truth(runs, data, [fold]), part)['map'] for fold, part in zip(folds, best_by_fold)]
    on_image = {}
    for p in best_predictions:
        on_image.setdefault(p['image_id'], []).append(p)
    scores = [review_box(a, on_image.get(a['image_id'], []))['best_score_iou50'] for a in pooled['annotations']]
    write(runs.base/'summary'/f'{setup}_oof.json', dict(setup=setup, epoch=best_name, folds=folds, predictions=best_predictions))
    return dict(setup=setup, epoch=best_name, map=best['map'], map50=best['map50'], per_class_ap=best['per_class_ap'], per_fold_map=per_fold,
                boxes=len(scores), correct_at_05=sum(s >= .05 for s in scores), correct_at_30=sum(s >= .3 for s in scores), epochs=rows)


def number(value):
    return 'n/a' if value is None else f'{value:.3f}'


def summary(runs, setups, folds):
    results = []
    for setup in setups:
        finished = [fold for fold in folds if (runs.folder(setup, fold)/'DONE').exists()]
        if finished != list(folds):
            print(f'{setup}: skipped, finished folds {finished} of {list(folds)}', flush=True); continue
        result = score(runs, setup, folds); results.append(result)
        classes = '  '.join(f"{k[:6]} {number(v['ap'])}/{number(v['ap50'])}" for k, v in result['per_class_ap'].items())
        print(f"{setup:14} epoch {result['epoch']}  mAP {number(result['map'])}  mAP50 {number(result['map50'])}  {classes}")
        print(f"{'':14} correct boxes: {result['correct_at_05']} at score .05, {result['correct_at_30']} at score .3, of {result['boxes']}.  mAP per fold: {[number(v) for v in result['per_fold_map']]}", flush=True)
    write(runs.base/'summary'/'summary.json', results)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', choices=['all', 'prepare', 'train', 'summary'], default='all')
    p.add_argument('--folds', type=int, nargs='+', default=list(range(FOLDS))); p.add_argument('--setups', nargs='+', choices=list(SETUPS), default=list(SETUPS))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); p.add_argument('--quick', action='store_true', help='One-view, two-epoch check in separate folders')
    args = p.parse_args(); runs = Runs(args.quick); failed = []
    if args.command in ('all', 'prepare', 'train'):
        prepare(runs, args.folds)
    if args.command in ('all', 'train'):
        # Fold by fold, so a stopped job still leaves complete comparisons.
        for fold in args.folds:
            for setup in args.setups:
                if not train_one(runs, setup, fold, args.device):
                    failed.append(f'{setup} fold {fold}')
    if args.command in ('all', 'summary'):
        summary(runs, args.setups, args.folds)
    print('FAILED RUNS: '+', '.join(failed) if failed else 'CV5 ENDED WITH NO FAILED RUNS', flush=True)
    return 1 if failed else 0


if __name__ == '__main__': raise SystemExit(main())
