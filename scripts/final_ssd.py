"""Train the cross-validation winner on all of train+val, then score the test images once.

The model, epoch and score cutoff below were fixed from cross-validation before this ran. Safe to rerun:
training resumes, and a finished test is reported again rather than repeated.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import os
import re
import subprocess
from cv5 import SETUPS, SOURCES, ssd_python
from make_cv_folds import final_plan
from sem_yolos.common import ROOT, read, write
from sem_yolos.metrics import fixed_metrics

SETUP = SETUPS['ssd_pitch32']
FINAL_EPOCH = 31
CUTOFF = .5
ENTRY = 'experiments/mobilevitv2_ssd/scripts/experiment.py'
QUICK = dict(stage_a_epochs=1, stage_b_epochs=2, training_view_mode='random_augmented')


def prepare(base):
    prepared = base/'prepared'
    if not (prepared/'test.json').exists():
        write(base/'plan.json', final_plan(read(ROOT/'data/splits/split_v2.json')))
        relative = base.relative_to(ROOT).as_posix()
        subprocess.run([sys.executable, 'scripts/prepare_data.py', '--assign', f'{relative}/plan.json', '--output', f'{relative}/prepared', *SOURCES[SETUP['data']]],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    counts = {split: len(read(prepared/f'{split}.json')['images']) for split in ('train', 'val', 'test')}
    if counts != dict(train=82, val=0, test=13):
        raise ValueError(f'Unexpected split sizes: {counts}')
    print('prepared', counts, flush=True)


def write_config(base, quick):
    config = read(ROOT/SETUP['config'])
    config.update(prepared=(base/'prepared').relative_to(ROOT).as_posix(), fixed_schedule=True, final_stage_b_epoch=1 if quick else FINAL_EPOCH)
    if quick:
        config.update(QUICK)
    write(base/'config.json', config)
    return base/'config.json'


def train(run, config, device):
    if (run/'DONE').exists():
        print('training already finished', flush=True); return
    resume = (run/'last.pt').exists()
    command = [ssd_python(), ENTRY, 'train', '--config', str(config), '--seed', '42', '--device', device, '--precision', 'fp32', '--output', str(run)]
    print('=== final training:', 'resuming' if resume else 'starting', flush=True)
    process = subprocess.Popen(command+(['--resume'] if resume else []), cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    tail = []
    for line in process.stdout:
        tail = (tail+[line])[-15:]
        row = re.search(r"'stage': '(\w)', 'epoch': (\d+)", line)
        if row:
            print(f'final stage {row[1]} epoch {int(row[2]):2d}', flush=True)
    if process.wait():
        print(''.join(tail))
        raise SystemExit('FINAL TRAINING FAILED')
    (run/'DONE').write_text('finished')


def test(run, base, device):
    output = run/'eval_test'
    if not (output/'metrics.json').exists():
        print('=== test images', flush=True)
        subprocess.run([ssd_python(), ENTRY, 'test', '--model', str(run/'last.pt'), '--device', device, '--precision', 'fp32', '--output', str(output)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    metrics, predictions = read(output/'metrics.json'), read(output/'predictions.json')
    fixed = fixed_metrics(read(base/'prepared/test.json'), predictions, CUTOFF)
    write(output/'cutoff.json', fixed)
    return metrics, fixed, output


def report(metrics, fixed, output, base):
    print(f"TEST  mAP {metrics['map']:.3f}  mAP50 {metrics['map50']:.3f}")
    for name, value in metrics['per_class_ap'].items():
        print(f"  {name:14} AP {value['ap']:.3f}  AP50 {value['ap50']:.3f}" if value['ap'] is not None else f'  {name:14} no boxes')
    d = fixed['detection']
    print(f"At score {CUTOFF}: {d['tp']} correct of {d['tp']+d['fn']} boxes, {d['fp']} false boxes. Clean images with a false box: {fixed['fraction_clean_with_false_detection']:.2f}")
    for name, value in fixed['per_class_detection'].items():
        print(f"  {name:14} {value['tp']} of {value['support']}")
    review = base/'review'
    if not review.exists():
        subprocess.run([sys.executable, 'scripts/draw_predictions.py', '--predictions', str(output/'predictions.json'), '--split', 'test',
                        '--prepared', (base/'prepared').relative_to(ROOT).as_posix(), '--threshold', str(CUTOFF), '--output', str(review)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print('Drawings:', review.relative_to(ROOT).as_posix(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); p.add_argument('--quick', action='store_true', help='Two-epoch check in separate folders')
    args = p.parse_args()
    name = 'final_quick' if args.quick else 'final'
    base, run = ROOT/'outputs'/name, ROOT/'outputs/mobilevitv2_ssd'/name/'pitch32_seed42'
    prepare(base)
    train(run, write_config(base, args.quick), args.device)
    report(*test(run, base, args.device), base)
    print('FINAL RUN ENDED', flush=True)


if __name__ == '__main__': main()
