"""Final fit on all of train+val, then one test run, for every cross-validated setup without a final yet.

The stage B epoch is the one with the best pooled cross-validation mAP in outputs/cv5/summary/summary.json, and
the score cutoff is the same .5 used by the SSD and YOLOS finals. Both are fixed before any test image is seen,
and nothing is chosen from the test result. Safe to rerun: training resumes, and a finished test is reported
again rather than repeated. Runs one setup at a time and keeps going if one fails.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import subprocess
from cv5 import SETUPS, SOURCES, YOLOS_SCHEDULE, ssd_python
from final_ssd import CUTOFF, ENTRY, QUICK, report, train
from make_cv_folds import final_plan
from sem_yolos.common import ROOT, prepared_fingerprint, read, write
from sem_yolos.metrics import coco_ap, evaluate_predictions, fixed_metrics

# These two already had their single test run, with epochs chosen earlier (final_ssd.py, final_yolos.py).
# Running them again under this rule would be a second look at the test images.
HAVE_FINAL = dict(ssd_pitch32='outputs/final', yolos_pitch32_weighted='outputs/final_yolos')
DEFAULT = [name for name in SETUPS if name not in HAVE_FINAL]
SUMMARY = ROOT/'outputs/cv5/summary/summary.json'


def cv_epoch(setup):
    rows = {row['setup']: row for row in read(SUMMARY)} if SUMMARY.exists() else {}
    if setup not in rows:
        raise SystemExit(f'{setup} is not in {SUMMARY}. Rebuild it for every setup first: python scripts/cv5.py summary')
    stage, epoch = rows[setup]['epoch'].split('_')
    if stage != 'B':
        raise SystemExit(f'{setup}: best CV epoch {rows[setup]["epoch"]} is in stage A; a final fit needs a stage B epoch')
    return int(epoch), rows[setup]


def folders(setup, quick):
    name = 'final_runs_quick' if quick else 'final_runs'
    base = ROOT/'outputs'/name/setup
    run = ROOT/'outputs/mobilevitv2_ssd'/name/setup if SETUPS[setup]['model'] == 'ssd' else base/'run'
    return base, run


def prepare(base, data):
    prepared = base/'prepared'
    if not (prepared/'test.json').exists():
        write(base/'plan.json', final_plan(read(ROOT/'data/splits/split_v2.json')))
        relative = base.relative_to(ROOT).as_posix()
        subprocess.run([sys.executable, 'scripts/prepare_data.py', '--assign', f'{relative}/plan.json', '--output', f'{relative}/prepared', *SOURCES[data]],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    counts = {split: len(read(prepared/f'{split}.json')['images']) for split in ('train', 'val', 'test')}
    if counts != dict(train=82, val=0, test=13):
        raise ValueError(f'Unexpected split sizes: {counts}')


def write_config(base, setup, epoch, quick):
    config = read(ROOT/SETUPS[setup]['config'])
    if SETUPS[setup]['model'] == 'yolos':
        config.update(YOLOS_SCHEDULE)  # the same schedule cross-validation ran
    config.update(prepared=(base/'prepared').relative_to(ROOT).as_posix(), fixed_schedule=True, final_stage_b_epoch=1 if quick else epoch)
    if quick:
        config.update(QUICK)
    write(base/'config.json', config)
    return base/'config.json'


def train_command(setup, config, run, device):
    model, sampling = SETUPS[setup]['model'], SETUPS[setup].get('sampling', 'natural')
    if model == 'ssd':
        return [ssd_python(), ENTRY, 'train', '--config', str(config), '--seed', '42', '--device', device, '--precision', 'fp32', '--output', str(run)], run/'last.pt'
    script = 'scripts/train_yolos.py' if model == 'yolos' else 'scripts/train_retinanet.py'
    return [sys.executable, script, '--config', str(config), '--sampling', sampling, '--device', device, '--output', str(run)], run/'last/training_state.pt'


def test_ssd(run, base, device):
    output = run/'eval_test'
    if not (output/'metrics.json').exists():
        subprocess.run([ssd_python(), ENTRY, 'test', '--model', str(run/'last.pt'), '--device', device, '--precision', 'fp32', '--output', str(output)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    fixed = fixed_metrics(read(base/'prepared/test.json'), read(output/'predictions.json'), CUTOFF)
    write(output/'cutoff.json', fixed)
    return read(output/'metrics.json'), fixed, output


def test_yolos(run, base, device):
    output, model = run/'eval_test', run/'last'
    if not (output/'metrics.json').exists():
        fingerprint = prepared_fingerprint(base/'prepared')
        write(model/'data_identity.json', dict(fingerprint=fingerprint))
        write(model/'threshold.json', dict(threshold=CUTOFF, objective='fixed from cross-validation', data_fingerprint=fingerprint))
        subprocess.run([sys.executable, 'scripts/evaluate.py', '--model', str(model), '--split', 'test', '--device', device, '--output', str(output)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    metrics = read(output/'metrics.json')
    return metrics, metrics, output


def test_retinanet(run, base, device):
    output = run/'eval_test'
    if not (output/'metrics.json').exists():
        import torch
        from torch.utils.data import DataLoader
        from sem_yolos.data import CocoDetectionDataset, collate_fn
        from train_retinanet import TensorTargets, build_model, predict_loader
        prepared = base/'prepared'
        if read(run/'data_identity.json')['fingerprint'] != prepared_fingerprint(prepared):
            raise ValueError('Model and prepared dataset identities differ')
        cfg = read(run/'run_config.json')['config']
        dataset = CocoDetectionDataset(ROOT, prepared/'test.json', TensorTargets(), cfg['size'], native=cfg.get('native_pitch', False), tile_overlap=cfg.get('tile_overlap'))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        torch_device = torch.device(device)
        model = build_model(cfg).to(torch_device)
        model.load_state_dict(torch.load(run/'last'/'model.pt', map_location=torch_device))
        predictions = predict_loader(model, loader, torch_device)
        output.mkdir(parents=True)
        write(output/'predictions.json', predictions)
        write(output/'metrics.json', evaluate_predictions(dataset.coco, predictions, CUTOFF))
    metrics = read(output/'metrics.json')
    return metrics, metrics, output


TESTS = dict(ssd=test_ssd, yolos=test_yolos, retinanet=test_retinanet)


def run_setup(setup, device, quick):
    epoch, cv = cv_epoch(setup)
    base, run = folders(setup, quick)
    print(f'===== {setup}: stage B epoch {epoch} from CV (CV mAP {cv["map"]:.3f}), cutoff {CUTOFF}', flush=True)
    write(base/'selection.json', dict(setup=setup, stage_b_epoch=epoch, cutoff=CUTOFF, cv_map=cv['map'], cv_map50=cv['map50'],
                                      rule='best pooled CV mAP epoch, fixed before the test run'))
    prepare(base, SETUPS[setup]['data'])
    config = write_config(base, setup, epoch, quick)
    train(run, *train_command(setup, config, run, device))
    metrics, fixed, output = TESTS[SETUPS[setup]['model']](run, base, device)
    report(metrics, fixed, output, base)
    d = fixed['detection']
    return dict(setup=setup, stage_b_epoch=epoch, cv_map=cv['map'], cv_map50=cv['map50'], test_map=metrics['map'], test_map50=metrics['map50'],
                correct=d['tp'], boxes=d['tp']+d['fn'], false_boxes=d['fp'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setups', nargs='+', choices=DEFAULT, default=DEFAULT)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); p.add_argument('--quick', action='store_true', help='Two-epoch check in separate folders')
    args = p.parse_args(); results, failed = [], []
    for setup in args.setups:
        try:
            results.append(run_setup(setup, args.device, args.quick))
        except (Exception, SystemExit) as error:
            print(f'FAILED: {setup}: {error}', flush=True); failed.append(setup)
    summary = ROOT/'outputs'/('final_runs_quick' if args.quick else 'final_runs')/'summary.json'
    old = {row['setup']: row for row in read(summary)} if summary.exists() else {}
    old.update({row['setup']: row for row in results}); write(summary, list(old.values()))
    print('\nsetup                       epoch  CV mAP  CV mAP50  test mAP  test mAP50  correct  false')
    for row in old.values():
        print(f"{row['setup']:27} B_{row['stage_b_epoch']:03d}  {row['cv_map']:.3f}   {row['cv_map50']:.3f}     {row['test_map']:.3f}     {row['test_map50']:.3f}"
              f"      {row['correct']}/{row['boxes']}     {row['false_boxes']}")
    print('FAILED SETUPS: '+', '.join(failed) if failed else 'FINAL TESTS ENDED WITH NO FAILURES', flush=True)
    return 1 if failed else 0


if __name__ == '__main__': raise SystemExit(main())
