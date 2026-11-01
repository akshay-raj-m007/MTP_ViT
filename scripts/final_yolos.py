"""Train the best YOLOS setup from cross-validation on all of train+val, then score the test images once.

The setup, epoch and score cutoff below were fixed from cross-validation before this ran. Safe to rerun:
training resumes, and a finished test is reported again rather than repeated.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import subprocess
from cv5 import SETUPS, YOLOS_SCHEDULE
from final_ssd import CUTOFF, QUICK, prepare, report, train
from sem_yolos.common import ROOT, prepared_fingerprint, read, write

SETUP = SETUPS['yolos_pitch32_weighted']
FINAL_EPOCH = 13


def write_config(base, quick):
    config = read(ROOT/SETUP['config'])
    config.update(YOLOS_SCHEDULE)
    config.update(prepared=(base/'prepared').relative_to(ROOT).as_posix(), fixed_schedule=True, final_stage_b_epoch=1 if quick else FINAL_EPOCH)
    if quick:
        config.update(QUICK)
    write(base/'config.json', config)
    return base/'config.json'


def test(run, base, device):
    output, model = run/'eval_test', run/'last'
    if not (output/'metrics.json').exists():
        print('=== test images', flush=True)
        fingerprint = prepared_fingerprint(base/'prepared')
        write(model/'data_identity.json', dict(fingerprint=fingerprint))
        write(model/'threshold.json', dict(threshold=CUTOFF, objective='fixed from cross-validation', data_fingerprint=fingerprint))
        subprocess.run([sys.executable, 'scripts/evaluate.py', '--model', str(model), '--split', 'test', '--device', device, '--output', str(output)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    metrics = read(output/'metrics.json')
    return metrics, metrics, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); p.add_argument('--quick', action='store_true', help='Two-epoch check in separate folders')
    args = p.parse_args()
    base = ROOT/'outputs'/('final_yolos_quick' if args.quick else 'final_yolos')
    run = base/'run'
    prepare(base)
    config = write_config(base, args.quick)
    train(run, [sys.executable, 'scripts/train_yolos.py', '--config', str(config), '--sampling', SETUP['sampling'], '--device', args.device, '--output', str(run)], run/'last/training_state.pt')
    report(*test(run, base, args.device), base)
    print('FINAL RUN ENDED', flush=True)


if __name__ == '__main__': main()
