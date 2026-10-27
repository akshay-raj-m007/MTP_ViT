"""Paths, source pins, and fingerprints; no imports from the YOLOS pipeline."""
import hashlib
import json
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
CACHE = ROOT / 'outputs/mobilevitv2_ssd'
CLASSES = ['bridge', 'gap', 'line_collapse', 'multi_bridge']
REVISION = '9b4d544e46011407dc12cde3735422b5eb8dc265'
CHECKPOINT_URL = 'https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.pt'
CHECKPOINT_SHA256 = '5538a12e1350bf06f05f29917c48caa8c0a1dab3a458727db06764b0f77df055'
YAML_SHA256 = '202034974608062959017a5cb3501ea8ce6604b282a228237ee397108095cdf7'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def isolated_environment():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Activate a separate experiment virtual environment first.')
    if Path(sys.prefix).resolve() in {(ROOT / 'venv').resolve(), (ROOT / '.venv').resolve()}:
        raise RuntimeError('Use the separate SSD environment, not the existing YOLOS environment.')


def output_path(path):
    path = (ROOT / path).resolve()
    if not path.is_relative_to(CACHE.resolve()):
        raise ValueError('Experiment outputs must be inside outputs/mobilevitv2_ssd/')
    return path
