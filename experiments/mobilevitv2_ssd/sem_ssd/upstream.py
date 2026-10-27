"""Pinned official source and checkpoint acquisition in the ignored experiment cache."""
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from .io import CACHE, REVISION, CHECKPOINT_URL, CHECKPOINT_SHA256, file_hash

SOURCE_URL = 'https://github.com/apple/ml-cvnets.git'
CACHE_PATCH_BEFORE = 'TMP_CACHE_LOC = "/tmp/cvnets"'
CACHE_PATCH_AFTER = 'TMP_CACHE_LOC = os.environ["SEM_SSD_CVNETS_CACHE"]'


def git(source, *args):
    return subprocess.check_output(['git', '-c', f'safe.directory={Path(source).resolve().as_posix()}',
                                   '-C', str(source), *args], text=True).strip()


def setup(skip_weights=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    source = CACHE / 'upstream'
    if not source.exists():
        subprocess.run(['git', 'clone', SOURCE_URL, str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
    if git(source, 'rev-parse', 'HEAD') != REVISION:
        raise RuntimeError(f'Upstream checkout must be {REVISION}; use a fresh cache')
    # Explicit cache-path-only adaptation to the newly downloaded source.
    # No detector, loss, or host project source is changed here.
    common = source / 'common/__init__.py'
    text = common.read_text(encoding='utf-8')
    if CACHE_PATCH_BEFORE in text:
        common.write_text(text.replace(CACHE_PATCH_BEFORE, CACHE_PATCH_AFTER), encoding='utf-8')
    elif CACHE_PATCH_AFTER not in text:
        raise RuntimeError('Unexpected upstream common.py; cache adaptation cannot be applied')
    checkpoint = CACHE / 'coco-ssd-mobilevitv2-0.75.pt'
    if not skip_weights and not checkpoint.exists():
        temporary = checkpoint.with_suffix('.download')
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response, temporary.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if file_hash(temporary) != CHECKPOINT_SHA256:
            raise RuntimeError('Downloaded checkpoint checksum mismatch; .download retained for inspection')
        temporary.replace(checkpoint)
    if checkpoint.exists() and file_hash(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint checksum mismatch')
    return dict(source=str(source), revision=REVISION, checkpoint=str(checkpoint),
                checkpoint_available=checkpoint.exists(), expected_sha256=CHECKPOINT_SHA256,
                source_adaptation='common.TMP_CACHE_LOC reads SEM_SSD_CVNETS_CACHE')


def activate(source=None):
    source = Path(source or CACHE / 'upstream').resolve()
    if git(source, 'rev-parse', 'HEAD') != REVISION:
        raise RuntimeError('Unverified CVNets source revision')
    common_text = (source / 'common/__init__.py').read_text(encoding='utf-8')
    if CACHE_PATCH_AFTER not in common_text:
        raise RuntimeError('Run the experiment setup command to redirect upstream caches first')
    os.environ['SEM_SSD_CVNETS_CACHE'] = str(CACHE / 'cvnets_cache')
    sys.path.insert(0, str(source))
    return source
