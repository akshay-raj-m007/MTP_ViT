import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["bridge", "gap", "line_collapse", "multi_bridge"]
CATEGORIES = [{"id": i, "name": n} for i, n in enumerate(CLASSES)]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)

def prepared_dir(root, configured='data/prepared/label_studio'):
    """Resolve current data paths and the name stored in pre-restructure checkpoints."""
    configured=str(configured).replace(chr(92),'/')
    if configured.strip('./')=='prepared_label_studio':configured='data/prepared/label_studio'
    return Path(root)/configured

def normalized_config(config):
    """Path relocation alone must not prevent resuming a saved experiment."""
    result=dict(config)
    value=str(result.get('prepared','data/prepared/label_studio')).replace(chr(92),'/')
    result['prepared']='data/prepared/label_studio' if value.strip('./')=='prepared_label_studio' else value
    return result

def raw_images_dir(root):
    current=Path(root)/'data/raw/images'
    return current if current.exists() else Path(root)/'images'

def resolve_image(root, exported):
    name = exported.replace(chr(92), "/").rsplit("/", 1)[-1]
    matches = [p for p in raw_images_dir(root).rglob(name) if p.is_file() and p.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact basename match for {name}, found {len(matches)}")
    path = matches[0].resolve()
    path.relative_to(raw_images_dir(root).resolve())
    return path
