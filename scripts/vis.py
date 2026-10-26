"""Attention visualization using the new run configuration and original query IDs."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

from scripts.infer_yolos import main

if __name__ == "__main__":
    main(attention=True)
