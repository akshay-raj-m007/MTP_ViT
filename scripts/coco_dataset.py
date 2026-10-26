"""Compatibility imports for the canonical prepared-data implementation."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

from sem_yolos.data import CocoDetectionDataset, collate_fn
