"""The 20 named training views, shared with YOLOS so both detectors see identical augmentations.

This is the one deliberate import from the YOLOS tree: sem_yolos/multi_augment.py is loaded by path
(it depends only on torch/torchvision), so no other YOLOS module or package code is imported.
"""
import importlib.util
from .io import ROOT

_spec = importlib.util.spec_from_file_location('sem_ssd_shared_multi_augment', ROOT / 'sem_yolos/multi_augment.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

geometry = _module.geometry
appearance = _module.appearance
validate_recipes = _module.validate_recipes
