
# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from transformers import YolosImageProcessor
from sem_yolos.common import ROOT,read,prepared_dir
from sem_yolos.data import CocoDetectionDataset,preview
from sem_yolos.runtime import seed_all

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pipeline_config.json');p.add_argument('--deterministic',action='store_true')
    p.add_argument('--all-images',action='store_true',help='Export previews for every training image and every configured view')
    p.add_argument('--allow-download',action='store_true',help='Download the image processor on a new machine if it is not cached')
    a=p.parse_args();c=read(a.config);seed_all(c['seed'])
    processor=YolosImageProcessor.from_pretrained(c['model'],local_files_only=not a.allow_download)
    processor.do_resize=False;processor.do_pad=False
    ds=CocoDetectionDataset(a.root,prepared_dir(a.root,c['prepared'])/'train.json',processor,c['size'],not a.deterministic,c['augmentation'],view_mode=c.get('training_view_mode','random_augmented'),recipes=c.get('augmentation_recipes'))
    preview(ds,a.output,all_images=a.all_images)
