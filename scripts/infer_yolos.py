"""Inference with saved preprocessing and validation-selected confidence threshold."""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from sem_yolos.common import read,write,CLASSES
from sem_yolos.data import letterbox
from sem_yolos.runtime import device_for,load_model,decode


def main(attention=False):
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args();cfg=read(args.model/'pipeline_config.json');threshold=read(args.model/'threshold.json')['threshold']
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    device=device_for(args.device);model,processor=load_model(args.model,device);model.eval()
    with Image.open(args.image) as raw: image=raw.convert('RGB')
    canvas,_,meta=letterbox(image,[],cfg['size']);encoding=processor(images=canvas,do_resize=False,return_tensors='pt')
    with torch.no_grad(): outputs=model(pixel_values=encoding['pixel_values'].to(device),output_attentions=attention)
    all_predictions=decode(outputs,[meta],[0]);accepted=[x for x in all_predictions if x['score']>=threshold]
    image_class=CLASSES[max(accepted,key=lambda x:x['score'])['category_id']] if accepted else 'no_defect'
    write(args.output/'predictions.json',dict(image=str(args.image.resolve()),threshold=threshold,image_class=image_class,
          predicted_class_presence=sorted({CLASSES[a['category_id']] for a in accepted}),detections=accepted,
          interpretation='No accepted detections is a threshold-based outcome, not certainty that an image is clean.'))
    draw=ImageDraw.Draw(image)
    for a in accepted:
        x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=2)
        draw.text((x,max(0,y-12)),f"{CLASSES[a['category_id']]} {a['score']:.3f}",fill='red')
    image.save(args.output/'annotated.png')
    if attention:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Select the original query, never an index into filtered predictions.
        query=int(outputs.logits[0].softmax(-1)[:,:-1].max(-1).values.argmax())
        att=outputs.attentions[-1][0].mean(0).cpu()
        n_det=model.config.num_detection_tokens;patch=model.config.patch_size
        gh=gw=cfg['size']//patch;end=att.shape[0]-n_det;start=end-gh*gw
        heat=att[end+query,start:end].reshape(gh,gw).numpy()
        fig,ax=plt.subplots();ax.imshow(canvas);ax.imshow(heat,extent=(0,cfg['size'],cfg['size'],0),alpha=.5,cmap='inferno')
        ax.set_title(f'Last-layer attention: original query {query}');ax.axis('off');fig.savefig(args.output/'attention.png',dpi=160);plt.close(fig)
    print(f'Saved {len(accepted)} accepted detections to {args.output}. {image_class} is threshold-derived.')

if __name__=='__main__': main()
