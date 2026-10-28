"""Shared box-aware augmentation and invertible fixed-canvas preprocessing."""
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from PIL import Image, ImageDraw, ImageFont
from .common import read, CLASSES
from .multi_augment import geometry,appearance,validate_recipes


def letterbox(image, boxes, size):
    w,h=image.size
    scale=min(size/w,size/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    left,top=(size-nw)//2,(size-nh)//2
    canvas=Image.new('RGB',(size,size),(114,114,114))
    canvas.paste(image.resize((nw,nh),Image.Resampling.BILINEAR),(left,top))
    b=torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4).clone()
    b[:,[0,2]]*=nw/w; b[:,[1,3]]*=nh/h
    b[:,0]+=left; b[:,1]+=top
    return canvas,b,dict(scale_x=nw/w,scale_y=nh/h,left=left,top=top,width=w,height=h,size=size)


def inverse_boxes(xyxy,meta):
    b=torch.as_tensor(xyxy,dtype=torch.float32).reshape(-1,4).clone()
    b[:,[0,2]]=(b[:,[0,2]]-meta['left'])/meta['scale_x']
    b[:,[1,3]]=(b[:,[1,3]]-meta['top'])/meta['scale_y']
    # Predictions can lie in padding. Clamp predictions only; raw GT is never clipped.
    b[:,[0,2]]=b[:,[0,2]].clamp(0,meta['width'])
    b[:,[1,3]]=b[:,[1,3]].clamp(0,meta['height'])
    return b


def augment(image,boxes,labels,config):
    b=tv_tensors.BoundingBoxes(torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4),format='XYWH',canvas_size=(image.height,image.width))
    ops=v2.Compose([v2.RandomHorizontalFlip(config.get('hflip',.5)),v2.RandomVerticalFlip(config.get('vflip',.5)),
                    v2.ColorJitter(brightness=config.get('brightness',.1),contrast=config.get('contrast',.1))])
    image,b=ops(image,b)
    noise=config.get('noise_std',0.)
    if noise:
        t=F.to_dtype(F.to_image(image),torch.float32,scale=True)
        image=F.to_pil_image((t+torch.randn_like(t)*noise).clamp(0,1))
    if len(b)!=len(labels): raise RuntimeError('Augmentation lost labels')
    return image,b.as_subclass(torch.Tensor),labels


class CocoDetectionDataset(Dataset):
    def __init__(self,root,ann_file,processor,size=512,training=False,augmentation=None,view_mode="random_augmented",recipes=None):
        self.root=Path(root); self.coco=read(ann_file); self.processor=processor; self.size=size
        self.training=training; self.augmentation=augmentation or {}
        if view_mode not in ('random_augmented','original_plus_augmented','original_plus_recipes'):
            raise ValueError(f'Unknown training view mode: {view_mode}')
        self.view_mode=view_mode
        self.recipes=recipes if recipes is not None else []
        if view_mode=='original_plus_recipes': validate_recipes(self.recipes)
        self.views_per_image=(1+len(self.recipes) if view_mode=='original_plus_recipes' else 2 if view_mode=='original_plus_augmented' else 1) if training else 1
        expected=[{'id':i,'name':n} for i,n in enumerate(CLASSES)]
        if self.coco['categories']!=expected: raise ValueError('Use canonical prepared annotations, not old Roboflow exports')
        self.images=self.coco['images']; self.image_ids=[i['id'] for i in self.images]
        self.anns_by_image=defaultdict(list)
        for a in self.coco['annotations']: self.anns_by_image[a['image_id']].append(a)
    def __len__(self): return len(self.images)*self.views_per_image
    def view_info(self,index):
        if not 0 <= index < len(self): raise IndexError(index)
        source_index=index % len(self.images)
        augmented=self.training and (self.view_mode=='random_augmented' or index>=len(self.images))
        return source_index,augmented
    def view_name(self,index):
        _,augmented=self.view_info(index)
        if not augmented:return 'original'
        if self.view_mode=='original_plus_recipes':return self.recipes[index//len(self.images)-1]['name']
        return 'augmented'
    def transformed(self,index):
        source_index,augmented=self.view_info(index)
        info=self.images[source_index]
        with Image.open(self.root/info['file_name']) as raw: image=raw.convert('RGB')
        anns=self.anns_by_image[info['id']]
        boxes=[a['bbox'] for a in anns]; labels=[a['category_id'] for a in anns]
        recipe=None
        if augmented:
            if self.view_mode=='original_plus_recipes':
                recipe=self.recipes[index//len(self.images)-1]
                image,boxes,labels=geometry(image,boxes,labels,recipe)
            else:image,boxes,labels=augment(image,boxes,labels,self.augmentation)
        canvas,boxes,meta=letterbox(image,boxes,self.size)
        if recipe is not None:
            content=(meta['left'],meta['top'],round(meta['width']*meta['scale_x']),round(meta['height']*meta['scale_y']))
            canvas=appearance(canvas,recipe,content)
        return canvas,boxes,labels,meta
    def __getitem__(self,index):
        source_index,augmented=self.view_info(index)
        info=self.images[source_index]; canvas,boxes,labels,meta=self.transformed(index)
        target={'image_id':info['id'],'annotations':[{'bbox':b.tolist(),'category_id':c,'area':float(b[2]*b[3]),'iscrowd':0} for b,c in zip(boxes,labels)]}
        encoded=self.processor(images=canvas,annotations=target,do_resize=False,return_tensors='pt')
        return dict(pixel_values=encoded['pixel_values'][0],labels=encoded['labels'][0],meta=meta,image_id=info['id'],view_kind='augmented' if augmented else 'original',view_name=self.view_name(index))


def collate_fn(batch):
    return dict(pixel_values=torch.stack([x['pixel_values'] for x in batch]),labels=[x['labels'] for x in batch],meta=[x['meta'] for x in batch],image_ids=[x['image_id'] for x in batch])


def sample_weights(dataset,power=.5,cap=3.,background_weight=1.):
    if power<0 or cap<1 or background_weight<=0: raise ValueError('Invalid sampling parameters')
    presence=[set(a['category_id'] for a in dataset.anns_by_image[i]) for i in dataset.image_ids]
    freq=Counter(c for cs in presence for c in cs); maximum=max(freq.values(),default=1)
    weights=[min(cap,max((maximum/freq[c])**power for c in cs)) if cs else background_weight for cs in presence]
    report=dict(image_class_frequencies={CLASSES[c]:n for c,n in freq.items()},min_weight=min(weights),max_weight=max(weights),
                expected_empty_fraction=sum(w for w,cs in zip(weights,presence) if not cs)/sum(weights),
                strategy='max capped inverse image-presence frequency raised to power; explicit weight for all empty targets',
                empty_targets_are_clean=True)
    report['expected_assumed_clean_fraction']=sum(w for w,im in zip(weights,dataset.images) if im.get('annotation_status')=='assumed_clean_coco')/sum(weights)
    report['expected_user_confirmed_clean_fraction']=sum(w for w,im in zip(weights,dataset.images) if im.get('annotation_status')=='user_confirmed_clean')/sum(weights)
    report['expected_clean_fraction']=report['expected_empty_fraction']
    # Frequencies above count unique source images, never duplicated views.
    report['unique_training_images']=len(dataset.images)
    report['training_view_mode']=dataset.view_mode
    report['views_per_image']=dataset.views_per_image
    report['augmentation_recipes']=dataset.recipes
    report['presentations_per_epoch']=len(dataset)
    return weights*dataset.views_per_image,report


def preview(dataset,output,all_images=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    seen=set()
    for i,info in enumerate(dataset.images):
        keys=set(info['classes'] or [4])
        if not all_images and keys<=seen: continue
        indices=[i+view*len(dataset.images) for view in range(dataset.views_per_image)]
        tiles=[]
        for index in indices:
            canvas,boxes,labels,_=dataset.transformed(index); draw=ImageDraw.Draw(canvas)
            for b,c in zip(boxes,labels):
                x,y,w,h=b.tolist(); draw.rectangle((x,y,x+w,y+h),outline='red',width=2);draw.text((x,y),CLASSES[c],fill='red')
            _,augmented=dataset.view_info(index)
            suffix='_'+dataset.view_name(index)
            canvas.save(output/(str(info['id'])+suffix+'.png'))
            if dataset.views_per_image>2:
                import textwrap
                tile=Image.new('RGB',(250,290),'white')
                tile.paste(canvas.resize((240,240),Image.Resampling.BILINEAR),(5,5))
                ImageDraw.Draw(tile).text((5,250),'\n'.join(textwrap.wrap(dataset.view_name(index),30)),fill='black',font=ImageFont.load_default(size=13))
                tiles.append(tile)
        if tiles:
            import math
            columns=5;sheet=Image.new('RGB',(columns*250,math.ceil(len(tiles)/columns)*290),'white')
            for slot,tile in enumerate(tiles):sheet.paste(tile,((slot%columns)*250,(slot//columns)*290))
            sheet.save(output/(str(info['id'])+'_all_views.png'))
        seen.update(keys)
        if not all_images and len(seen)==5: break
