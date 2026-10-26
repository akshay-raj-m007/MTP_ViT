"""Explicit augmentation recipes with full-canvas geometry and no dropped boxes."""
import math
import re
import torch
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode,v2
from torchvision.transforms.v2 import functional as F


def validate_recipes(recipes):
    if not isinstance(recipes,list) or not 1<=len(recipes)<=63:
        raise ValueError('Provide 1..63 named augmentation recipes (19 gives 20 total views).')
    names=set();allowed={'name','hflip','vflip','scale_x','scale_y','angle','shear_x','shear_y','zoom','noise_std','brightness','contrast'}
    for recipe in recipes:
        if not isinstance(recipe,dict) or set(recipe)-allowed:raise ValueError(f'Unknown recipe fields: {recipe}')
        name=recipe.get('name','')
        if not isinstance(name,str) or not re.fullmatch('[a-z0-9_]+',name) or name in names:
            raise ValueError('Recipe names must be unique lowercase letters/digits/underscores')
        names.add(name)
        for field in ('hflip','vflip'):
            if field in recipe and not isinstance(recipe[field],bool):raise ValueError(f'{field} must be boolean')
        for field,default,lo,hi in [('scale_x',1.,.5,1.5),('scale_y',1.,.5,1.5),('zoom',1.,.5,1.5),
                                   ('angle',0.,-180.,180.),('shear_x',0.,-20.,20.),('shear_y',0.,-20.,20.),('noise_std',0.,0.,.1),('brightness',0.,0.,.3),('contrast',0.,0.,.3)]:
            value=recipe.get(field,default)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError(f'{field} must be finite and between {lo} and {hi}')


def full_canvas_affine(image,boxes,angle,shear_x=0.,shear_y=0.):
    """Use the SAME padded canvas/affine for pixels and boxes; crop padding only.

    torchvision's separate rotate(expand=True) kernels disagree for some shapes,
    and its empty-box kernel retains the old canvas. Explicit shared padding
    avoids both cases. Positive recipe rotation is counter-clockwise.
    """
    h,w=image.shape[-2:]
    # Bound the norm of two shear maps; rotation preserves the radius.
    factor=(1+abs(math.tan(math.radians(shear_x))))*(1+abs(math.tan(math.radians(shear_y))))
    side=math.ceil(math.hypot(w,h)*factor)+4
    left,top=(side-w)//2,(side-h)//2
    padding=[left,top,side-w-left,side-h-top]
    image=F.pad(image,padding,fill=114);boxes=F.pad(boxes,padding)
    frame=tv_tensors.BoundingBoxes([[float(left),float(top),float(w),float(h)]],
                                  format='XYWH',canvas_size=(side,side),clamping_mode=None)
    options=dict(angle=-angle,translate=[0,0],scale=1.,shear=[shear_x,shear_y])
    image=F.affine(image,**options,interpolation=InterpolationMode.BILINEAR,fill=114)
    boxes=F.affine(boxes,**options);frame=F.affine(frame,**options)
    x,y,bw,bh=frame[0].tolist()
    def snap(value):
        nearest=round(value)
        return nearest if abs(value-nearest)<1e-4 else value
    x0,y0=math.floor(snap(x)),math.floor(snap(y))
    x1,y1=math.ceil(snap(x+bw)),math.ceil(snap(y+bh))
    if min(x0,y0)<0 or max(x1,y1)>side:raise RuntimeError('Affine exceeded full-content padding')
    image=F.crop(image,y0,x0,y1-y0,x1-x0)
    boxes=F.crop(boxes,y0,x0,y1-y0,x1-x0)
    return image,boxes


def geometry(image,boxes,labels,recipe):
    image=F.to_image(image)
    b=tv_tensors.BoundingBoxes(torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4),format='XYWH',
                              canvas_size=tuple(image.shape[-2:]),clamping_mode=None)
    for key,fn in [('hflip',F.horizontal_flip),('vflip',F.vertical_flip)]:
        if recipe.get(key,False):image,b=fn(image),fn(b)
    h,w=image.shape[-2:]
    target=[max(1,round(h*recipe.get('scale_y',1.))),max(1,round(w*recipe.get('scale_x',1.)))]
    if target!=[h,w]:
        image=F.resize(image,target,interpolation=InterpolationMode.BILINEAR,antialias=True)
        b=F.resize(b,target)
    if 'zoom' in recipe:
        # A shared 1.25x reference canvas makes scale differences survive letterboxing.
        # Enlarging the canvas as necessary preserves all content; no zoom crop.
        h,w=image.shape[-2:];zoom=recipe['zoom']
        nh,nw=max(1,round(h*zoom)),max(1,round(w*zoom))
        canvas_h,canvas_w=max(math.ceil(h*1.25),nh),max(math.ceil(w*1.25),nw)
        image=F.resize(image,[nh,nw],interpolation=InterpolationMode.BILINEAR,antialias=True);b=F.resize(b,[nh,nw])
        left,top=(canvas_w-nw)//2,(canvas_h-nh)//2
        padding=[left,top,canvas_w-nw-left,canvas_h-nh-top]
        image=F.pad(image,padding,fill=114);b=F.pad(b,padding)
    angle=recipe.get('angle',0.)
    sx,sy=recipe.get('shear_x',0.),recipe.get('shear_y',0.)
    if angle or sx or sy:
        image,b=full_canvas_affine(image,b,angle,sx,sy)
    if tuple(image.shape[-2:])!=tuple(b.canvas_size):raise RuntimeError('Image and transformed box canvases disagree')
    result=b.as_subclass(torch.Tensor)
    if len(result)!=len(labels):raise RuntimeError('Geometry dropped a label')
    if result.numel():
        h,w=image.shape[-2:]
        if not torch.isfinite(result).all() or (result[:,2:]<=0).any():raise RuntimeError('Invalid transformed box')
        if (result[:,:2]<-1e-3).any() or (result[:,0]+result[:,2]>w+1e-3).any() or (result[:,1]+result[:,3]>h+1e-3).any():
            raise RuntimeError('Transformed box outside expanded canvas; no clipping or dropping allowed')
    return F.to_pil_image(image),result,labels


def appearance(canvas,recipe):
    # Noise sigma is defined on the final canvas in normalized [0,1] intensity units.
    brightness,contrast=recipe.get('brightness',0.),recipe.get('contrast',0.)
    if brightness or contrast:canvas=v2.ColorJitter(brightness=brightness,contrast=contrast)(canvas)
    sigma=recipe.get('noise_std',0.)
    if sigma:
        value=F.to_dtype(F.to_image(canvas),torch.float32,scale=True)
        noise=torch.randn((1,*value.shape[-2:]),dtype=value.dtype,device=value.device)
        value=(value+noise*sigma).clamp(0,1)
        canvas=F.to_pil_image(value)
    return canvas
