import copy
import sys
from pathlib import Path
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sem_ssd.io import ROOT, EXPERIMENT, CACHE, CLASSES, read
from sem_ssd.dataset import category_map, SemDataset
from sem_ssd.preprocessing import transform, inverse
from sem_ssd.metrics import coco_ap
from sem_ssd.sampling import weights_and_report, epoch_keys


@pytest.fixture
def config():
    return read(EXPERIMENT/'configs/natural.json')


def test_noncontiguous_categories():
    cats=[dict(id=11+i*7,name=n) for i,n in enumerate(CLASSES)]
    cats.append(dict(id=99,name='no_defect'))
    assert category_map(cats)=={11:0,18:1,25:2,32:3}
    with pytest.raises(ValueError): category_map(cats+[dict(id=111,name='unknown')])


@pytest.mark.parametrize('mode',['resize','letterbox'])
def test_geometry_roundtrip_and_empty(mode):
    source=Image.new('L',(113,57),128)
    tensor,boxes,meta,_=transform(source,[[11,7,20,13],[50,20,15,10]],320,mode)
    assert tensor.shape==(3,320,320)
    assert torch.allclose(inverse(boxes,meta),torch.tensor([[11.,7,31,20],[50,20,65,30]]),atol=1e-4)
    assert transform(source,[],320,mode)[1].shape==(0,4)


def test_flips():
    aug=dict(hflip=1.,vflip=1.,brightness=0.,contrast=0.)
    _,boxes,meta,_=transform(Image.new('RGB',(100,50)),[[10,5,20,10]],320,augmentation=aug)
    assert torch.allclose(inverse(boxes,meta),torch.tensor([[70.,35,90,45]]))


def test_sampling_budget_and_resume(config):
    ds=SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=True)
    assert ds.views_per_image==21 and len(ds)==66*21
    a=torch.Generator().manual_seed(1);b=torch.Generator().manual_seed(2)
    natural=epoch_keys(ds,config,a,b)
    assert sorted(k[0] for k in natural)==list(range(len(ds)))
    config['sampling']='weighted'
    weights,report=weights_and_report(ds,config)
    assert report['image_class_counts']['gap']==5 and report['presentations_per_epoch']==1386
    assert len(weights)==1386 and weights==weights[:66]*21
    assert max(weights)==pytest.approx((15/5)**.5)
    assert report['expected_clean_fraction']<26/66
    saved_a=a.get_state();saved_b=b.get_state()
    first=epoch_keys(ds,config,a,b)
    a.set_state(saved_a);b.set_state(saved_b)
    assert epoch_keys(ds,config,a,b)==first
    assert len(first)==len(natural)==len(ds)
    a.set_state(saved_a);b.manual_seed(999)
    assert [i for i,s in epoch_keys(ds,config,a,b)]==[i for i,s in first]


def test_every_view_once_per_natural_epoch_and_validation_unexpanded(config):
    ds=SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=True)
    names=[ds.view_name(i) for i in range(len(ds))]
    assert names.count('original')==66
    assert all(names.count(r['name'])==66 for r in config['augmentation_recipes'])
    keys=epoch_keys(ds,config,torch.Generator().manual_seed(3),torch.Generator().manual_seed(4))
    assert {(i%66,i//66) for i,_ in keys}=={(s,v) for s in range(66) for v in range(21)}
    val=SemDataset(ROOT,ROOT/config['prepared'],'val',config)
    assert len(val)==16 and val.views_per_image==1
    plain=SemDataset(ROOT,ROOT/config['prepared'],'train',config)
    assert len(plain)==66


def test_recipe_views_keep_labels_and_boxes_inside(config):
    ds=SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=True)
    plain=SemDataset(ROOT,ROOT/config['prepared'],'train',config)
    n=len(ds.images)
    for source in range(n):
        reference=plain[source]
        original=ds[(source,7)]
        assert torch.equal(original['image'],reference['image']) and torch.equal(original['boxes'],reference['boxes'])
        for view in range(1,ds.views_per_image):
            item=ds[(source+view*n,7)]
            assert item['image'].shape==(3,config['size'],config['size'])
            assert torch.equal(item['labels'],reference['labels'])
            if len(item['boxes']):
                assert bool((item['boxes'][:,2:]>item['boxes'][:,:2]).all())
                assert bool((item['boxes']>=-1e-4).all()) and bool((item['boxes']<=1+1e-4).all())
    first=ds[(n*20,11)]['image'];again=ds[(n*20,11)]['image']
    assert torch.equal(first,again)  # appearance noise is reproducible from the epoch seed


def test_boxes_follow_pixels_with_this_torchvision(config):
    # Same mask check as the YOLOS suite, run inside the SSD environment (older torchvision).
    from PIL import ImageDraw
    from torchvision.transforms.v2 import functional as F
    from sem_ssd.recipes import geometry
    ds=SemDataset(ROOT,ROOT/config['prepared'],'train',config)
    for info in ds.images:
        for ann in ds.anns[info['id']]:
            x,y,w,h=ann['bbox'];snapped=[round(x),round(y),round(x+w)-round(x),round(y+h)-round(y)]
            mask=Image.new('RGB',(info['width'],info['height']),'black')
            ImageDraw.Draw(mask).rectangle((snapped[0],snapped[1],snapped[0]+snapped[2]-1,snapped[1]+snapped[3]-1),fill='white')
            for recipe in config['augmentation_recipes']:
                out,b,_=geometry(mask,[snapped],[0],recipe)
                lit=(F.to_image(out)[0]>127).nonzero()
                bx,by,bw,bh=b[0].tolist()
                found=(float(lit[:,1].min()),float(lit[:,0].min()),float(lit[:,1].max())+1,float(lit[:,0].max())+1)
                assert all(abs(g-w_)<=1.5 for g,w_ in zip(found,(bx,by,bx+bw,by+bh))),(info['file_name'],recipe['name'],found)


def test_pitch32_views_and_tiled_validation():
    config=read(EXPERIMENT/'configs/pitch32_natural.json')
    assert config['resize_mode']=='pad' and config['size']==640 and config['native_pitch']
    ds=SemDataset(ROOT,ROOT/config['prepared'],'train',config,training=True)
    n=len(ds.images);assert ds.views_per_image==21 and len(ds)==n*21
    for source,im in enumerate(ds.images):
        for view in (1,12,20):
            item=ds[(source+view*n,view)]
            assert item['image'].shape==(3,640,640) and item['meta']['scale_x']<=1
            assert len(item['labels'])>0 if ds.anns[im['id']] else len(item['labels'])==0
    val=SemDataset(ROOT,ROOT/config['prepared'],'val',config,tiled=True)
    assert len(val)>len(val.images) and val.tiled_images
    assert len(SemDataset(ROOT,ROOT/config['prepared'],'val',config))==16
    recovered={}
    for index in range(len(val)):
        item=val[index]
        recovered.setdefault(item['image_id'],[]).extend(inverse(item['boxes'],item['meta'],clamp=False).tolist())
    for im in val.images:
        for ann in val.anns[im['id']]:
            x,y,w,h=ann['bbox']
            assert min(max(abs(a-b) for a,b in zip((x,y,x+w,y+h),got)) for got in recovered[im['id']])<1.5


def metric_fixture():
    data=dict(images=[dict(id=i,width=100,height=100) for i in range(3)],
              categories=[dict(id=i,name=n) for i,n in enumerate(CLASSES)],
              annotations=[dict(id=i,image_id=i,category_id=0,bbox=[10,10,20,20],area=400,iscrowd=0) for i in range(2)])
    preds=[dict(image_id=i,category_id=0,bbox=[10,10,20,20],score=.9-i*.2) for i in range(2)]
    return data,preds


def test_coco_zero_id_perfect_wrong_class_and_false_positives():
    data,preds=metric_fixture()
    assert coco_ap(data,preds)['map']==pytest.approx(1.)
    assert data['annotations'][0]['id']==0
    wrong=[dict(p,category_id=1) for p in preds]
    assert coco_ap(data,wrong)['map']==0.
    # Duplicate ranks ahead of the second true detection, reducing interpolated AP.
    dup=preds+[dict(preds[0],score=.8)]
    assert 0<coco_ap(data,dup)['map']<1
    clean=preds+[dict(preds[0],image_id=2,score=.99)]
    assert 0<coco_ap(data,clean)['map']<1
    assert coco_ap(data,[])['map']==0


@pytest.fixture(scope='module')
def detector():
    from sem_ssd.model import create_model
    torch.set_num_threads(4)
    return create_model(read(EXPERIMENT/'configs/natural.json'))


def test_pretrained_regression_and_background_preserved(detector):
    model,loss,report=detector
    state=torch.load(CACHE/'coco-ssd-mobilevitv2-0.75.pt',map_location='cpu',weights_only=True)
    assert report['missing_keys']==report['unexpected_keys']==[]
    assert report['retained_pretrained_parameters']==2947355
    current=model.state_dict()
    for layer in report['replaced_layers']:
        for suffix in ('weight','bias'):
            key=layer['module']+'.'+suffix
            old=state[key].reshape(layer['anchors'],85,-1)
            new=current[key].reshape(layer['anchors'],9,-1)
            assert torch.equal(old[:,:5],new[:,:5])


def test_match_decode_and_empty_loss(detector):
    from sem_ssd.model import match_targets
    model,loss,_=detector
    anchors=torch.tensor([[[.5,.5,.2,.2],[.1,.1,.1,.1]]])
    target=dict(boxes=torch.tensor([[.4,.4,.6,.6]]),labels=torch.tensor([2]))
    matched=match_targets(model,[target],anchors,torch.device('cpu'))
    assert matched['box_labels'][0].tolist()==[2,0]
    recovered=model.match_prior.convert_to_boxes(matched['box_coordinates'],anchors)
    assert torch.allclose(recovered[0,0],target['boxes'][0],atol=1e-6)
    scores=torch.zeros(1,2,5,requires_grad=True)
    pred=dict(scores=scores,boxes=torch.zeros(1,2,4,requires_grad=True))
    assert torch.allclose(loss(pred,matched)['total_loss'],loss.base(None,pred,matched)['total_loss'])
    empty=match_targets(model,[dict(boxes=torch.empty(0,4),labels=torch.empty(0,dtype=torch.long))],anchors,torch.device('cpu'))
    assert torch.isfinite(empty['box_coordinates']).all()
    assert loss.base(None,pred,empty)['total_loss']==0
    result=loss(pred,empty)
    assert result['total_loss']>0
    result['total_loss'].backward()
    assert scores.grad.abs().sum()>0


def test_stage_freezing(detector):
    from sem_ssd.model import configure_stage
    model,_,_=detector
    a=configure_stage(model,'A')
    assert a['trainable_parameters']==498226
    assert not any(p.requires_grad for p in model.encoder.parameters())
    b=configure_stage(model,'B')
    assert b['trainable_parameters']==1826869
    assert all(p.requires_grad==n.startswith('layer_5.') for n,p in model.encoder.named_parameters())
    assert all(not m.training for n,m in model.encoder.named_modules() if isinstance(m,torch.nn.BatchNorm2d) and not n.startswith('layer_5.'))


def test_partial_accumulation_matches_group_mean():
    from sem_ssd.training import accumulation_divisor
    accumulated=torch.nn.Linear(1,1,bias=False)
    reference=copy.deepcopy(accumulated)
    opt=torch.optim.SGD(accumulated.parameters(),lr=.01)
    ref_opt=torch.optim.SGD(reference.parameters(),lr=.01)
    xs=torch.arange(1,12,dtype=torch.float32).reshape(-1,1)/12
    for step,x in enumerate(xs):
        accumulated(x).square().mean().div(accumulation_divisor(step,len(xs),8)).backward()
        if (step+1)%8==0 or step+1==len(xs):
            opt.step();opt.zero_grad()
    for group in xs.split(8):
        reference(group).square().mean().backward()
        ref_opt.step();ref_opt.zero_grad()
    assert torch.allclose(accumulated.weight,reference.weight,atol=1e-7)


def test_class_agnostic_recall_and_duplicate_matching():
    from sem_ssd.metrics import localization_recall
    data,predictions=metric_fixture()
    wrong=[dict(p,category_id=3) for p in predictions]
    assert localization_recall(data,wrong)['recall']==1.
    assert localization_recall(data,[wrong[0],wrong[0]])['recall']==.5
    assert localization_recall(data,[dict(p,score=.1) for p in wrong])['recall']==0.
    assert localization_recall(data,[dict(wrong[0],image_id=2)])['recall']==0.


def test_coco_metrics_can_be_saved_as_strict_json(tmp_path):
    from sem_ssd.io import write
    data,predictions=metric_fixture()
    metrics=coco_ap(data,predictions)
    assert type(metrics['per_class_ap']['bridge']['support']) is int
    path=tmp_path/'metrics_A_001.json'
    write(path,dict(stage='A',epoch=1,**metrics))
    saved=read(path)
    assert saved['per_class_ap']['bridge']['support']==2
    assert saved['map']==pytest.approx(1.)
