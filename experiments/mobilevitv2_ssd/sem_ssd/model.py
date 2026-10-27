"""Strict COCO detector load, class-channel surgery, official anchors/loss/decode."""
import argparse
import copy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from .io import EXPERIMENT, CACHE, CHECKPOINT_SHA256, YAML_SHA256, REVISION, file_hash
from .upstream import activate


def flatten(data, prefix=''):
    result = {}
    for key, value in data.items():
        name = f'{prefix}.{key}' if prefix else key
        result.update(flatten(value, name) if isinstance(value, dict) else {name:value})
    return result


def build_options(config):
    if file_hash(EXPERIMENT / 'configs/upstream_coco.yaml') != YAML_SHA256:
        raise ValueError('Official detector YAML checksum mismatch')
    activate()
    from cvnets.models.base_model import BaseAnyNNModel
    from cvnets.models.classification.base_image_encoder import BaseImageEncoder
    from cvnets.models.classification.mobilevit_v2 import MobileViTv2
    from cvnets.models.detection.base_detection import BaseDetection
    from cvnets.models.detection.ssd import SingleShotMaskDetector
    from cvnets.anchor_generator.ssd_anchor_generator import SSDAnchorGenerator
    from cvnets.matcher_det.ssd_matcher import SSDMatcher
    from cvnets.layers import arguments_nn_layers
    from loss_fn.detection.ssd_multibox_loss import SSDLoss
    parser = argparse.ArgumentParser(add_help=False)
    for cls in (BaseAnyNNModel, BaseImageEncoder, MobileViTv2, BaseDetection,
                SingleShotMaskDetector, SSDAnchorGenerator, SSDMatcher, SSDLoss):
        parser = cls.add_arguments(parser)
    parser = arguments_nn_layers(parser)
    opts = parser.parse_args([])
    values = flatten(yaml.safe_load((EXPERIMENT / 'configs/upstream_coco.yaml').read_text()))
    values.update({'model.detection.n_classes':81, 'model.normalization.name':'batch_norm',
                   'ddp.rank':0, 'ddp.use_distributed':False, 'dev.device':torch.device('cpu'),
                   'model.detection.ssd.conf_threshold':config['score_floor'],
                   'model.detection.ssd.nms_iou_threshold':config['nms_iou'],
                   'model.detection.ssd.top_k':config['pre_nms_topk'],
                   'model.detection.ssd.objects_per_image':config['max_detections']})
    for key, value in values.items():
        setattr(opts, key, value)
    return opts


def replace_class_channels(model):
    """Each anchor emits [4 regression, background, 80 COCO classes]. Retain first five."""
    report = []
    for head_index, head in enumerate(model.ssd_heads):
        matches = [(name, layer) for name, layer in head.loc_cls_layer.named_modules()
                   if isinstance(layer, nn.Conv2d) and layer.out_channels == head.n_anchors * 85]
        if len(matches) != 1:
            raise RuntimeError(f'Unexpected SSD head layout: {matches}')
        name, old = matches[0]
        new = nn.Conv2d(old.in_channels, head.n_anchors * 9, old.kernel_size,
                        stride=old.stride, padding=old.padding, dilation=old.dilation,
                        groups=old.groups, bias=old.bias is not None, padding_mode=old.padding_mode)
        nn.init.xavier_uniform_(new.weight)
        if new.bias is not None:
            nn.init.zeros_(new.bias)
        with torch.no_grad():
            new.weight.view(head.n_anchors,9,*new.weight.shape[1:])[:,:5].copy_(
                old.weight.view(head.n_anchors,85,*old.weight.shape[1:])[:,:5])
            if old.bias is not None:
                new.bias.view(head.n_anchors,9)[:,:5].copy_(old.bias.view(head.n_anchors,85)[:,:5])
        parent_name, _, leaf = name.rpartition('.')
        parent = head.loc_cls_layer.get_submodule(parent_name) if parent_name else head.loc_cls_layer
        setattr(parent, leaf, new)
        head.n_classes = 5
        report.append(dict(module=f'ssd_heads.{head_index}.loc_cls_layer.{name}',
                           anchors=head.n_anchors, old_shape=list(old.weight.shape), new_shape=list(new.weight.shape),
                           preserved='regression offsets and background channel for each anchor',
                           initialized_parameters=head.n_anchors*4*(new.weight[0].numel()+(new.bias is not None))))
    model.n_detection_classes = 5
    return report


def create_model(config, checkpoint=None, pretrained=True):
    opts = build_options(config)
    from cvnets.models.classification.mobilevit_v2 import MobileViTv2
    from cvnets.models.detection.ssd import SingleShotMaskDetector
    encoder = MobileViTv2.build_model(opts)
    model = SingleShotMaskDetector(opts, encoder)
    full_count = sum(p.numel() for p in model.parameters())
    report = dict(upstream_revision=REVISION, full_coco_parameters=full_count,
                  missing_keys=[], unexpected_keys=[], loaded_pretrained_parameters=0)
    if pretrained:
        checkpoint = Path(checkpoint or CACHE / 'coco-ssd-mobilevitv2-0.75.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Complete COCO detector checkpoint required: {checkpoint}. Run setup or pass --checkpoint.')
        checksum = file_hash(checkpoint)
        if checksum != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint is not the verified official SSD-MobileViTv2-0.75 checkpoint')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        report.update(checkpoint_sha256=checksum, loaded_pretrained_parameters=full_count)
    report['replaced_layers'] = replace_class_channels(model)
    total = sum(p.numel() for p in model.parameters())
    initialized = sum(v['initialized_parameters'] for v in report['replaced_layers'])
    report.update(total_parameters=total, initialized_parameters=initialized,
                  retained_pretrained_parameters=total-initialized if pretrained else 0,
                  internal_labels={'background':0,'bridge':1,'gap':2,'line_collapse':3,'multi_bridge':4})
    return model, EmptyAwareLoss(opts, config['empty_negatives']), report


class EmptyAwareLoss(nn.Module):
    """Upstream multibox loss plus explicit hard background negatives on empty images."""
    def __init__(self, opts, empty_negatives=3):
        super().__init__()
        from loss_fn.detection.ssd_multibox_loss import SSDLoss
        self.base = SSDLoss(opts)
        self.empty_negatives = empty_negatives

    def forward(self, prediction, target):
        # FP32 loss and matching even when the network forward uses AMP.
        prediction = {k:v.float() for k,v in prediction.items()}
        result = self.base(None, prediction, target)
        empty = ~(target['box_labels'] > 0).any(dim=1)
        extra = prediction['scores'].sum() * 0
        if empty.any():
            losses = -F.log_softmax(prediction['scores'][empty], dim=-1)[:,:,0]
            k = min(self.empty_negatives, losses.shape[1])
            denominator = max(1, int((target['box_labels'] > 0).sum()))
            extra = losses.topk(k, dim=1).values.sum() / denominator
        return dict(total_loss=result['total_loss']+extra, reg_loss=result['reg_loss'],
                    cls_loss=result['cls_loss']+extra, empty_background_loss=extra)


def raw_forward(model, images):
    features = model.get_backbone_features(images)
    scores, boxes, anchors = model.ssd_forward(features, device=images.device)
    if scores.shape[1] != anchors.shape[1] or boxes.shape[1] != anchors.shape[1]:
        raise RuntimeError('Feature-map anchors disagree with head output')
    return dict(scores=scores, boxes=boxes), anchors.float()


def match_targets(model, targets, anchors, device):
    coordinates, labels = [], []
    for target in targets:
        gt_boxes, gt_labels = target['boxes'].to(device), target['labels'].to(device)
        if not len(gt_labels):
            # Upstream encodes zero-size dummy boxes as log(0); no dummy regression targets needed.
            matched_boxes = torch.zeros_like(anchors[0])
            matched_labels = torch.zeros(len(anchors[0]), dtype=torch.long, device=device)
        else:
            matched_boxes, matched_labels = model.match_prior(gt_boxes, gt_labels, anchors[0])
        coordinates.append(matched_boxes); labels.append(matched_labels)
    return dict(box_coordinates=torch.stack(coordinates), box_labels=torch.stack(labels))


def configure_stage(model, stage):
    if stage not in ('A','B'):
        raise ValueError(stage)
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith('encoder.') or (stage=='B' and name.startswith('encoder.layer_5.'))
    if stage == 'B' and not hasattr(model.encoder, 'layer_5'):
        raise RuntimeError('Expected official MobileViTv2 encoder.layer_5')
    set_training_mode(model, stage)
    return dict(stage=stage, trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def set_training_mode(model, stage):
    model.train()
    model.encoder.eval()  # Includes frozen backbone BatchNorm running statistics.
    if stage == 'B':
        model.encoder.layer_5.train()


def decode(model, prediction, anchors):
    boxes = model.match_prior.convert_to_boxes(prediction['boxes'].float(), anchors.float())
    scores = prediction['scores'].float().softmax(-1)
    return model.postprocess_detections(boxes, scores)
