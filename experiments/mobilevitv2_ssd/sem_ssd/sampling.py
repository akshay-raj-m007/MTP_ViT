from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler
from .io import CLASSES


def weights_and_report(dataset, config):
    presence = [{dataset.mapping[a['category_id']] for a in dataset.anns[im['id']]} for im in dataset.images]
    counts = Counter(c for cs in presence for c in cs)
    largest = max(counts.values(), default=1)
    power, cap, background = config['sampling_power'], config['sampling_cap'], config['background_weight']
    if power < 0 or cap < 1 or background <= 0:
        raise ValueError('Invalid sampler configuration')
    weights = [min(cap, max((largest/counts[c])**power for c in cs)) if cs else background for cs in presence]
    active = weights if config['sampling'] == 'weighted' else [1.] * len(weights)
    total = sum(active)
    report = dict(sampling=config['sampling'], unique_images=len(presence), presentations_per_epoch=len(presence),
        image_class_counts={CLASSES[c]:counts[c] for c in range(4)},
        box_class_counts=dict(Counter(CLASSES[dataset.mapping[a['category_id']]] for im in dataset.images for a in dataset.anns[im['id']])),
        weight_min=min(active), weight_max=max(active), strategy='maximum capped inverse image-presence frequency raised to power',
        expected_clean_fraction=sum(w for w,cs in zip(active,presence) if not cs)/total,
        expected_class_presentations={CLASSES[c]:len(presence)*sum(w for w,cs in zip(active,presence) if c in cs)/total for c in range(4)})
    return weights, report


def epoch_keys(dataset, config, sampling_rng, augmentation_rng):
    weights, _ = weights_and_report(dataset, config)
    if config['sampling'] == 'weighted':
        indices = list(WeightedRandomSampler(weights, len(dataset), replacement=True, generator=sampling_rng))
    else:
        indices = torch.randperm(len(dataset), generator=sampling_rng).tolist()
    seeds = torch.randint(0, 2**31, (len(indices),), generator=augmentation_rng).tolist()
    return list(zip(indices, seeds))
