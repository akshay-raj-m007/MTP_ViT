"""Measure line pitch per SEM image, then rotate/resize every image to one target pitch.

Step 1 (review):   python scripts/pitch_normalize.py --review
    Writes contact sheets to outputs/pitch_review/. Cyan guide lines are drawn every measured pitch,
    phase-aligned to the white lines: if they sit on the lines across the whole image, the pitch is right.
    Fix wrong ones in data/pitch_norm/pitch_overrides.json, e.g. {"fig6.jpg": {"periods": 2}} (line
    periods across the image, counted perpendicular to the lines) or {"fig6.jpg": {"pitch": 90.5}};
    "orientation": "vertical"|"horizontal" overrides direction; {"fig6.jpg": {"confirm": true}} accepts a
    flagged estimate that is actually right. Rerun --review until all look right.
Step 2 (write):    python scripts/pitch_normalize.py --write
    Writes data/pitch_norm/: images/ (same file names), pitch_table.json, coco.json (boxes mapped
    exactly by the same rotation/scale), label_studio_tasks.json + labeling_config.xml (boxes pre-filled
    for review), and outputs/pitch_review/normalized_*.png sheets with the mapped boxes drawn.
Horizontal-line images are rotated 90 degrees clockwise so every pattern is vertical. Nothing is cropped.
"""

# Allow direct execution without installing the project package.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))

import argparse
import csv
import uuid
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sem_yolos.common import ROOT, CLASSES, UPLOAD_PREFIX, read, write, prepared_dir, resolve_image

COLORS = ["#ff3b30", "#00c7ff", "#ffcc00", "#d65cff"]  # bridge, gap, line_collapse, multi_bridge
OUT = ROOT/'data/pitch_norm'
REVIEW = ROOT/'outputs/pitch_review'
MIN_CONFIDENCE, MIN_PERIODS = .2, 3.


def short(name):
    """fig name without directories or the Label Studio upload prefix(es)."""
    name = Path(name.replace(chr(92), '/')).name
    while UPLOAD_PREFIX.match(name): name = UPLOAD_PREFIX.sub('', name)
    return name.split('-', 1)[-1] if '-' in name else name


def profile(gray, orientation):
    # Vertical lines vary along x, so average over rows; then remove the linear illumination trend.
    p = gray.mean(axis=0 if orientation == 'vertical' else 1)
    x = np.arange(len(p)); p = p-np.polyval(np.polyfit(x, p, 1), x)
    return p


def dft(p, f):
    x = np.arange(len(p))
    return (p*np.hanning(len(p))*np.exp(-2j*np.pi*f*x)).sum()


def line_spectrum(gray, orientation):
    """Power spectrum averaged over every pixel row (vertical lines) or column; tolerant of tilted lines."""
    lines = gray if orientation == 'vertical' else gray.T
    n = lines.shape[1]; pad = 16*n
    lines = lines-lines.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(lines*np.hanning(n), pad, axis=1))**2).mean(axis=0)
    freqs = np.fft.rfftfreq(pad); band = (freqs >= 1.5/n) & (freqs <= 1/3)
    return freqs[band], power[band]


def peak_share(freqs, power, f):
    return float(power[np.abs(freqs-f) <= .1*f].sum()/power.sum())  # power within +/-10% of f


def estimate(gray, orientation=None, pitch=None):
    """Return orientation, pitch (px), phase, band power share of the peak."""
    spectra = {o: line_spectrum(gray, o) for o in ('vertical', 'horizontal')}
    if orientation is None:  # lines run along the direction whose cross-line spectrum has the sharper peak
        orientation = max(spectra, key=lambda o: peak_share(*spectra[o], spectra[o][0][np.argmax(spectra[o][1])]))
    freqs, power = spectra[orientation]
    f = freqs[np.argmax(power)] if pitch is None else 1/pitch
    confidence = peak_share(freqs, power, f)
    p = profile(gray, orientation)
    # p ~ A cos(2 pi f x + phase): white-line centres are at x = (2 pi k - phase)/(2 pi f).
    return orientation, float(1/f), float(np.angle(dft(p, f))), confidence


def white_centres(length, pitch, phase):
    start = (-phase/(2*np.pi)) % 1*pitch
    return np.arange(start, length, pitch)


def load_records(prepared):
    records = []
    for split in ('train', 'val', 'test'):
        data = read(prepared/f'{split}.json')
        boxes = {}
        for a in data['annotations']: boxes.setdefault(a['image_id'], []).append(a)
        for im in data['images']:
            records.append(dict(split=split, file_name=Path(im['file_name']).name, short=short(im['file_name']),
                                source=im['file_name'], width=im['width'], height=im['height'],
                                boxes=[dict(bbox=a['bbox'], category_id=a['category_id']) for a in boxes.get(im['id'], [])]))
    return sorted(records, key=lambda r: int(''.join(c for c in r['short'] if c.isdigit()) or 0))


def measure(records, overrides, target):
    for r in records:
        with Image.open(resolve_image(ROOT, r['source'])) as im: gray = np.asarray(im.convert('L'), float)
        o = overrides.get(r['short'], {})
        auto = estimate(gray)
        orientation = o.get('orientation', auto[0])
        length = gray.shape[1] if orientation == 'vertical' else gray.shape[0]
        pitch = o.get('pitch') or (length/o['periods'] if o.get('periods') else None)
        orientation, pitch, phase, confidence = estimate(gray, orientation, pitch) if (pitch or orientation != auto[0]) else auto
        source = 'override' if set(o)-{'confirm'} else ('confirmed' if o.get('confirm') else 'auto')
        r.update(orientation=orientation, pitch=round(pitch, 3), phase=phase, confidence=round(confidence, 3),
                 periods=round(length/pitch, 2), pitch_source=source, auto_pitch=round(auto[1], 3),
                 rotate_cw90=orientation == 'horizontal', scale=round(target/pitch, 5))
        r['flag'] = (r['pitch_source'] == 'auto' and (confidence < MIN_CONFIDENCE or r['periods'] < MIN_PERIODS))
    return records


def font(size):
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()


def sheets(items, render, prefix, per_page=12, cols=4, tile=330):
    REVIEW.mkdir(parents=True, exist_ok=True)
    for old in REVIEW.glob(f'{prefix}_*.png'): old.unlink()
    fnt = font(14)
    for page in range(0, len(items), per_page):
        chunk = items[page:page+per_page]; rows = (len(chunk)+cols-1)//cols
        sheet = Image.new('RGB', (cols*tile, rows*(tile+52)), 'white')
        for i, item in enumerate(chunk):
            img, caption, colour = render(item, tile)
            x, y = (i % cols)*tile, (i//cols)*(tile+52)
            sheet.paste(img, (x+(tile-img.width)//2, y+(tile-img.height)//2))
            ImageDraw.Draw(sheet).multiline_text((x+4, y+tile+2), caption, fill=colour, font=fnt, spacing=2)
        sheet.save(REVIEW/f'{prefix}_{page//per_page+1:02d}.png')


def render_review(r, tile):
    with Image.open(resolve_image(ROOT, r['source'])) as im: img = im.convert('RGB')
    s = (tile-8)/max(img.size); img = img.resize((max(1, round(img.width*s)), max(1, round(img.height*s))), Image.LANCZOS)
    draw = ImageDraw.Draw(img, 'RGBA'); length = r['width'] if r['orientation'] == 'vertical' else r['height']
    for c in white_centres(length, r['pitch'], r['phase']):
        v = c*s
        draw.line([(v, 0), (v, img.height)] if r['orientation'] == 'vertical' else [(0, v), (img.width, v)], fill=(0, 230, 255, 150), width=1)
    flag = 'CHECK  ' if r['flag'] else ''
    caption = (f"{flag}{r['short']} [{r['split']}] {r['width']}x{r['height']}\n"
               f"{r['orientation']} pitch {r['pitch']:.1f}px ({r['periods']:.1f} periods) conf {r['confidence']:.2f} {r['pitch_source']}")
    return img, caption, '#d00000' if r['flag'] else ('#0060c0' if r['pitch_source'] == 'override' else 'black')


def transform_image(img, r):
    if r['rotate_cw90']: img = img.transpose(Image.ROTATE_270)  # 90 degrees clockwise
    size = (max(1, round(img.width*r['scale'])), max(1, round(img.height*r['scale'])))
    return img.resize(size, Image.LANCZOS)


def transform_box(bbox, r, size):
    x, y, w, h = bbox
    W, H = (r['height'], r['width']) if r['rotate_cw90'] else (r['width'], r['height'])
    if r['rotate_cw90']: x, y, w, h = r['height']-y-h, x, h, w  # (x, y) -> (H - y, x)
    sx, sy = size[0]/W, size[1]/H
    return [x*sx, y*sy, w*sx, h*sy]


def check_box(r, bbox, mapped, size):
    """Paint the box as a mask, push it through the same rotation and resize, compare extents."""
    mask = Image.new('L', (r['width'], r['height']), 0)
    x, y, w, h = [round(v) for v in bbox]
    if w < 1 or h < 1: return
    ImageDraw.Draw(mask).rectangle((x, y, x+w-1, y+h-1), fill=255)
    if r['rotate_cw90']: mask = mask.transpose(Image.ROTATE_270)
    lit = np.argwhere(np.asarray(mask.resize(size, Image.NEAREST)) > 127)
    found = (lit[:, 1].min(), lit[:, 0].min(), lit[:, 1].max()+1, lit[:, 0].max()+1)
    expected = transform_box([x, y, w, h], r, size); expected = (expected[0], expected[1], expected[0]+expected[2], expected[1]+expected[3])
    tolerance = 1.5+max(r['scale'], 1)
    if any(abs(a-b) > tolerance for a, b in zip(found, expected)):
        raise AssertionError(f"Box mapping mismatch on {r['short']}: {found} vs {expected}")


def write_outputs(records, target):
    images = OUT/'images'; images.mkdir(parents=True, exist_ok=True)
    coco = dict(images=[], annotations=[], categories=[dict(id=i, name=n) for i, n in enumerate(CLASSES)])
    tasks = []
    for i, r in enumerate(records):
        with Image.open(resolve_image(ROOT, r['source'])) as im:
            mode = im.mode; out = transform_image(im.convert('RGB') if mode not in ('L', 'RGB') else im.copy(), r)
        path = images/r['file_name']
        out.save(path, quality=100, subsampling=0) if path.suffix.lower() in ('.jpg', '.jpeg') else out.save(path)
        r.update(new_width=out.width, new_height=out.height, normalized_file=str(path.relative_to(ROOT)).replace(chr(92), '/'))
        coco['images'].append(dict(id=i, file_name=r['normalized_file'], width=out.width, height=out.height, split=r['split']))
        result = []
        for b in r['boxes']:
            mapped = transform_box(b['bbox'], r, out.size); check_box(r, b['bbox'], mapped, out.size)
            coco['annotations'].append(dict(id=len(coco['annotations']), image_id=i, category_id=b['category_id'], bbox=mapped,
                                            area=mapped[2]*mapped[3], iscrowd=0, segmentation=[]))
            result.append(dict(id=uuid.uuid4().hex[:10], type='rectanglelabels', from_name='label', to_name='image',
                               original_width=out.width, original_height=out.height, image_rotation=0,
                               value=dict(x=100*mapped[0]/out.width, y=100*mapped[1]/out.height, width=100*mapped[2]/out.width,
                                          height=100*mapped[3]/out.height, rotation=0, rectanglelabels=[CLASSES[b['category_id']]])))
        tasks.append(dict(data=dict(image='/data/local-files/?d='+r['normalized_file'], split=r['split'], source=r['file_name']),
                          annotations=[dict(result=result)]))
    table = [{k: r[k] for k in ('file_name', 'short', 'split', 'width', 'height', 'orientation', 'pitch', 'pitch_source', 'auto_pitch',
                                'confidence', 'periods', 'rotate_cw90', 'scale', 'new_width', 'new_height', 'normalized_file')} for r in records]
    write(OUT/'pitch_table.json', dict(target_pitch=target, images=table))
    write(OUT/'coco.json', coco)
    write(OUT/'label_studio_tasks.json', tasks)
    (OUT/'labeling_config.xml').write_text('<View>\n  <Image name="image" value="$image" zoom="true" zoomControl="true"/>\n'
        '  <RectangleLabels name="label" toName="image">\n' + ''.join(f'    <Label value="{c}" background="{k}"/>\n' for c, k in zip(CLASSES, COLORS))
        + '  </RectangleLabels>\n</View>\n', encoding='utf-8')
    sheets(records, render_normalized, 'normalized')


def render_normalized(r, tile):
    with Image.open(ROOT/r['normalized_file']) as im: img = im.convert('RGB')
    s = min(1., (tile-8)/max(img.size))  # never enlarge here, so true relative sizes stay visible
    img = img.resize((max(1, round(img.width*s)), max(1, round(img.height*s))), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    for b in r['boxes']:
        x, y, w, h = [v*s for v in transform_box(b['bbox'], r, (r['new_width'], r['new_height']))]
        draw.rectangle([x, y, x+w, y+h], outline=COLORS[b['category_id']], width=2)
    caption = (f"{r['short']} [{r['split']}] {r['width']}x{r['height']} -> {r['new_width']}x{r['new_height']}\n"
               f"x{r['scale']:.2f}{' rotated 90cw' if r['rotate_cw90'] else ''}  shown at {s:.2f}")
    return img, caption, 'black'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--review', action='store_true'); mode.add_argument('--write', action='store_true')
    p.add_argument('--prepared', default='data/prepared/label_studio_v2')
    p.add_argument('--target-pitch', type=float, default=32.)
    p.add_argument('--overrides', type=Path, default=OUT/'pitch_overrides.json')
    args = p.parse_args()
    overrides = read(args.overrides) if args.overrides.exists() else {}
    records = load_records(prepared_dir(ROOT, args.prepared))
    unknown = set(overrides)-{r['short'] for r in records}
    if unknown: raise ValueError(f'Overrides for unknown images: {sorted(unknown)}')
    records = measure(records, overrides, args.target_pitch)
    flagged = [r for r in records if r['flag']]
    if args.review:
        sheets(flagged+[r for r in records if not r['flag']], render_review, 'pitch')
        with open(REVIEW/'pitch_estimates.csv', 'w', newline='') as f:
            fields = ['short', 'split', 'width', 'height', 'orientation', 'pitch', 'periods', 'confidence', 'pitch_source', 'auto_pitch', 'flag', 'scale']
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(records)
        sizes = [max(r['width'], r['height'])*r['scale'] for r in records]
        print(f'{len(records)} images, {len(flagged)} flagged CHECK (listed first): {[r["short"] for r in flagged]}')
        print(f'horizontal (will rotate): {[r["short"] for r in records if r["rotate_cw90"]]}')
        print(f'normalized longest side: min {min(sizes):.0f}, median {np.median(sizes):.0f}, max {max(sizes):.0f} px')
        print(f'sheets: {REVIEW}')
    else:
        if flagged: raise ValueError(f'Unreviewed low-confidence pitches; confirm or override them first: {[r["short"] for r in flagged]}')
        write_outputs(records, args.target_pitch)
        print(f'wrote {OUT} and normalized sheets in {REVIEW}')


if __name__ == '__main__':
    main()
