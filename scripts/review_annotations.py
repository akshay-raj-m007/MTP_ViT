"""Generate a read-only, local manual review bundle from prepared COCO ground truth."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageOps

from sem_yolos.common import CLASSES, ROOT, read, resolve_image, prepared_fingerprint


COLORS = {
    "bridge": "#E63946",
    "gap": "#F4A261",
    "line_collapse": "#16A6A0",
    "multi_bridge": "#A46DE0",
}
EMPTY_MESSAGE = "NO ANNOTATED BOXES \u2014 CLEAN STATUS REQUIRES MANUAL REVIEW"
SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "split", "image_id", "filename", "image_width", "image_height",
    "annotation_count", "annotation_ids", "current_box_classes",
    "existing_annotation_status", "structural_warnings", "review_status",
    "clean_review_status", "issue_types", "affected_annotation_ids",
    "notes", "required_action",
)


def font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def add_issue(report, code, message, record=None, annotation_id=None, severity="error"):
    item = {
        "severity": severity, "code": code, "message": message,
        "split": record["split"] if record else None,
        "image_id": record["image_id"] if record else None,
        "annotation_id": annotation_id,
    }
    report["issues"].append(item)
    if record is not None:
        record["warnings"].append(f"{code}: {message}")


def image_path(root, exported):
    """Resolve current relative paths or an exported Windows basename without guessing."""
    if not isinstance(exported, str) or not exported:
        raise ValueError("missing or non-string file_name")
    raw = (root / "data/raw/images").resolve()
    candidate = (root / exported.replace("\\", "/")).resolve()
    if candidate.is_file() and candidate.is_relative_to(raw):
        return candidate
    return resolve_image(root, exported)


def valid_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return "bbox must have four COCO xywh values"
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return "bbox has a non-finite or non-numeric coordinate"
    x, y, bw, bh = box
    if bw <= 0 or bh <= 0:
        return "bbox width and height must be positive"
    if x < 0 or y < 0 or x + bw > width + 1e-6 or y + bh > height + 1e-6:
        return f"bbox is outside actual image bounds {width}x{height}"
    return None


def validate_dataset(root, prepared):
    """Return every image record, including records that cannot be rendered."""
    report = {
        "counts": {}, "issues": [], "skipped": [], "annotation_id_zero": [],
        "notes": [
            "Structural validity does not establish that a defect location or class is semantically correct.",
            "An empty annotation list means no annotated boxes, not verified cleanliness.",
            "COCOeval treats annotation ID 0 as unmatched; image ID 0 and category ID 0 are allowed.",
        ],
    }
    records = []
    for split in SPLITS:
        path = prepared / f"{split}.json"
        try:
            data = read(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            add_issue(report, "split_unreadable", f"{path}: {exc}")
            report["counts"][split] = {"images": 0, "annotations": 0, "rendered": 0}
            continue
        images, annotations = data.get("images"), data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            add_issue(report, "invalid_coco_lists", f"{path}: images/annotations must be lists")
            report["counts"][split] = {"images": 0, "annotations": 0, "rendered": 0}
            continue
        report["counts"][split] = {
            "images": len(images), "annotations": len(annotations), "rendered": 0,
        }
        expected = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
        categories = data.get("categories", [])
        if categories != expected:
            add_issue(report, "category_mapping", f"{split}: categories differ from {expected}")
        names = {
            c.get("id"): c.get("name") for c in categories if isinstance(c, dict)
        }
        split_records = []
        by_id = {}
        for position, image_info in enumerate(images):
            info = image_info if isinstance(image_info, dict) else {}
            image_id = info.get("id")
            name = info.get("file_name", "")
            key = f"{position + 1:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(str(name).replace(chr(92), '/')).stem)[:55]}_{image_id}"
            record = {
                "split": split, "image_id": image_id, "filename": name,
                "key": key, "existing_status": info.get("annotation_status", "not_recorded"),
                "declared_size": (info.get("width"), info.get("height")),
                "actual_size": None, "path": None, "file_sha256": None,
                "annotations": [], "valid_annotations": [], "warnings": [],
                "paths": {}, "crop_entries": [], "augmentation_paths": [],
                "preprocessing": None,
            }
            records.append(record)
            split_records.append(record)
            if image_id in by_id:
                add_issue(report, "duplicate_image_id", f"{split}: image ID {image_id} occurs more than once", record)
                add_issue(report, "duplicate_image_id", f"{split}: image ID {image_id} occurs more than once", by_id[image_id])
            else:
                by_id[image_id] = record
            try:
                image_file = image_path(root, name)
                with Image.open(image_file) as opened:
                    width, height = opened.size
                    opened.load()
                record["path"] = image_file
                record["actual_size"] = (width, height)
                record["file_sha256"] = hashlib.sha256(image_file.read_bytes()).hexdigest()
                if record["declared_size"] != (width, height):
                    add_issue(report, "dimension_mismatch",
                              f"COCO says {record['declared_size']}; file is {width}x{height}", record)
            except Exception as exc:
                add_issue(report, "image_unreadable", f"{name}: {exc}", record)
                report["skipped"].append({
                    "split": split, "image_id": image_id, "filename": name,
                    "artifact": "image/crops/preprocessing", "reason": str(exc),
                })
        seen_annotations = set()
        for annotation in annotations:
            if not isinstance(annotation, dict):
                add_issue(report, "invalid_annotation", f"{split}: non-object annotation")
                continue
            ann_id = annotation.get("id")
            image_id = annotation.get("image_id")
            record = by_id.get(image_id)
            if ann_id in seen_annotations:
                add_issue(report, "duplicate_annotation_id",
                          f"{split}: annotation ID {ann_id} occurs more than once", record, ann_id)
            seen_annotations.add(ann_id)
            if ann_id == 0:
                report["annotation_id_zero"].append({"split": split, "image_id": image_id})
                add_issue(report, "annotation_id_zero",
                          "COCOeval uses annotation ID 0 as its unmatched sentinel", record, ann_id, "warning")
            if record is None:
                add_issue(report, "orphan_annotation",
                          f"{split}: annotation {ann_id} refers to absent image {image_id}",
                          annotation_id=ann_id)
                report["skipped"].append({
                    "split": split, "image_id": image_id, "annotation_id": ann_id,
                    "artifact": "crop", "reason": "orphan annotation",
                })
                continue
            record["annotations"].append(annotation)
            class_id = annotation.get("category_id")
            label = names.get(class_id)
            if class_id not in range(len(CLASSES)) or label != CLASSES[class_id]:
                code = "no_defect_box" if label == "no_defect" or class_id == 4 else "invalid_category"
                add_issue(report, code, f"annotation {ann_id} has box category {class_id} ({label})",
                          record, ann_id)
                continue
            if record["actual_size"] is None:
                add_issue(report, "box_unchecked", f"annotation {ann_id}: image unavailable", record, ann_id)
                continue
            problem = valid_box(annotation.get("bbox"), *record["actual_size"])
            if problem:
                add_issue(report, "invalid_bbox", f"annotation {ann_id}: {problem}", record, ann_id)
            else:
                record["valid_annotations"].append(annotation)
    # File identity and content identity both catch originals copied into another split.
    by_path, by_hash, by_image_id = defaultdict(list), defaultdict(list), defaultdict(list)
    for rec in records:
        if rec["path"] is not None:
            by_path[str(rec["path"])].append(rec)
            by_hash[rec["file_sha256"]].append(rec)
        by_image_id[rec["image_id"]].append(rec)
    for key, groups in (("same_file", by_path), ("same_bytes", by_hash), ("same_image_id", by_image_id)):
        for value, members in groups.items():
            if len({rec["split"] for rec in members}) > 1:
                for rec in members:
                    add_issue(report, "cross_split_" + key,
                              f"source {value} appears in splits {sorted({m['split'] for m in members})}", rec)
    return records, report


def annotation_label(annotation):
    class_id = annotation.get("category_id")
    name = CLASSES[class_id] if isinstance(class_id, int) and 0 <= class_id < 4 else str(class_id)
    return f"{name}  annotation #{annotation.get('id')}"


def panel(image, record, annotations, title, detail_lines=()):
    """Keep SEM pixels at their input resolution beneath a separate information panel."""
    content = image.convert("RGB").copy()
    canvas_draw = ImageDraw.Draw(content)
    label_font = font(12)
    for ann in annotations:
        x, y, bw, bh = ann["bbox"]
        color = COLORS[CLASSES[ann["category_id"]]]
        canvas_draw.rectangle((x, y, x + bw, y + bh), outline=color, width=1)
        label = annotation_label(ann)
        text_box = canvas_draw.textbbox((0, 0), label, font=label_font)
        label_width = text_box[2] - text_box[0]
        if y >= 16:
            canvas_draw.text((x, y - 15), label, fill=color, font=label_font)
        elif x + bw + label_width + 3 < image.width:
            canvas_draw.text((x + bw + 3, y), label, fill=color, font=label_font)
    lines = [
        title,
        f"Split: {record['split']}  |  Image ID: {record['image_id']}  |  File: {record['filename']}",
        f"SEM pixels: {image.width} x {image.height}  |  Annotated boxes: {len(record['annotations'])}",
        f"Existing status: {record['existing_status']}",
    ]
    if not record["annotations"]:
        lines.append(EMPTY_MESSAGE)
    lines.extend(detail_lines)
    lines.extend(annotation_label(ann) for ann in record["annotations"])
    if record["warnings"]:
        lines.append("STRUCTURAL WARNINGS: " + " | ".join(record["warnings"]))
    top = max(126, 20 * len(lines) + 30)
    width = max(image.width, 820)
    output = Image.new("RGB", (width, top + image.height), "#1F2429")
    output.paste(content, (0, top))
    draw = ImageDraw.Draw(output)
    head_font = font(14)
    for index, line in enumerate(lines):
        color = "#FFD480" if "WARNING" in line or EMPTY_MESSAGE in line else "#F4F5F7"
        draw.text((10, 8 + index * 20), str(line)[:130], font=head_font, fill=color)
    # A stable, separate legend; nothing from it covers source pixels.
    y = top - 21
    x = 10
    for name in CLASSES:
        draw.rectangle((x, y + 3, x + 12, y + 15), fill=COLORS[name])
        draw.text((x + 17, y), name, fill="white", font=font(11))
        x += 160
    return output


def crop_window(box, image_size, padding):
    x, y, bw, bh = box
    pad = max(16.0, max(bw, bh) * padding)
    left = max(0, math.floor(x - pad))
    top = max(0, math.floor(y - pad))
    right = min(image_size[0], math.ceil(x + bw + pad))
    bottom = min(image_size[1], math.ceil(y + bh + pad))
    return left, top, right, bottom


def crop_artifacts(image, record, annotation, bundle, padding, minimum_display):
    box = annotation["bbox"]
    left, top, right, bottom = crop_window(box, image.size, padding)
    plain = image.crop((left, top, right, bottom)).convert("RGB")
    native_w, native_h = plain.size
    marked = plain.copy()
    x, y, bw, bh = box
    marked_draw = ImageDraw.Draw(marked)
    marked_draw.rectangle(
        (x - left, y - top, x + bw - left, y + bh - top),
        outline=COLORS[CLASSES[annotation["category_id"]]], width=1,
    )
    short_label = f"{CLASSES[annotation['category_id']]} #{annotation['id']}"
    label_y = y - top - 14 if y - top >= 14 else min(native_h - 13, y + bh - top + 2)
    marked_draw.text((max(0, x - left), max(0, label_y)), short_label,
                     fill=COLORS[CLASSES[annotation["category_id"]]], font=font(11))
    scale = max(1.0, minimum_display / max(native_w, native_h))
    display_size = (max(1, round(native_w * scale)), max(1, round(native_h * scale)))
    shown_plain = plain.resize(display_size, Image.Resampling.NEAREST)
    shown_marked = marked.resize(display_size, Image.Resampling.NEAREST)
    header = 77
    gap = 14
    comparison = Image.new("RGB", (display_size[0] * 2 + gap, display_size[1] + header), "#20252B")
    comparison.paste(shown_plain, (0, header))
    comparison.paste(shown_marked, (display_size[0] + gap, header))
    d = ImageDraw.Draw(comparison)
    d.text((7, 5), annotation_label(annotation), fill="white", font=font(14))
    d.text((7, 26), f"Unmarked (left) / ground truth (right); crop origin ({left},{top})", fill="white", font=font(12))
    d.text((7, 46), f"Display magnification x{scale:.2f}; no new image detail", fill="#FFD480", font=font(12))
    base = Path("crops") / record["split"] / record["key"] / f"annotation_{annotation['id']}"
    paths = {
        "unmarked": (base.with_name(base.name + "_unmarked.png")).as_posix(),
        "marked": (base.with_name(base.name + "_marked.png")).as_posix(),
        "comparison": (base.with_name(base.name + "_comparison.png")).as_posix(),
    }
    for path, rendered in ((paths["unmarked"], plain), (paths["marked"], marked),
                           (paths["comparison"], comparison)):
        target = bundle / path
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(target, format="PNG")
    return {
        "split": record["split"], "image_id": record["image_id"],
        "annotation_id": annotation["id"], "class": CLASSES[annotation["category_id"]],
        "original_bbox_xywh": box, "crop_xyxy": [left, top, right, bottom],
        "crop_origin": [left, top], "crop_dimensions": [native_w, native_h],
        "display_scale": scale, "display_dimensions": list(display_size), "paths": paths,
    }


def make_contact_sheet(records, split, bundle):
    existing = [rec for rec in records if rec["split"] == split and rec["paths"].get("overlay")]
    if not existing:
        return None
    columns, tile_w, tile_h = 6, 210, 205
    rows = math.ceil(len(existing) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h + 50), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 6), f"{split.upper()} - navigation only; open full previews and crops to check small defects",
              fill="black", font=font(15))
    for index, rec in enumerate(existing):
        with Image.open(bundle / rec["paths"]["overlay"]) as opened:
            thumbnail = ImageOps.contain(opened.convert("RGB"), (196, 166))
        x = (index % columns) * tile_w + 7
        y = (index // columns) * tile_h + 50
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y + 170), str(rec["filename"])[:25], fill="black", font=font(10))
    relative = f"contact_sheets/{split}.png"
    target = bundle / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG")
    return relative


def generate_base_artifacts(records, report, bundle, crop_padding, min_display):
    crops = []
    for rec in records:
        if rec["path"] is None:
            continue
        try:
            with Image.open(rec["path"]) as opened:
                image = opened.convert("RGB")
            original_rel = f"originals/{rec['split']}/{rec['key']}.png"
            overlay_rel = f"overlays/{rec['split']}/{rec['key']}.png"
            for rel in (original_rel, overlay_rel):
                (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
            image.save(bundle / original_rel, format="PNG")
            panel(image, rec, rec["valid_annotations"], "GROUND-TRUTH BOX REVIEW").save(
                bundle / overlay_rel, format="PNG"
            )
            rec["paths"].update(original=original_rel, overlay=overlay_rel)
            report["counts"][rec["split"]]["rendered"] += 1
            for ann in rec["annotations"]:
                if ann not in rec["valid_annotations"]:
                    report["skipped"].append({
                        "split": rec["split"], "image_id": rec["image_id"],
                        "annotation_id": ann.get("id"), "artifact": "crop",
                        "reason": "structurally invalid annotation; see issues",
                    })
                    continue
                try:
                    entry = crop_artifacts(image, rec, ann, bundle, crop_padding, min_display)
                    rec["crop_entries"].append(entry)
                    crops.append(entry)
                except Exception as exc:
                    add_issue(report, "crop_failed", str(exc), rec, ann.get("id"))
                    report["skipped"].append({
                        "split": rec["split"], "image_id": rec["image_id"],
                        "annotation_id": ann.get("id"), "artifact": "crop", "reason": str(exc),
                    })
        except Exception as exc:
            add_issue(report, "render_failed", str(exc), rec)
            report["skipped"].append({
                "split": rec["split"], "image_id": rec["image_id"],
                "artifact": "original/overlay/crops", "reason": str(exc),
            })
    return crops


def generate_preprocessing(records, report, bundle, size):
    try:
        import torch
        from sem_yolos.data import letterbox, inverse_boxes
    except Exception as exc:
        report["preprocessing"] = {"status": "unavailable", "reason": str(exc)}
        return
    completed = 0
    for rec in records:
        if rec["path"] is None:
            continue
        try:
            with Image.open(rec["path"]) as opened:
                image = opened.convert("RGB")
            source = [a["bbox"] for a in rec["valid_annotations"]]
            canvas, transformed, meta = letterbox(image, source, size)
            source_xyxy = torch.tensor(
                [[x, y, x + w, y + h] for x, y, w, h in source], dtype=torch.float32
            ).reshape(-1, 4)
            transformed_xyxy = transformed.clone()
            transformed_xyxy[:, 2:] += transformed_xyxy[:, :2]
            recovered = inverse_boxes(transformed_xyxy, meta)
            maximum_error = float((recovered - source_xyxy).abs().max()) if source else 0.0
            if maximum_error > 0.001:
                add_issue(report, "preprocessing_inverse_mismatch",
                          f"maximum inverse box error {maximum_error:.6f} px", rec)
            draw_rec = dict(rec, annotations=rec["valid_annotations"])
            canvas_anns = [dict(a, bbox=b.tolist()) for a, b in zip(rec["valid_annotations"], transformed)]
            preview = panel(canvas, draw_rec, canvas_anns, "DETERMINISTIC LETTERBOX PREVIEW", (
                f"Canvas {size} x {size}; resize x={meta['scale_x']:.5f}, y={meta['scale_y']:.5f}; pad left={meta['left']}, top={meta['top']}",
                f"Inverse transform maximum error: {maximum_error:.6f} source pixels; before normalization",
                "Canvas box sizes: " + (", ".join(
                    f"#{a.get('id')} {float(b[2]):.2f}x{float(b[3]):.2f}px"
                    for a, b in zip(rec["valid_annotations"], transformed)) or "none"),
            ))
            rel = f"preprocessing/{rec['split']}/{rec['key']}.png"
            (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
            preview.save(bundle / rel)
            rec["preprocessing"] = {"path": rel, "metadata": meta,
                                    "inverse_max_error_pixels": maximum_error}
            completed += 1
        except Exception as exc:
            add_issue(report, "preprocessing_failed", str(exc), rec)
            report["skipped"].append({"split": rec["split"], "image_id": rec["image_id"],
                                      "artifact": "preprocessing", "reason": str(exc)})
    report["preprocessing"] = {"status": "generated", "images": completed, "canvas_size": size}


def select_augmentation_records(records, all_training):
    train = [r for r in records if r["split"] == "train" and r["path"] is not None]
    if all_training:
        return train
    chosen = [r for r in train if any(a.get("category_id") == 1 for a in r["valid_annotations"])]
    seen = {id(r) for r in chosen}
    for category in (0, 2, 3, None):
        matches = [r for r in train if (not r["annotations"] if category is None else
                   any(a.get("category_id") == category for a in r["valid_annotations"]))]
        for rec in matches[:1]:
            if id(rec) not in seen:
                chosen.append(rec)
                seen.add(id(rec))
    return chosen


def generate_augmentations(records, report, bundle, root, prepared, config, size, seed,
                           all_training):
    try:
        import numpy as np
        import torch
        from sem_yolos.data import CocoDetectionDataset
    except Exception as exc:
        report["augmentation"] = {"status": "unavailable", "reason": str(exc)}
        return
    recipes = config.get("augmentation_recipes", [])
    if config.get("training_view_mode") != "original_plus_recipes" or not recipes:
        report["augmentation"] = {"status": "unavailable", "reason":
                                  "config has no original_plus_recipes training views"}
        return
    selected = select_augmentation_records(records, all_training)
    try:
        dataset = CocoDetectionDataset(root, prepared / "train.json", None,
                                       size=size, training=True, view_mode="original_plus_recipes",
                                       recipes=recipes)
    except Exception as exc:
        report["augmentation"] = {"status": "unavailable", "reason": str(exc)}
        return
    # Use the training implementation; no model or processor is loaded.
    by_id = {info["id"]: i for i, info in enumerate(dataset.images)}
    count = 0
    for rec in selected:
        if rec["image_id"] not in by_id:
            add_issue(report, "augmentation_image_missing", "image absent from train dataset", rec)
            continue
        source_index = by_id[rec["image_id"]]
        tiles = []
        for view in range(dataset.views_per_image):
            name = "original" if view == 0 else recipes[view - 1]["name"]
            try:
                local_seed = seed + source_index * 1009 + view
                random.seed(local_seed)
                np.random.seed(local_seed % (2 ** 32))
                torch.manual_seed(local_seed)
                canvas, boxes, labels, meta = dataset.transformed(
                    source_index + view * len(dataset.images))
                anns = [dict(id=a.get("id", "?"), category_id=int(c), bbox=b.tolist())
                        for a, b, c in zip(rec["annotations"], boxes, labels)]
                shown = panel(canvas, dict(rec, annotations=anns), anns,
                              f"TRAINING VIEW: {name}",
                              (f"Seed {local_seed}; source image {rec['image_id']}; canvas {size} x {size}",
                               "Transformed boxes from training augmentation; preview only"))
                rel = f"augmentation/train/{rec['key']}/{view:02d}_{name}.png"
                target = bundle / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shown.save(target)
                rec["augmentation_paths"].append({"view": view, "recipe": name,
                                                  "seed": local_seed, "path": rel,
                                                  "boxes_xywh": [b.tolist() for b in boxes],
                                                  "letterbox_metadata": meta})
                tile = ImageOps.contain(shown, (220, 210))
                tiles.append((name, tile))
                count += 1
            except Exception as exc:
                add_issue(report, "augmentation_failed", f"{name}: {exc}", rec)
                report["skipped"].append({"split": "train", "image_id": rec["image_id"],
                                          "artifact": "augmentation", "recipe": name,
                                          "reason": str(exc)})
        if tiles:
            columns, tile_w, tile_h = 5, 240, 245
            sheet = Image.new("RGB", (columns * tile_w,
                                      math.ceil(len(tiles) / columns) * tile_h), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (name, tile) in enumerate(tiles):
                x, y = (index % columns) * tile_w + 10, (index // columns) * tile_h + 5
                sheet.paste(tile, (x, y))
                draw.text((x, y + 215), name, fill="black", font=font(12))
            rel = f"augmentation/contact_sheets/{rec['key']}.png"
            (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
            sheet.save(bundle / rel)
            rec["paths"]["augmentation_contact"] = rel
    report["augmentation"] = {"status": "generated", "source_images": len(selected),
                              "views": count, "views_per_image": dataset.views_per_image,
                              "seed": seed, "configuration": config.get("training_view_mode"),
                              "all_training": all_training}


def write_review_csv(records, destination):
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rec in records:
            classes = sorted({CLASSES[a["category_id"]] for a in rec["valid_annotations"]})
            writer.writerow({
                "split": rec["split"], "image_id": rec["image_id"],
                "filename": rec["filename"],
                "image_width": rec["actual_size"][0] if rec["actual_size"] else "",
                "image_height": rec["actual_size"][1] if rec["actual_size"] else "",
                "annotation_count": len(rec["annotations"]),
                "annotation_ids": ";".join(str(a.get("id")) for a in rec["annotations"]),
                "current_box_classes": ";".join(classes),
                "existing_annotation_status": rec["existing_status"],
                "structural_warnings": "; ".join(rec["warnings"]),
                "review_status": "UNREVIEWED",
                "clean_review_status": "UNREVIEWED" if not rec["annotations"] else "NOT_APPLICABLE",
                "issue_types": "", "affected_annotation_ids": "", "notes": "",
                "required_action": "",
            })


def link(path, title=None):
    if not path:
        return "<span class='missing'>unavailable</span>"
    escaped = html.escape(str(path), quote=True)
    return f'<a href="{escaped}">{html.escape(str(title or path))}</a>'


def write_index(records, report, bundle, contact_sheets):
    e = html.escape
    pieces = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SEM ground-truth annotation review</title><style>
body{font-family:system-ui,sans-serif;background:#111820;color:#ecf2f5;margin:0;padding:1.4rem;line-height:1.5}
a{color:#83d8ff}h1,h2,h3{line-height:1.2}p{max-width:75rem}.intro,.record,.issue{background:#202a33;border:1px solid #53636e;border-radius:8px;padding:1rem;margin:1rem 0}
.warning{color:#ffd887}.missing{color:#ff9d9d}.group{border-top:2px solid #6c8798;margin-top:2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:1rem}
.record img{max-width:100%;height:210px;object-fit:contain;background:#13191f;border:1px solid #51606d}.record code{overflow-wrap:anywhere}details{margin:.5rem 0}small{color:#b9c8d0}
</style></head><body><h1>SEM ground-truth annotation review</h1>
<div class="intro"><p>These are prepared COCO ground-truth annotations only. No model predictions or semantic correctness decisions are included.</p>
<p>Start with the full-resolution overlay, open each unmarked crop, then compare marked crops. An empty annotation list does not prove that an image is clean. Contact sheets are navigation aids only.</p>
<p>Record decisions in the <a href="review.csv">review CSV</a>. See <a href="validation_report.json">structural validation JSON</a>, <a href="artifact_manifest.json">artifact manifest</a>, and <a href="REVIEW_GUIDE.md">review guide</a>.</p></div>"""]
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        count = report["counts"].get(split, {})
        pieces.append(f'<section class="group"><h2>{split.upper()}: {count.get("images", 0)} images, '
                      f'{count.get("annotations", 0)} boxes</h2>')
        if contact_sheets.get(split):
            pieces.append(f'<p>{link(contact_sheets[split], "Open contact sheet")}</p>')
        for group_name in (*CLASSES, "no_annotated_boxes"):
            group = [r for r in subset if (not r["annotations"] if group_name == "no_annotated_boxes"
                     else any(a.get("category_id") == CLASSES.index(group_name) for a in r["annotations"]))]
            pieces.append(f'<h3>{e(group_name.replace("_", " "))} ({len(group)})</h3><div class="grid">')
            for rec in group:
                name = e(str(rec["filename"]))
                pieces.append(f'<article class="record"><h4>{name}</h4><p>Image ID {e(str(rec["image_id"]))}; '
                              f'{len(rec["annotations"])} annotated boxes; existing status: '
                              f'{e(str(rec["existing_status"]))}</p>')
                if not rec["annotations"]:
                    pieces.append(f'<p class="warning">{e(EMPTY_MESSAGE)}</p>')
                if rec["warnings"]:
                    pieces.append('<p class="warning">' + e(" | ".join(rec["warnings"])) + '</p>')
                overlay = rec["paths"].get("overlay")
                if overlay:
                    pieces.append(f'<a href="{e(overlay, quote=True)}"><img src="{e(overlay, quote=True)}" '
                                  f'alt="Ground-truth overlay for {name}"></a>')
                pieces.append('<p>Full resolution: ' + link(overlay, "overlay") + ' · ' +
                              link(rec["paths"].get("original"), "unmarked original") + '</p>')
                pieces.append('<p>Preprocessing: ' + link((rec["preprocessing"] or {}).get("path"),
                                                            "letterbox and boxes") + '</p>')
                for crop in rec["crop_entries"]:
                    paths = crop["paths"]
                    pieces.append('<p>Box ' + e(str(crop["annotation_id"])) + ' (' +
                                  e(crop["class"]) + '): ' + link(paths["comparison"], "side by side") +
                                  ' · ' + link(paths["unmarked"], "unmarked crop") +
                                  ' · ' + link(paths["marked"], "marked crop") + '</p>')
                if rec["augmentation_paths"]:
                    pieces.append('<details><summary>Training views (' +
                                  str(len(rec["augmentation_paths"])) + ')</summary><p>' +
                                  link(rec["paths"].get("augmentation_contact"), "view contact sheet") +
                                  '</p><p>' + ' · '.join(link(a["path"], a["recipe"])
                                                          for a in rec["augmentation_paths"]) +
                                  '</p></details>')
                pieces.append('</article>')
            pieces.append('</div>')
        pieces.append('</section>')
    pieces.append('<section class="group"><h2>Structural issues and skipped artifacts</h2>')
    if report["issues"]:
        for issue in report["issues"]:
            pieces.append('<div class="issue"><strong>' + e(issue["code"]) + '</strong> [' +
                          e(issue["severity"]) + '] ' + e(str(issue["message"])) +
                          ' · split=' + e(str(issue["split"])) +
                          ' image=' + e(str(issue["image_id"])) +
                          ' annotation=' + e(str(issue["annotation_id"])) + '</div>')
    else:
        pieces.append('<p>No structural issues detected. Semantic review is still required.</p>')
    if report["skipped"]:
        pieces.append('<h3>Skipped</h3><pre>' + e(json.dumps(report["skipped"], indent=2)) + '</pre>')
    pieces.append('</section></body></html>')
    (bundle / "index.html").write_text("\n".join(pieces), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a local, read-only SEM annotation review bundle")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared/label_studio_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/manual_annotation_review"))
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline_config_20_views.json"))
    parser.add_argument("--canvas-size", type=int, default=None)
    parser.add_argument("--crop-padding", type=float, default=1.5)
    parser.add_argument("--min-crop-display", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--all-training-augmentations", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    prepared = (root / args.prepared).resolve()
    config_path = (root / args.config).resolve()
    allowed_output = (root / "outputs/manual_annotation_review").resolve()
    output_root = (root / args.output_root).resolve()
    if not output_root.is_relative_to(allowed_output):
        parser.error("--output-root must be inside outputs/manual_annotation_review")
    if args.crop_padding < 0 or args.min_crop_display <= 0:
        parser.error("crop padding must be nonnegative and minimum display must be positive")
    config = read(config_path)
    size = args.canvas_size or config.get("size", 512)
    if not isinstance(size, int) or size <= 0:
        parser.error("canvas size must be a positive integer")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("review_%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    bundle = output_root / run_name
    bundle.mkdir(exist_ok=False)
    records, report = validate_dataset(root, prepared)
    manifest = read(prepared / "split_manifest.json")
    current_fingerprint = prepared_fingerprint(prepared)
    if manifest.get("split_fingerprint") != current_fingerprint:
        add_issue(report, "split_fingerprint_mismatch",
                  "Prepared COCO files differ from the split manifest; rerun prepare_data.py")
    report["source"] = {"root": str(root), "prepared": str(prepared),
                        "config": str(config_path), "generated_at": datetime.now().isoformat(),
                        "split_fingerprint": current_fingerprint}
    crops = generate_base_artifacts(records, report, bundle, args.crop_padding,
                                    args.min_crop_display)
    generate_preprocessing(records, report, bundle, size)
    if args.augmentations:
        generate_augmentations(records, report, bundle, root, prepared, config, size,
                               args.seed, args.all_training_augmentations)
    else:
        report["augmentation"] = {"status": "disabled"}
    contact_sheets = {split: make_contact_sheet(records, split, bundle) for split in SPLITS}
    write_review_csv(records, bundle / "review.csv")
    report["summary"] = {"images": len(records), "annotations": sum(len(r["annotations"]) for r in records),
                         "empty_images": sum(not r["annotations"] for r in records),
                         "crops": len(crops), "issues": len(report["issues"]),
                         "review_decisions": 0}
    manifest = {"bundle": str(bundle), "summary": report["summary"], "crops": crops,
                "images": [{"split": r["split"], "image_id": r["image_id"],
                            "filename": r["filename"], "actual_size": r["actual_size"],
                            "annotation_ids": [a.get("id") for a in r["annotations"]],
                            "existing_annotation_status": r["existing_status"],
                            "warnings": r["warnings"], "paths": r["paths"],
                            "preprocessing": r["preprocessing"],
                            "augmentation_views": r["augmentation_paths"],
                            "crops": r["crop_entries"]} for r in records]}
    (bundle / "validation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (bundle / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_index(records, report, bundle, contact_sheets)
    (bundle / "REVIEW_GUIDE.md").write_text(REVIEW_GUIDE, encoding="utf-8")
    print(json.dumps({"bundle": str(bundle), "index": str(bundle / "index.html"),
                      "csv": str(bundle / "review.csv"), "report": str(bundle / "validation_report.json"),
                      "summary": report["summary"], "preprocessing": report["preprocessing"],
                      "augmentation": report["augmentation"]}, indent=2))
    return bundle


REVIEW_GUIDE = """# Manual SEM annotation review

Open `index.html` locally in a browser. Follow its split and class sections. Every box has
a full-resolution overlay and native-pixel unmarked/marked crops. The side-by-side view
may magnify pixels for visibility; it adds no detail. The displayed crop origin and
original COCO `xywh` box are recorded in `artifact_manifest.json`.

Inspect the unmarked image and crop before judging the ground-truth outline. For every
image, set `review_status` in `review.csv` to your manual decision and record issue
types, affected annotation IDs, notes, and required action. For images with no boxes,
set `clean_review_status` only after examining the original at full resolution.
`existing_annotation_status` is inherited metadata and does not constitute this review.

Suggested `review_status` values: `UNREVIEWED`, `PASS`, `FIX`, `UNCERTAIN`.
Suggested `clean_review_status` values: `NOT_APPLICABLE`, `UNREVIEWED`,
`CONFIRMED_CLEAN`, `DEFECT_PRESENT`, `UNCERTAIN`. Suggested `issue_types`:
`wrong_class`, `misplaced_box`, `box_too_large`, `box_too_small`,
`missing_defect`, `duplicate_box`, `uncertain_clean`, `uncertain_source`.

The structural report checks file availability, dimensions, categories, ID uniqueness,
orphan annotations, finite positive in-bounds boxes, no-defect boxes, annotation ID 0,
and repeated originals across splits. A pass does not establish semantic correctness.
The preprocessing preview uses the actual training letterbox and checks its inverse.
Training augmentation previews use the configured recipes and transformed boxes; they
are visual examples only. A transformed box staying in bounds does not establish
physical validity. Annotation verification is separate from model evaluation.
No model inference, training, or annotation edits are done.
"""


if __name__ == "__main__":
    main()
