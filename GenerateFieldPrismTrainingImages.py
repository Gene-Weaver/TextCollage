#!/usr/bin/env python3
"""
GenerateFieldPrismTrainingImages.py

Generates a single-class (ruler) YOLO dataset from PREP_final, focused on
FieldPrism-style rulers only.

Processing:
  1. FieldPrism images (filename stem has < 3 underscores):
     - Copied to output with only ruler (class 0) annotations kept.
  2. Standalone ruler crops:
     - Each ruler bbox from FieldPrism images is extracted and saved as its
       own training image with label `0 0.5 0.5 1.0 1.0`.
  3. Non-FieldPrism images (filename stem has >= 3 underscores):
     - Original annotations discarded. 1-10 FieldPrism ruler crops are pasted
       onto the image via copy/paste augmentation (random scale, rotation,
       non-overlapping placement, soft-mask blending). Total pasted area
       capped at 80% of image area.

Output:
  /datac/FieldPrism_YOLO_Training_Data/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml   (nc: 1, names: ['ruler'])

Usage:
  python GenerateFieldPrismTrainingImages.py \\
      --source /home/brlab/Dropbox/TextCollage/datasets/PREP_final/ \\
      --output /datac/FieldPrism_YOLO_Training_Data/ \\
      --min-rulers 1 --max-rulers 10 --seed 2025 --workers 16
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage


# ── Aspect ratio filter ──────────────────────────────────────────────────────

MAX_RULER_ASPECT_RATIO = 3.0

def is_valid_ruler_aspect(w_norm: float, h_norm: float,
                          img_w: int, img_h: int) -> bool:
    """True if the ruler bbox has a pixel aspect ratio within 1:3 – 3:1.

    FieldPrism rulers are nearly square; rogue rulers are long and skinny.
    The normalised bbox dimensions must be converted to pixels first because
    images vary greatly in resolution / aspect ratio.
    """
    pw = w_norm * img_w
    ph = h_norm * img_h
    if pw < 1 or ph < 1:
        return False
    ratio = max(pw, ph) / min(pw, ph)
    return ratio <= MAX_RULER_ASPECT_RATIO


def _get_image_size(path: str) -> Tuple[int, int]:
    """Return (width, height) using a fast PIL header read."""
    with PILImage.open(path) as img:
        return img.size  # (w, h)


# ── Filename classification ──────────────────────────────────────────────────

def is_fieldprism(filename: str) -> bool:
    """FieldPrism images have < 3 underscores in the stem."""
    stem = os.path.splitext(filename)[0]
    return stem.count("_") < 3


# ── Label I/O ────────────────────────────────────────────────────────────────

def parse_label_file(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO label file. Returns list of (class_id, xc, yc, w, h)."""
    if not os.path.exists(label_path):
        return []
    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append((cls, xc, yc, w, h))
            except ValueError:
                continue
    return annotations


def write_label_file(label_path: str, annotations: List[Tuple[int, float, float, float, float]]):
    """Write YOLO-format annotations to a label file."""
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in annotations:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ── Crop extraction ──────────────────────────────────────────────────────────

def extract_crop(image: np.ndarray, bbox: Tuple[float, float, float, float],
                 min_pixels: int = 30) -> Optional[np.ndarray]:
    """Extract a crop from *image* at normalised *bbox* (xc, yc, w, h).
    Returns None if the crop is smaller than *min_pixels* in either dim."""
    h_img, w_img = image.shape[:2]
    xc, yc, w, h = bbox
    x1 = int((xc - w / 2) * w_img)
    y1 = int((yc - h / 2) * h_img)
    x2 = int((xc + w / 2) * w_img)
    y2 = int((yc + h / 2) * h_img)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if (x2 - x1) < min_pixels or (y2 - y1) < min_pixels:
        return None
    return image[y1:y2, x1:x2].copy()


# ── Copy/paste augmentation helpers ──────────────────────────────────────────

def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Check if two (x1, y1, x2, y2) rectangles overlap."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _rect_area(r: Tuple[int, int, int, int]) -> int:
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def paste_ruler(
    image: np.ndarray,
    crop: np.ndarray,
    placed_rects: List[Tuple[int, int, int, int]],
    cumulative_area: int,
    max_area: int,
    rng: np.random.Generator,
    max_attempts: int = 50,
) -> Optional[Tuple[np.ndarray, Tuple[int, float, float, float, float], Tuple[int, int, int, int], int]]:
    """Paste a single ruler *crop* onto *image* with random augmentation.

    Returns (modified_image, yolo_annotation, placed_rect, crop_area) or None.
    """
    h_img, w_img = image.shape[:2]
    h_crop, w_crop = crop.shape[:2]

    # --- random scale (0.3x – 2.0x), clamped to fit inside target image ---
    scale = rng.uniform(0.3, 2.0)
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    # Clamp so the crop (before rotation) fits in the image
    if new_w > w_img:
        ratio = w_img / new_w
        new_w = w_img
        new_h = max(1, int(new_h * ratio))
    if new_h > h_img:
        ratio = h_img / new_h
        new_h = h_img
        new_w = max(1, int(new_w * ratio))
    if new_w < 10 or new_h < 10:
        return None
    scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # --- random rotation (-30 to +30 degrees) ---
    angle = rng.uniform(-30, 30)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    rot_w = int(new_h * sin_a + new_w * cos_a)
    rot_h = int(new_h * cos_a + new_w * sin_a)
    M[0, 2] += (rot_w - new_w) / 2
    M[1, 2] += (rot_h - new_h) / 2

    if rot_w > w_img or rot_h > h_img or rot_w < 10 or rot_h < 10:
        return None

    rotated = cv2.warpAffine(scaled, M, (rot_w, rot_h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Build mask from a white rectangle with the same rotation
    mask_src = np.ones((new_h, new_w), dtype=np.uint8) * 255
    mask_rotated = cv2.warpAffine(mask_src, M, (rot_w, rot_h),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Tight bounding rect of the mask content
    coords = cv2.findNonZero(mask_rotated)
    if coords is None:
        return None
    rx, ry, rw, rh = cv2.boundingRect(coords)
    crop_pixel_area = int(np.count_nonzero(mask_rotated))

    # Check area budget
    if cumulative_area + crop_pixel_area > max_area:
        return None

    # --- find non-overlapping placement (up to max_attempts tries) ---
    for _ in range(max_attempts):
        x_off = int(rng.integers(0, max(1, w_img - rot_w)))
        y_off = int(rng.integers(0, max(1, h_img - rot_h)))

        candidate_rect = (x_off + rx, y_off + ry, x_off + rx + rw, y_off + ry + rh)

        # Check overlap with every existing rect
        if any(_rects_overlap(candidate_rect, pr) for pr in placed_rects):
            continue

        # --- soft-mask blending ---
        mask_float = mask_rotated.astype(np.float32) / 255.0
        blur_k = max(3, (min(rot_w, rot_h) // 10) | 1)  # odd kernel
        mask_blurred = cv2.GaussianBlur(mask_float, (blur_k, blur_k), 0)
        mask_3ch = np.stack([mask_blurred] * 3, axis=-1)

        roi = image[y_off:y_off + rot_h, x_off:x_off + rot_w]
        blended = (rotated.astype(np.float32) * mask_3ch +
                   roi.astype(np.float32) * (1.0 - mask_3ch))
        image[y_off:y_off + rot_h, x_off:x_off + rot_w] = blended.astype(np.uint8)

        # YOLO annotation (normalised)
        abs_cx = (candidate_rect[0] + candidate_rect[2]) / 2.0
        abs_cy = (candidate_rect[1] + candidate_rect[3]) / 2.0
        ann = (0,
               abs_cx / w_img,
               abs_cy / h_img,
               rw / w_img,
               rh / h_img)

        return image, ann, candidate_rect, crop_pixel_area

    return None  # could not place without overlap


# ── Top-level worker functions (picklable) ───────────────────────────────────

def _worker_process_fieldprism(args: dict):
    """Copy a FieldPrism image and filter its labels to valid rulers only."""
    img_path = args["img_path"]
    lbl_path = args["lbl_path"]
    out_img_path = args["out_img_path"]
    out_lbl_path = args["out_lbl_path"]

    shutil.copy2(img_path, out_img_path)

    # Get image dims for aspect-ratio check
    try:
        img_w, img_h = _get_image_size(img_path)
    except Exception:
        img_w, img_h = 1, 1  # fallback: keep all if we can't read

    annotations = parse_label_file(lbl_path)
    ruler_anns = [
        (cls, xc, yc, w, h)
        for cls, xc, yc, w, h in annotations
        if cls == 0 and is_valid_ruler_aspect(w, h, img_w, img_h)
    ]
    write_label_file(out_lbl_path, ruler_anns)
    return {"type": "fieldprism", "has_rulers": len(ruler_anns) > 0}


def _worker_extract_crop(args: dict):
    """Extract a single ruler crop and save it as a standalone image."""
    img_path = args["img_path"]
    bbox = args["bbox"]  # (xc, yc, w, h)
    out_img_path = args["out_img_path"]
    out_lbl_path = args["out_lbl_path"]
    min_pixels = args.get("min_pixels", 30)

    image = cv2.imread(img_path)
    if image is None:
        return {"type": "crop", "success": False}

    crop = extract_crop(image, bbox, min_pixels=min_pixels)
    if crop is None:
        return {"type": "crop", "success": False}

    cv2.imwrite(out_img_path, crop)
    write_label_file(out_lbl_path, [(0, 0.5, 0.5, 1.0, 1.0)])
    return {"type": "crop", "success": True}


def _worker_process_non_fieldprism(args: dict):
    """Paste FieldPrism ruler crops onto a non-FieldPrism image."""
    img_path = args["img_path"]
    out_img_path = args["out_img_path"]
    out_lbl_path = args["out_lbl_path"]
    ruler_index = args["ruler_index"]  # list of {"image_path": str, "bbox": (xc, yc, w, h)}
    min_rulers = args["min_rulers"]
    max_rulers = args["max_rulers"]
    seed = args["seed"]
    min_pixels = args.get("min_pixels", 30)

    rng = np.random.default_rng(seed)

    image = cv2.imread(img_path)
    if image is None:
        return {"type": "non_fieldprism", "n_pasted": 0}

    h_img, w_img = image.shape[:2]
    max_area = int(h_img * w_img * 0.8)  # 80% cap

    n_rulers = int(rng.integers(min_rulers, max_rulers + 1))

    placed_rects: List[Tuple[int, int, int, int]] = []
    annotations: List[Tuple[int, float, float, float, float]] = []
    cumulative_area = 0

    # Pre-select ruler entries (with replacement if index is small)
    idx_choices = rng.integers(0, len(ruler_index), size=n_rulers)

    for idx in idx_choices:
        entry = ruler_index[idx]
        src_img = cv2.imread(entry["image_path"])
        if src_img is None:
            continue

        crop = extract_crop(src_img, tuple(entry["bbox"]), min_pixels=min_pixels)
        if crop is None:
            continue

        result = paste_ruler(image, crop, placed_rects, cumulative_area, max_area, rng)
        if result is None:
            continue

        image, ann, rect, area = result
        placed_rects.append(rect)
        annotations.append(ann)
        cumulative_area += area

        if cumulative_area >= max_area:
            break

    cv2.imwrite(out_img_path, image)
    write_label_file(out_lbl_path, annotations)
    return {"type": "non_fieldprism", "n_pasted": len(annotations)}


# ── Main generator class ─────────────────────────────────────────────────────

class FieldPrismDatasetGenerator:
    """Generates a single-class (ruler) YOLO dataset for FieldPrism."""

    SPLITS = ("train", "val", "test")

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        min_rulers: int = 1,
        max_rulers: int = 10,
        seed: int = 2025,
        min_crop_pixels: int = 30,
        workers: int = 0,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.min_rulers = min_rulers
        self.max_rulers = max_rulers
        self.seed = seed
        self.min_crop_pixels = min_crop_pixels
        self.workers = workers if workers > 0 else os.cpu_count() or 4
        self.rng = np.random.default_rng(seed)

    # ── index building ────────────────────────────────────────────────────

    def collect_ruler_crop_index(self, split: str) -> List[dict]:
        """Build an index of all ruler bboxes in FieldPrism images for *split*.

        Filters out rogue rulers whose pixel aspect ratio exceeds 3:1.
        """
        img_dir = os.path.join(self.source_dir, "images", split)
        lbl_dir = os.path.join(self.source_dir, "labels", split)
        index = []
        skipped = 0
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if not is_fieldprism(fname):
                continue
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            annotations = parse_label_file(lbl_path)
            rulers = [a for a in annotations if a[0] == 0]
            if not rulers:
                continue
            # Fast header read for image dimensions
            img_path = os.path.join(img_dir, fname)
            try:
                img_w, img_h = _get_image_size(img_path)
            except Exception:
                continue
            for ann in rulers:
                _, xc, yc, w, h = ann
                if is_valid_ruler_aspect(w, h, img_w, img_h):
                    index.append({
                        "image_path": img_path,
                        "bbox": (xc, yc, w, h),
                    })
                else:
                    skipped += 1
        if skipped:
            print(f"    Skipped {skipped} rogue rulers (aspect ratio > {MAX_RULER_ASPECT_RATIO}:1)")
        return index

    # ── generation ────────────────────────────────────────────────────────

    def generate(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for split in self.SPLITS:
            print(f"\n{'=' * 60}")
            print(f"Processing split: {split}")
            print(f"{'=' * 60}")

            out_img_dir = os.path.join(self.output_dir, "images", split)
            out_lbl_dir = os.path.join(self.output_dir, "labels", split)
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            img_dir = os.path.join(self.source_dir, "images", split)
            lbl_dir = os.path.join(self.source_dir, "labels", split)

            # Build ruler crop index
            ruler_index = self.collect_ruler_crop_index(split)
            print(f"  Ruler crop index: {len(ruler_index)} entries from FieldPrism images")

            if len(ruler_index) == 0:
                print("  WARNING: No ruler crops found. Non-FieldPrism images will be skipped.")

            # Classify images into tasks
            all_images = sorted(
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            fp_tasks = []
            nfp_tasks = []
            crop_tasks = []

            for fname in all_images:
                stem = os.path.splitext(fname)[0]
                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, stem + ".txt")

                if is_fieldprism(fname):
                    fp_tasks.append({
                        "img_path": img_path,
                        "lbl_path": lbl_path,
                        "out_img_path": os.path.join(out_img_dir, fname),
                        "out_lbl_path": os.path.join(out_lbl_dir, stem + ".txt"),
                    })
                else:
                    if ruler_index:
                        # Deterministic per-image seed
                        img_seed = self.rng.integers(0, 2**31)
                        nfp_tasks.append({
                            "img_path": img_path,
                            "out_img_path": os.path.join(out_img_dir, fname),
                            "out_lbl_path": os.path.join(out_lbl_dir, stem + ".txt"),
                            "ruler_index": ruler_index,
                            "min_rulers": self.min_rulers,
                            "max_rulers": self.max_rulers,
                            "seed": int(img_seed),
                            "min_pixels": self.min_crop_pixels,
                        })

            # Build standalone crop tasks
            for i, entry in enumerate(ruler_index):
                src_stem = os.path.splitext(os.path.basename(entry["image_path"]))[0]
                crop_name = f"crop_{src_stem}_{i}"
                crop_tasks.append({
                    "img_path": entry["image_path"],
                    "bbox": entry["bbox"],
                    "out_img_path": os.path.join(out_img_dir, crop_name + ".jpg"),
                    "out_lbl_path": os.path.join(out_lbl_dir, crop_name + ".txt"),
                    "min_pixels": self.min_crop_pixels,
                })

            print(f"  Tasks: {len(fp_tasks)} FieldPrism, {len(nfp_tasks)} non-FieldPrism, "
                  f"{len(crop_tasks)} standalone crops")

            # ── dispatch to process pool ──────────────────────────────────
            fp_ruler_count = 0
            nfp_total_pasted = 0
            crop_success = 0

            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                # Submit all tasks
                futures = {}

                for task in fp_tasks:
                    f = pool.submit(_worker_process_fieldprism, task)
                    futures[f] = "fp"

                for task in crop_tasks:
                    f = pool.submit(_worker_extract_crop, task)
                    futures[f] = "crop"

                for task in nfp_tasks:
                    f = pool.submit(_worker_process_non_fieldprism, task)
                    futures[f] = "nfp"

                total = len(futures)
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    kind = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  ERROR ({kind}): {e}")
                        continue

                    if kind == "fp" and result.get("has_rulers"):
                        fp_ruler_count += 1
                    elif kind == "crop" and result.get("success"):
                        crop_success += 1
                    elif kind == "nfp":
                        nfp_total_pasted += result.get("n_pasted", 0)

                    if done_count % 200 == 0 or done_count == total:
                        print(f"  [{split}] {done_count}/{total} tasks complete")

            print(f"  Split {split} done: "
                  f"{len(fp_tasks)} FieldPrism ({fp_ruler_count} with rulers), "
                  f"{len(nfp_tasks)} non-FieldPrism ({nfp_total_pasted} rulers pasted), "
                  f"{crop_success}/{len(crop_tasks)} standalone crops saved")

        self._write_data_yaml()
        print(f"\nDataset generation complete: {self.output_dir}")

    # ── data.yaml ─────────────────────────────────────────────────────────

    def _write_data_yaml(self):
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        content = (
            f"path: {self.output_dir}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 1\n"
            "names:\n"
            "- ruler\n"
        )
        with open(yaml_path, "w") as f:
            f.write(content)
        print(f"  Wrote {yaml_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate FieldPrism ruler training dataset from PREP_final",
    )
    parser.add_argument(
        "--source",
        default="/home/brlab/Dropbox/TextCollage/datasets/PREP_final/",
        help="Path to PREP_final dataset root",
    )
    parser.add_argument(
        "--output",
        default="/datac/FieldPrism_YOLO_Training_Data/",
        help="Output dataset directory",
    )
    parser.add_argument("--min-rulers", type=int, default=1,
                        help="Min rulers to paste per non-FieldPrism image")
    parser.add_argument("--max-rulers", type=int, default=10,
                        help="Max rulers to paste per non-FieldPrism image")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: cpu_count)")

    args = parser.parse_args()

    generator = FieldPrismDatasetGenerator(
        source_dir=args.source,
        output_dir=args.output,
        min_rulers=args.min_rulers,
        max_rulers=args.max_rulers,
        seed=args.seed,
        workers=args.workers,
    )
    generator.generate()


if __name__ == "__main__":
    main()
