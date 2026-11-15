"""
Build a "hard" test subset from the MCS1K test split.

Hardness is computed ONLY from the image and its ground-truth mask, never from
any model's predictions, so no method gets an unfair advantage on the hard set.

Three signals (each converted to a percentile rank, then averaged):
  1. Small object      -> object area as a fraction of the image (smaller = harder)
  2. Low contrast      -> Bhattacharyya distance between the Lab color histogram of
                          the object and of a ring of local background around it
                          (lower distance = object blends in = harder)
  3. Complex boundary  -> perimeter^2 / (4*pi*area), the isoperimetric ratio
                          (higher = more irregular outline = harder)

Outputs (under --data_root):
  MCS1K_hard/{Imgs,GT}   the N hardest images  -> REPORT on this
  MCS1K_rest/{Imgs,GT}   everything else       -> use for TUNING thresholds/blends
  hardness.csv           per-image features, ranks, and hardness score

Usage (from the repo folder in Colab):
  !python build_hard_set.py --n 70
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd


def object_features(img_bgr, mask):
    """Return (area_frac, contrast, complexity, n_components) or None if unusable."""
    h, w = mask.shape
    obj = mask > 127
    area = int(obj.sum())
    if area == 0 or area == h * w:
        return None

    area_frac = area / float(h * w)

    # Boundary complexity: perimeter^2 / (4*pi*area). A circle is 1.0.
    contours, _ = cv2.findContours(
        obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    complexity = (perimeter ** 2) / (4.0 * np.pi * area)
    n_components = len(contours)

    # Local background ring: dilate the mask by a width that scales with object size
    r = int(0.5 * np.sqrt(area))
    r = max(5, min(r, min(h, w) // 4))
    # Distance transform instead of a huge dilation kernel: same ring, fast at any size
    dist = cv2.distanceTransform((~obj).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > 0) & (dist <= r)
    if ring.sum() < 50:  # object fills almost the whole frame
        ring = ~obj

    # Color contrast: Bhattacharyya distance between Lab histograms (0 = identical)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    bins = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    h_obj = cv2.calcHist([lab], [0, 1, 2], obj.astype(np.uint8), bins, ranges)
    h_ring = cv2.calcHist([lab], [0, 1, 2], ring.astype(np.uint8), bins, ranges)
    cv2.normalize(h_obj, h_obj, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(h_ring, h_ring, 1.0, 0.0, cv2.NORM_L1)
    contrast = cv2.compareHist(h_obj, h_ring, cv2.HISTCMP_BHATTACHARYYA)

    return area_frac, contrast, complexity, n_components


def copy_subset(stems, img_dir, gt_dir, img_ext_map, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # derived folder only; rebuilt from scratch each run
    os.makedirs(os.path.join(out_dir, "Imgs"))
    os.makedirs(os.path.join(out_dir, "GT"))
    for s in stems:
        shutil.copy(os.path.join(img_dir, img_ext_map[s]), os.path.join(out_dir, "Imgs"))
        shutil.copy(os.path.join(gt_dir, s + ".png"), os.path.join(out_dir, "GT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data/TestDataset")
    ap.add_argument("--src_name", default="MCS1K", help="existing test folder with Imgs/ and GT/")
    ap.add_argument("--n", type=int, default=70, help="size of the hard set")
    ap.add_argument("--w_size", type=float, default=1.0, help="weight of the small-object signal")
    ap.add_argument("--w_contrast", type=float, default=1.0, help="weight of the low-contrast signal")
    ap.add_argument("--w_boundary", type=float, default=1.0, help="weight of the boundary-complexity signal")
    args = ap.parse_args()

    src = os.path.join(args.data_root, args.src_name)
    img_dir, gt_dir = os.path.join(src, "Imgs"), os.path.join(src, "GT")

    img_ext_map = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }

    rows, skipped = [], []
    for stem in sorted(img_ext_map):
        gt_path = os.path.join(gt_dir, stem + ".png")
        img = cv2.imread(os.path.join(img_dir, img_ext_map[stem]))
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            skipped.append((stem, "unreadable image or mask"))
            continue
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Score at a fixed scale (long side 512) so features are comparable and fast
        scale = 512.0 / max(img.shape[:2])
        if scale < 1.0:
            new_wh = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, new_wh, interpolation=cv2.INTER_NEAREST)
        feats = object_features(img, mask)
        if feats is None:
            skipped.append((stem, "empty or full mask"))
            continue
        rows.append((stem,) + feats)

    df = pd.DataFrame(rows, columns=["stem", "area_frac", "contrast", "complexity", "n_components"])

    # Percentile ranks in [0, 1]; flip so that 1 always means "harder"
    df["r_small"] = 1.0 - df["area_frac"].rank(pct=True)
    df["r_lowcontrast"] = 1.0 - df["contrast"].rank(pct=True)
    df["r_complex"] = df["complexity"].rank(pct=True)
    wsum = args.w_size + args.w_contrast + args.w_boundary
    df["hardness"] = (
        args.w_size * df["r_small"]
        + args.w_contrast * df["r_lowcontrast"]
        + args.w_boundary * df["r_complex"]
    ) / wsum

    df = df.sort_values("hardness", ascending=False).reset_index(drop=True)
    df["split"] = np.where(df.index < args.n, "hard", "rest")
    df.to_csv("hardness.csv", index=False)

    hard = df[df["split"] == "hard"]["stem"].tolist()
    rest = df[df["split"] == "rest"]["stem"].tolist()
    copy_subset(hard, img_dir, gt_dir, img_ext_map, os.path.join(args.data_root, args.src_name + "_hard"))
    copy_subset(rest, img_dir, gt_dir, img_ext_map, os.path.join(args.data_root, args.src_name + "_rest"))

    # ---- Report ----
    print(f"Scored {len(df)} images, skipped {len(skipped)}: {skipped[:5]}")
    print(f"Hard set: {len(hard)} images  |  Rest (tuning): {len(rest)} images\n")
    cols = ["area_frac", "contrast", "complexity", "n_components", "hardness"]
    print("Feature means (hard vs rest):")
    print(df.groupby("split")[cols].mean().round(3).to_string(), "\n")
    print("Correlation between the three hardness signals (high = they overlap a lot):")
    print(df[["r_small", "r_lowcontrast", "r_complex"]].corr().round(2).to_string(), "\n")
    print("10 hardest images:")
    print(df.head(10)[["stem"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
