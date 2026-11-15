"""
Download the MCS1K dataset from Kaggle and arrange it the way this repo expects.

Source dataset: https://www.kaggle.com/datasets/aalihhiader/military-camouflage-soldiers-dataset-mcs1k
(it ships as  <something>/Training/{images,GT}  and  <something>/Testing/{images,GT})

Produces, under --out (default ./data):
  TrainDataset/Imgs   training photos (.jpg)
  TrainDataset/GT     binary masks (.png, values 0/255)   <- the Kaggle masks are .jpg, so they are converted
  TrainDataset/Edge   boundary maps (.png) made from the masks (needed by BGNet training)
  TestDataset/MCS1K/Imgs, GT
  ValDataset/{Imgs,GT,Edge}   only if --val_frac > 0 (random, seeded hold-out taken from the training images)

Kaggle credentials are needed for the download step: put kaggle.json in ~/.kaggle/ or set the
KAGGLE_USERNAME and KAGGLE_KEY environment variables. Skip the download with --raw_dir pointing
at an already-unzipped copy.

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --val_frac 0.15        # also carve out a validation split
"""
import argparse
import os
import random
import shutil
import subprocess
import sys

import cv2

DATASET = "aalihhiader/military-camouflage-soldiers-dataset-mcs1k"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def download(dataset, raw_dir):
    if shutil.which("kaggle") is None:
        sys.exit("The `kaggle` command was not found. Install it with `pip install kaggle`, "
                 "or download the dataset by hand and pass --raw_dir.")
    os.makedirs(raw_dir, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", raw_dir, "--unzip"], check=True)


def find_split_root(raw_dir):
    """Find the folder that contains both Training/ and Testing/ (its name varies)."""
    for root, dirs, _ in os.walk(raw_dir):
        if "Training" in dirs and "Testing" in dirs:
            return root
    sys.exit(f"No folder containing both Training/ and Testing/ was found under {raw_dir}.")


def pairs(split_dir):
    """Return sorted (stem, image_path, mask_path) for a split folder with images/ and GT/."""
    img_dir, gt_dir = os.path.join(split_dir, "images"), os.path.join(split_dir, "GT")
    imgs = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
            for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)}
    gts = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
           for f in os.listdir(gt_dir) if f.lower().endswith(IMG_EXTS)}
    missing_mask, missing_img = sorted(set(imgs) - set(gts)), sorted(set(gts) - set(imgs))
    if missing_mask or missing_img:
        print(f"  warning: {len(missing_mask)} images without a mask and {len(missing_img)} masks "
              f"without an image in {split_dir}; those are skipped.")
    return [(s, imgs[s], gts[s]) for s in sorted(set(imgs) & set(gts))]


def write_pair(stem, img_path, gt_path, out_dir, edge_kernel=None):
    """Write image as .jpg, mask as binary .png, and (optionally) an edge map as .png."""
    img = cv2.imread(img_path)
    mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"  warning: could not read {stem}, skipped")
        return False
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # JPEG masks have gray fuzz at the edges
    cv2.imwrite(os.path.join(out_dir, "Imgs", stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir, "GT", stem + ".png"), mask)
    if edge_kernel:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_kernel, edge_kernel))
        cv2.imwrite(os.path.join(out_dir, "Edge", stem + ".png"), cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k))
    return True


def build(items, out_dir, subfolders, edge_kernel=None):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # derived data only; rebuilt from the raw download every time
    for sub in subfolders:
        os.makedirs(os.path.join(out_dir, sub))
    ok = sum(write_pair(s, i, g, out_dir, edge_kernel) for s, i, g in items)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--raw_dir", default="./data/raw", help="where the Kaggle download is unzipped")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--skip_download", action="store_true", help="use what is already in --raw_dir")
    ap.add_argument("--edge_kernel", type=int, default=5, help="size of the morphological-gradient kernel for edge maps")
    ap.add_argument("--val_frac", type=float, default=0.0, help="fraction of training images held out as a validation set")
    ap.add_argument("--seed", type=int, default=0, help="seed for the validation split")
    args = ap.parse_args()

    if not args.skip_download:
        download(args.dataset, args.raw_dir)
    split_root = find_split_root(args.raw_dir)
    print(f"Using raw data in {split_root}")

    train_items = pairs(os.path.join(split_root, "Training"))
    test_items = pairs(os.path.join(split_root, "Testing"))

    val_items = []
    if args.val_frac > 0:
        rng = random.Random(args.seed)
        shuffled = train_items[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(args.val_frac * len(shuffled)))
        val_items = sorted(shuffled[:n_val])
        val_stems = {s for s, _, _ in val_items}
        train_items = [t for t in train_items if t[0] not in val_stems]

    out = args.out
    n_train = build(train_items, os.path.join(out, "TrainDataset"), ["Imgs", "GT", "Edge"], args.edge_kernel)
    n_test = build(test_items, os.path.join(out, "TestDataset", "MCS1K"), ["Imgs", "GT"])
    n_val = build(val_items, os.path.join(out, "ValDataset"), ["Imgs", "GT", "Edge"], args.edge_kernel) if val_items else 0

    print("\nDone.")
    print(f"  {out}/TrainDataset        {n_train} images + masks + edge maps")
    if val_items:
        print(f"  {out}/ValDataset          {n_val} images + masks + edge maps (seed {args.seed})")
    print(f"  {out}/TestDataset/MCS1K   {n_test} images + masks")


if __name__ == "__main__":
    main()
