"""
Validation harness: compare BGNet variants on the hard test set (and the rest).

Every method is a function  method(ctx, img_rgb) -> float32 probability map in [0,1]
at the ORIGINAL image size. To add a method, write one function and register it in
METHODS. The harness handles prediction, scoring and the comparison table.

Rows in the table:
  pretrained            authors' weights, plain inference (reference)
  finetuned             your BGNet-19, plain inference     (reference; must match
                        the numbers you got from etest.py/eval.py -> that is the
                        harness self-check)
  ft+tta_flip, ...      Method A variants on top of the fine-tuned model

Scoring: S-measure, weighted F, mean F, mean E, MAE (py_sod_metrics, same library
as eval.py) plus boundary F-score. Each method also gets, versus the fine-tuned
plain row: how many images improved and a paired-bootstrap 95% CI on the mean
S-measure difference, because the hard set is only ~70 images.

Usage (from the repo folder in Colab):
  !python validate.py --ckpt checkpoints/mcs1k_ft/BGNet-19.pth \
        --pretrained checkpoints/BGNet.pth --subsets MCS1K_hard MCS1K_rest
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

IMG_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Model wrapper (torch is imported lazily so the pure functions stay testable)
# --------------------------------------------------------------------------
class BGNetRunner:
    def __init__(self, ckpt, device=None):
        import torch
        from net.bgnet import Net

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def predict(self, img_rgb, size=416, flip=False):
        """One forward pass -> prob map at the original resolution, min-max normalised
        exactly as etest.py does (sigmoid, then (x-min)/(max-min))."""
        torch = self.torch
        h, w = img_rgb.shape[:2]
        # PIL bilinear (antialiased), matching torchvision transforms.Resize on a PIL image,
        # which is what the repo's test loader most likely uses. cv2.INTER_LINEAR aliases
        # when shrinking large photos and shifts every score slightly.
        from PIL import Image
        x = np.asarray(Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR))
        x = (x.astype(np.float32) / 255.0 - self.mean) / self.std
        if flip:
            x = x[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            res = self.model(x)[2]
            res = torch.nn.functional.interpolate(res, size=(h, w), mode="bilinear", align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
        if flip:
            res = res[:, ::-1]
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        return np.ascontiguousarray(res.astype(np.float32))


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
def m_plain(ctx, img):
    return ctx["runner"].predict(img, 416)


def m_tta_flip(ctx, img):
    r = ctx["runner"]
    return (r.predict(img, 416) + r.predict(img, 416, flip=True)) / 2


def m_tta_ms(ctx, img):
    r = ctx["runner"]
    outs = [r.predict(img, s, flip=f) for s in (352, 416, 480) for f in (False, True)]
    return np.mean(outs, axis=0)


def clean_prob(prob, min_frac=0.01, thr=0.5, hole_frac=0.01):
    """Soft morphological cleanup. Removes small blobs (zeroed) and fills small holes
    (set to 1), but leaves the rest of the soft map untouched so S/E-measure are not
    destroyed by hard binarisation."""
    h, w = prob.shape
    binm = (prob > thr).astype(np.uint8)
    out = prob.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        # drop only blobs that are tiny in absolute terms AND small next to the main object;
        # the largest component is always kept, even if the object itself is tiny
        keep_floor = min(max(min_frac * h * w, 0.1 * areas.max()), areas.max())
        for i, a in enumerate(areas, start=1):
            if a < keep_floor:
                out[labels == i] = 0.0
    kept = (out > thr)
    filled = ndi.binary_fill_holes(kept)
    holes = filled & ~kept
    hl, hn = ndi.label(holes)
    for i in range(1, hn + 1):
        region = hl == i
        if region.sum() <= hole_frac * h * w:
            out[region] = 1.0
    return out


def adaptive_binary(prob):
    """Classic adaptive threshold: 2 x mean of the map, clipped. Returns a BINARY map."""
    t = float(np.clip(2.0 * prob.mean(), 0.1, 0.9))
    return (prob >= t).astype(np.float32)


def make_post(base_fn, post_fn):
    return lambda ctx, img: post_fn(base_fn(ctx, img))


def crf_refine(prob, img, iters=5):
    from pydensecrf import densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = prob.shape
    p = np.clip(prob, 1e-5, 1 - 1e-5)
    soft = np.stack([1 - p, p]).astype(np.float32)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(np.ascontiguousarray(unary_from_softmax(soft)))
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=10, rgbim=np.ascontiguousarray(img), compat=5)
    q = np.array(d.inference(iters)).reshape(2, h, w)
    return q[1].astype(np.float32)


METHODS = {
    # name: (uses_finetuned_model, function)
    "ft+tta_flip": m_tta_flip,
    "ft+tta_ms": m_tta_ms,
    "ft+cleanup": make_post(m_plain, clean_prob),
    "ft+tta_ms+cleanup": make_post(m_tta_ms, clean_prob),
    "ft+adaptive_thr(binary)": make_post(m_plain, adaptive_binary),
    "ft+tta_ms+adaptive_thr(binary)": make_post(m_tta_ms, adaptive_binary),
}


def add_crf_methods(enabled):
    if not enabled:
        return
    try:
        import pydensecrf  # noqa: F401
    except Exception:
        print("[info] pydensecrf not installed -> skipping CRF methods "
              "(Colab: !pip install git+https://github.com/lucasb-eyer/pydensecrf.git)")
        return
    METHODS["ft+crf"] = lambda ctx, img: crf_refine(m_plain(ctx, img), img)
    METHODS["ft+tta_ms+crf"] = lambda ctx, img: crf_refine(m_tta_ms(ctx, img), img)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def boundary_f(prob, gt_bin, thr=0.5, tol_frac=0.005):
    """Boundary F-score: precision/recall of predicted vs GT boundary pixels within a
    distance tolerance (0.5% of the image diagonal, min 2 px)."""
    h, w = gt_bin.shape
    pred = prob > thr
    tol = max(2.0, tol_frac * np.hypot(h, w))

    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)

    pe, ge = edge(pred), edge(gt_bin)
    if ge.sum() == 0 and pe.sum() == 0:
        return 1.0
    if ge.sum() == 0 or pe.sum() == 0:
        return 0.0
    d_to_g = ndi.distance_transform_edt(~ge)
    d_to_p = ndi.distance_transform_edt(~pe)
    precision = (d_to_g[pe] <= tol).mean()
    recall = (d_to_p[ge] <= tol).mean()
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class Scorer:
    """Aggregate metrics with py_sod_metrics + per-image S-measure and boundary F."""

    def __init__(self):
        from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

        self._cls = (MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure)
        MAE_, EM, FM, SM, WFM = self._cls
        self.mae, self.em, self.fm, self.sm, self.wfm = MAE_(), EM(), FM(), SM(), WFM()
        self.per_image = []

    def step(self, name, prob, gt_u8):
        pred_u8 = (np.clip(prob, 0, 1) * 255).round().astype(np.uint8)
        for m in (self.mae, self.em, self.fm, self.sm, self.wfm):
            m.step(pred=pred_u8, gt=gt_u8)
        one = self._cls[3]()
        one.step(pred=pred_u8, gt=gt_u8)
        self.per_image.append({
            "image": name,
            "Sm": float(one.get_results()["sm"]),
            "BF": boundary_f(prob, gt_u8 > 127),
        })

    def summary(self):
        fm = self.fm.get_results()["fm"]
        em = self.em.get_results()["em"]
        try:
            mean_fm = float(np.mean(fm["curve"]))
            mean_em = float(np.mean(em["curve"]))
        except Exception:
            print("[warn] could not read F/E curves; keys:", list(fm), list(em))
            mean_fm = mean_em = float("nan")
        df = pd.DataFrame(self.per_image)
        return {
            "Sm": float(self.sm.get_results()["sm"]),
            "wFm": float(self.wfm.get_results()["wfm"]),
            "meanFm": mean_fm,
            "meanEm": mean_em,
            "MAE": float(self.mae.get_results()["mae"]),
            "BoundaryF": float(df["BF"].mean()),
        }


def paired_bootstrap_ci(diff, n=5000, seed=0):
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def list_pairs(root, name):
    img_dir, gt_dir = os.path.join(root, name, "Imgs"), os.path.join(root, name, "GT")
    stems = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    return [(f, os.path.join(img_dir, f), os.path.join(gt_dir, os.path.splitext(f)[0] + ".png")) for f in stems]


def evaluate(name, fn, ctx, pairs, save_dir=None):
    sc = Scorer()
    for i, (fname, ip, gp) in enumerate(pairs):
        img = cv2.imread(ip)[:, :, ::-1].copy()
        gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE)
        if gt.shape != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        prob = fn(ctx, img)
        sc.step(fname, prob, gt)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, os.path.splitext(fname)[0] + ".png"),
                        (np.clip(prob, 0, 1) * 255).astype(np.uint8))
        if (i + 1) % 25 == 0:
            print(f"    {name}: {i + 1}/{len(pairs)}", flush=True)
    return sc.summary(), pd.DataFrame(sc.per_image)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--pretrained", default=None, help="authors' checkpoint, for the reference row")
    ap.add_argument("--root", default="./data/TestDataset")
    ap.add_argument("--subsets", nargs="+", default=["MCS1K_hard", "MCS1K_rest"])
    ap.add_argument("--methods", nargs="*", default=None, help="subset of method names to run")
    ap.add_argument("--crf", action="store_true", help="include CRF methods if pydensecrf is installed")
    ap.add_argument("--save_preds", action="store_true", help="save prediction PNGs under validate_out/")
    ap.add_argument("--out", default="validate_out")
    args = ap.parse_args()

    add_crf_methods(args.crf)
    names = args.methods if args.methods else list(METHODS)
    unknown = [m for m in names if m not in METHODS]
    if unknown:
        sys.exit(f"Unknown methods {unknown}. Available: {list(METHODS)}")

    ft = {"runner": BGNetRunner(args.ckpt)}
    rows, per_image_all = [], []

    for subset in args.subsets:
        pairs = list_pairs(args.root, subset)
        print(f"\n=== {subset}: {len(pairs)} images ===")
        runs = []
        if args.pretrained:
            runs.append(("pretrained", m_plain, {"runner": BGNetRunner(args.pretrained)}))
        runs.append(("finetuned", m_plain, ft))
        runs += [(n, METHODS[n], ft) for n in names]

        base_sm = None
        for name, fn, ctx in runs:
            print(f"  running {name} ...", flush=True)
            sd = os.path.join(args.out, subset, name) if args.save_preds else None
            summ, per = evaluate(name, fn, ctx, pairs, sd)
            per["method"], per["subset"] = name, subset
            per_image_all.append(per)
            if name == "finetuned":
                base_sm = per["Sm"].values
            row = {"subset": subset, "method": name, **summ}
            if base_sm is not None and name not in ("pretrained", "finetuned"):
                d = per["Sm"].values - base_sm
                lo, hi = paired_bootstrap_ci(d)
                row.update({"imgs_better": int((d > 1e-4).sum()), "imgs_worse": int((d < -1e-4).sum()),
                            "dSm": float(d.mean()), "dSm_CI95": f"[{lo:+.4f}, {hi:+.4f}]"})
            rows.append(row)

    table = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
    pd.concat(per_image_all).to_csv(os.path.join(args.out, "per_image_scores.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for subset in args.subsets:
        print(f"\n##### {subset} #####")
        print(table[table.subset == subset].drop(columns="subset").round(4).to_string(index=False))
    print("\nSaved:", os.path.join(args.out, "results_table.csv"), "and per_image_scores.csv")
    print("Self-check: 'finetuned' on MCS1K_hard should read Sm~0.802, wFm~0.701 "
          "(from your etest.py/eval.py run). If it doesn't, the harness inference differs from etest.py.")


if __name__ == "__main__":
    main()
