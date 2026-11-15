"""
Gradio demo: BGNet camouflaged-object segmentation with the test-time strategies
evaluated in validate.py. Upload an image (or click an example), pick a method, and
see the predicted mask, an overlay, and a confidence heatmap.

Run in Colab (from the repo folder, after uploading this file next to validate.py):
  !pip install -q gradio
  !python app.py --ckpt checkpoints/mcs1k_ft/BGNet-19.pth --pretrained checkpoints/BGNet.pth

It prints a public link (…gradio.live). The cell keeps running while the demo is up,
and it holds your GPU session, so STOP THE CELL when you are done.
"""
import argparse
import os
import time

import cv2
import numpy as np

import validate as V

MAX_SIDE = 1280  # cap very large uploads: the model works at 416 px anyway

# label shown in the dropdown -> (function, number of model passes)
METHOD_CHOICES = {
    "Plain BGNet (1 pass)": (V.m_plain, 1),
    "Flip TTA (2 passes)": (V.m_tta_flip, 2),
    "Multi-scale TTA (6 passes): best S-measure in our tests": (V.m_tta_ms, 6),
    "Multi-scale TTA + cleanup (6 passes)": (V.METHODS["ft+tta_ms+cleanup"], 6),
    "Adaptive threshold, binary output (1 pass)": (V.METHODS["ft+adaptive_thr(binary)"], 1),
}
DEFAULT_METHOD = "Multi-scale TTA (6 passes): best S-measure in our tests"


def make_outputs(img_rgb, prob, thr):
    """img_rgb uint8 HxWx3, prob float HxW in [0,1] -> (mask, overlay, heatmap, coverage)."""
    binm = prob >= thr
    mask = np.repeat((binm.astype(np.uint8) * 255)[:, :, None], 3, axis=2)

    overlay = img_rgb.astype(np.float32).copy()
    red = np.array([255, 40, 40], np.float32)
    overlay[binm] = 0.55 * overlay[binm] + 0.45 * red
    overlay = overlay.astype(np.uint8)
    cnts, _ = cv2.findContours(binm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (255, 255, 0), max(2, img_rgb.shape[1] // 300))

    heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return mask, overlay, np.ascontiguousarray(heat), float(binm.mean())


def build_infer(runners):
    """runners: dict label -> BGNetRunner. Returns the function Gradio calls."""

    def infer(img, method, model_name, thr):
        if img is None:
            return None, None, None, "Upload an image or pick an example first."
        img = np.ascontiguousarray(img[:, :, :3]).astype(np.uint8)
        scale = MAX_SIDE / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        fn, passes = METHOD_CHOICES[method]
        runner = runners[model_name]
        t0 = time.time()
        prob = fn({"runner": runner}, img)
        dt = time.time() - t0
        mask, overlay, heat, cover = make_outputs(img, prob, thr)

        note = ""
        if np.unique(prob).size <= 2:
            note = "  (this method outputs a hard black-and-white mask, so the heatmap has no gradient)"
        msg = (f"Model: {model_name}  |  Method: {method}\n"
               f"Detected region covers {cover * 100:.1f}% of the image  |  {passes} model pass(es), {dt:.2f}s{note}")
        return mask, overlay, heat, msg

    return infer


def find_examples(folder, hardness_csv="hardness.csv", n=4):
    """Pick the hardest images from the hard set if hardness.csv exists, else the first few."""
    if not os.path.isdir(os.path.join(folder, "Imgs")):
        return []
    files = {os.path.splitext(f)[0]: os.path.join(folder, "Imgs", f)
             for f in sorted(os.listdir(os.path.join(folder, "Imgs")))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))}
    stems = list(files)
    if os.path.exists(hardness_csv):
        try:
            import pandas as pd
            order = pd.read_csv(hardness_csv).sort_values("hardness", ascending=False)["stem"].tolist()
            stems = [s for s in order if s in files] or stems
        except Exception:
            pass
    return [[files[s]] for s in stems[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint")
    ap.add_argument("--pretrained", default=None, help="authors' checkpoint (adds a model toggle)")
    ap.add_argument("--examples", default="./data/TestDataset/MCS1K_hard")
    ap.add_argument("--no_share", action="store_true", help="local link only (default creates a public link)")
    args = ap.parse_args()

    import gradio as gr

    runners = {"Fine-tuned on MCS1K (ours)": V.BGNetRunner(args.ckpt)}
    if args.pretrained:
        runners["Pretrained (authors' weights)"] = V.BGNetRunner(args.pretrained)
    infer = build_infer(runners)

    kwargs = dict(
        fn=infer,
        inputs=[
            gr.Image(type="numpy", label="Input image"),
            gr.Dropdown(list(METHOD_CHOICES), value=DEFAULT_METHOD, label="Method"),
            gr.Dropdown(list(runners), value=list(runners)[0], label="Model weights"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Mask threshold"),
        ],
        outputs=[
            gr.Image(label="Predicted mask"),
            gr.Image(label="Overlay (red = detected, yellow = outline)"),
            gr.Image(label="Confidence heatmap (red = more confident)"),
            gr.Textbox(label="Details", lines=3),
        ],
        examples=[ex + [DEFAULT_METHOD, list(runners)[0], 0.5] for ex in find_examples(args.examples)] or None,
        title="Camouflaged soldier segmentation (BGNet)",
        description=("BGNet fine-tuned on MCS1K. Pick a test-time strategy to see how it changes the prediction. "
                     "The method list matches the experiments in our validation table."),
    )
    try:  # newer Gradio
        demo = gr.Interface(**kwargs, flagging_mode="never")
    except TypeError:  # older Gradio
        demo = gr.Interface(**kwargs, allow_flagging="never")
    demo.launch(share=not args.no_share, debug=False)


if __name__ == "__main__":
    main()
