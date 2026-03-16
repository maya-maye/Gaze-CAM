"""
generate_teaser.py  –  Create the paper teaser figure.

Layout: wide, multi-clip figure.
  Columns = 4 different action clips (different verbs/recipes)
  Rows    = Input+Gaze, R3D-18, SlowFast R50, TimeSformer, ViViT
  Each cell is a single representative frame from that clip.

Usage:
    python scripts/generate_teaser.py [--split 1] [--num_clips 4] [--max_test 500]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from gaze_cam.config import (
    DEVICE, OUTPUT_DIR, SPLIT,
    CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
    weights_path,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model
from gaze_cam.gradcam import build_cam_engine
from gaze_cam.gaze_utils import (
    load_gaze_file, slice_gaze_for_clip, nearest_valid_xy,
)

MODELS = ["r3d18", "slowfast_r50", "timesformer", "vivit"]
MODEL_LABELS = {
    "r3d18": "R3D-18",
    "slowfast_r50": "SlowFast R50",
    "timesformer": "TimeSformer",
    "vivit": "ViViT",
}

# Try to pick clips with these verbs for visual variety
PREFERRED_VERBS = ["Cut", "Stir", "Take", "Pour", "Open", "Put", "Mix", "Wash", "Close", "Crack", "Spread", "Scoop"]


def decode_clip_frames(clip_path):
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_cam_heatmap(cam_2d, h, w):
    cam_u8 = (cam_2d * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_u8, (w, h))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def blend(frame_rgb, heatmap_rgb, alpha=0.45):
    out = frame_rgb.astype(float) * (1 - alpha) + \
          heatmap_rgb.astype(float) * alpha
    return out.clip(0, 255).astype(np.uint8)


def draw_gaze_dot(frame_rgb, gx, gy, radius=10,
                  color=(0, 255, 0), thickness=3):
    out = frame_rgb.copy()
    cv2.circle(out, (int(gx), int(gy)), radius, color, thickness)
    cv2.circle(out, (int(gx), int(gy)), 3, color, -1)
    return out


def get_gaze_at_middle(gaze_df, f0, mid_fi, h, w):
    """Return (gx_px, gy_px) at a specific frame index, or None."""
    g_clip = slice_gaze_for_clip(gaze_df, f0, f0 + mid_fi + 1)
    session_f = f0 + mid_fi
    gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
    gi_local = gi - int(round(f0 * (GAZE_HZ / VIDEO_FPS)))
    gi_local = int(np.clip(gi_local, 0, max(0, len(g_clip) - 1)))
    if len(g_clip) == 0:
        return None
    gx, gy, _ = nearest_valid_xy(g_clip, gi_local, max_radius=30)
    if gx is None:
        return None
    gx_px = np.clip(gx * (w / CALIBRATION_W), 0, w - 1)
    gy_px = np.clip(gy * (h / CALIBRATION_H), 0, h - 1)
    return gx_px, gy_px


def find_diverse_clips(test_loader, num_clips=4):
    """Find clips with different verbs and valid gaze. Return list of dicts."""
    seen_verbs = set()
    # Priority: preferred verbs first, then any
    candidates = []
    fallbacks = []

    for x_batch, y_batch, paths, metas in test_loader:
        for i in range(len(paths)):
            meta = metas[i]
            session = meta["session"]
            try:
                gaze_df = load_gaze_file(session)
            except FileNotFoundError:
                continue
            g_clip = slice_gaze_for_clip(gaze_df, meta["f0"], meta["f1"])
            if len(g_clip) < 10:
                continue

            # Extract verb from action label
            stem = meta["stem"]
            gt_id = int(y_batch[i].item())
            entry = {
                "clip_path": Path(paths[i]),
                "meta": meta,
                "gt_id": gt_id,
                "gaze_df": gaze_df,
                "stem": stem,
            }

            # Parse verb from stem (format: SESSION-start-end-ACTION)
            # or we'll get it from id_to_label later
            fallbacks.append(entry)

    return fallbacks[:num_clips * 3]  # return extras for filtering later


def compute_cam_single(arch, model, meta, test_loader):
    """Run a single clip through model, return CAM (T,H,W) and pred_id."""
    gcam = build_cam_engine(arch, model)
    model.eval()
    target_stem = meta["stem"]
    for x_batch, y_batch, paths, metas in test_loader:
        for i in range(len(paths)):
            if metas[i]["stem"] != target_stem:
                continue
            if isinstance(x_batch, (list, tuple)):
                x_in = [t[i:i+1].to(DEVICE) for t in x_batch]
            else:
                x_in = x_batch[i:i+1].to(DEVICE)
            cam_bthw, logits, _ = gcam.compute(x_in, class_idx=None)
            pred_id = int(logits.argmax(1).cpu().item())
            gcam.close()
            return cam_bthw[0], pred_id
    gcam.close()
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Generate paper teaser figure")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--num_clips", type=int, default=8,
                        help="Number of different clips to show as columns")
    parser.add_argument("--max_test", type=int, default=800)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Generating teaser: {args.num_clips} clips × {len(MODELS)} models\n")

    # Step 1: Find diverse clips using r3d18 loader (smallest)
    print("Finding diverse clips...")
    _, test_loader_find, label_to_id, id_to_label, na = make_loaders(
        split=args.split, batch_size=8, arch="r3d18",
        max_train=2, max_test=args.max_test,
    )

    raw_candidates = find_diverse_clips(test_loader_find, args.num_clips)
    if not raw_candidates:
        print("ERROR: No suitable clips found!")
        return

    # Pick clips with different verbs for variety
    selected = []
    seen_verbs = set()
    for c in raw_candidates:
        gt_label = id_to_label.get(c["gt_id"], "")
        verb = gt_label.split(" ")[0] if gt_label else ""
        if verb in seen_verbs:
            continue
        # Prefer verbs from our preferred list
        if verb in PREFERRED_VERBS or len(selected) < args.num_clips:
            seen_verbs.add(verb)
            c["gt_label"] = gt_label
            c["verb"] = verb
            selected.append(c)
        if len(selected) >= args.num_clips:
            break

    # If we still don't have enough, grab any remaining
    if len(selected) < args.num_clips:
        for c in raw_candidates:
            if c in selected:
                continue
            gt_label = id_to_label.get(c["gt_id"], "")
            c["gt_label"] = gt_label
            c["verb"] = gt_label.split(" ")[0] if gt_label else ""
            selected.append(c)
            if len(selected) >= args.num_clips:
                break

    num_clips = len(selected)
    print(f"Selected {num_clips} clips:")
    for c in selected:
        print(f"  {c['gt_label']:30s}  ({c['meta']['session']})")

    # Step 2: Decode middle frames and gaze for each clip
    clip_data = []  # list of {frame, gaze_pt, gt_label, meta, h, w, ...}
    for c in selected:
        frames = decode_clip_frames(c["clip_path"])
        if not frames:
            continue
        h, w = frames[0].shape[:2]
        mid = len(frames) // 2
        frame = frames[mid]
        gaze_pt = get_gaze_at_middle(c["gaze_df"], c["meta"]["f0"], mid, h, w)
        clip_data.append({
            "frame": frame,
            "gaze_pt": gaze_pt,
            "gt_label": c["gt_label"],
            "meta": c["meta"],
            "gaze_df": c["gaze_df"],
            "h": h, "w": w,
            "mid_fi": mid,
            "n_frames": len(frames),
        })

    num_clips = len(clip_data)
    if num_clips == 0:
        print("ERROR: Could not decode any clip frames!")
        return

    # Step 3: For each model, compute CAMs for all selected clips
    # cam_grid[arch][clip_idx] = (cam_2d_at_mid, pred_label)
    cam_grid = {arch: [None] * num_clips for arch in MODELS}

    for arch in MODELS:
        print(f"\nProcessing {MODEL_LABELS[arch]}...")
        _, test_loader, _, id_to_label_a, na_a = make_loaders(
            split=args.split, batch_size=8, arch=arch,
            max_train=2, max_test=args.max_test,
        )
        model = build_model(arch, na_a)
        ckpt_path = weights_path(args.split, arch)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()

        for ci, cd in enumerate(clip_data):
            cam_thw, pred_id = compute_cam_single(
                arch, model, cd["meta"], test_loader
            )
            if cam_thw is not None:
                # Upsample and pick the middle frame
                T_vid = cd["n_frames"]
                h, w = cd["h"], cd["w"]
                cam_t = cam_thw.unsqueeze(0).unsqueeze(0)
                cam_up = F.interpolate(
                    cam_t, size=(T_vid, h, w),
                    mode="trilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                cam_mid = cam_up[cd["mid_fi"]]
                pred_label = id_to_label_a.get(pred_id, str(pred_id))
                cam_grid[arch][ci] = (cam_mid, pred_label)
                correct = "correct" if pred_label == cd["gt_label"] else "wrong"
                print(f"  Clip {ci}: pred={pred_label} ({correct})")

        del model
        torch.cuda.empty_cache()

    # Step 4: Build the figure — wide layout
    n_rows = 1 + len(MODELS)  # Input row + 4 model rows
    n_cols = num_clips

    fig_w = 2.4 * n_cols + 1.0  # wide rectangular layout
    fig_h = 2.0 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(n_rows, n_cols, figure=fig,
                  hspace=0.08, wspace=0.04,
                  left=0.10, right=0.98, top=0.92, bottom=0.01)

    # Row 0: Input frames + green gaze dot
    for ci, cd in enumerate(clip_data):
        ax = fig.add_subplot(gs[0, ci])
        panel = cd["frame"].copy()
        if cd["gaze_pt"] is not None:
            panel = draw_gaze_dot(panel, cd["gaze_pt"][0], cd["gaze_pt"][1],
                                  radius=14, color=(0, 255, 0), thickness=3)
        ax.imshow(panel)
        ax.axis("off")
        # Action label as column title
        ax.set_title(cd["gt_label"], fontsize=9, fontweight="bold", pad=6)
        if ci == 0:
            ax.text(-0.12, 0.5, "Input + Gaze", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", rotation=90,
                    va="center", ha="right", color="black")

    # Rows 1–4: model CAM overlays
    for ri, arch in enumerate(MODELS):
        row = ri + 1
        for ci, cd in enumerate(clip_data):
            ax = fig.add_subplot(gs[row, ci])
            frame = cd["frame"]
            h, w = cd["h"], cd["w"]

            if cam_grid[arch][ci] is not None:
                cam_mid, pred_label = cam_grid[arch][ci]
                heatmap = make_cam_heatmap(cam_mid, h, w)
                panel = blend(frame, heatmap, alpha=0.45)
            else:
                panel = frame.copy()
                pred_label = "N/A"

            if cd["gaze_pt"] is not None:
                panel = draw_gaze_dot(panel, cd["gaze_pt"][0], cd["gaze_pt"][1],
                                      radius=14, color=(0, 255, 0), thickness=3)
            ax.imshow(panel)
            ax.axis("off")

            # Prediction annotation in bottom-right
            correct = pred_label == cd["gt_label"]
            sym = "\u2713" if correct else "\u2717"
            clr = "#2ecc71" if correct else "#e74c3c"
            ax.text(0.97, 0.04, f"{sym}", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color=clr,
                    ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="black",
                              alpha=0.6, ec="none"))

            if ci == 0:
                ax.text(-0.12, 0.5, MODEL_LABELS[arch],
                        transform=ax.transAxes,
                        fontsize=9, fontweight="bold", rotation=90,
                        va="center", ha="right",
                        color="black")

    # Legend for gaze dot
    fig.text(0.5, 0.96,
             "Green circle = human gaze    |    "
             "Heatmap = model saliency    |    "
             "\u2713 correct    \u2717 incorrect",
             fontsize=8, ha="center", va="bottom", color="#555555")

    out_path = out_dir / "fig_teaser.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\nTeaser figure saved: {out_path}")

    # Copy to docs/writeup/figures/
    writeup_fig = Path("docs/writeup/figures/fig_teaser.png")
    writeup_fig.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(out_path), str(writeup_fig))
    print(f"Copied to: {writeup_fig}")


if __name__ == "__main__":
    main()
