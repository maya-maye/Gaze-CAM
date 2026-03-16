"""
Generate a teaser figure for the paper: a grid showing
  Row per model  x  Columns: Raw Frame | Grad-CAM Overlay | CAM + Gaze
All four models on the SAME clip so the reader can directly compare.

Usage:
    python scripts/make_teaser.py [--clip-index 5] [--split 1]
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
    SPLIT, DEVICE, OUTPUT_DIR, CLIPS_ROOT,
    CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
    SUPPORTED_ARCHS, get_model_cfg,
    weights_path,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model, model_forward
from gaze_cam.gradcam import build_cam_engine
from gaze_cam.gaze_utils import (
    load_gaze_file, slice_gaze_for_clip,
    parse_clip_stem, nearest_valid_xy,
)

ARCHS = ["r3d18", "slowfast_r50", "timesformer", "vivit"]
ARCH_LABELS = {
    "r3d18": "R3D-18",
    "slowfast_r50": "SlowFast R50",
    "timesformer": "TimeSformer",
    "vivit": "ViViT",
}


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


def get_cam_for_clip(arch, model, cam_engine, dataloader, target_stem):
    """Find target_stem in dataloader, compute CAM, return (cam_thw, pred_idx, gt_idx)."""
    model.eval()
    for x, y, paths, metas in dataloader:
        for i, m in enumerate(metas):
            if m["stem"] == target_stem:
                if isinstance(x, (list, tuple)):
                    xi = [t[i:i+1].to(DEVICE) for t in x]
                else:
                    xi = x[i:i+1].to(DEVICE)
                yi = y[i].item()
                cam_bthw, logits, used_cls = cam_engine.compute(xi, class_idx=None)
                pred = logits.argmax(1).item()
                return cam_bthw[0].detach().cpu(), pred, yi
    return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-index", type=int, default=None,
                        help="Index into test set to use (picks a clip)")
    parser.add_argument("--clip-stem", type=str, default=None,
                        help="Specific clip stem to use")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--frame-idx", type=int, default=None,
                        help="Which video frame to show (default: middle)")
    args = parser.parse_args()

    out_dir = Path("docs/writeup/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Find a good clip stem ---
    # Load r3d18 test loader first just to pick a clip
    _, test_loader, _, id2label, na = make_loaders(
        split=args.split, arch="r3d18", batch_size=1, max_train=1,
    )

    # Pick a clip
    if args.clip_stem:
        target_stem = args.clip_stem
    else:
        idx = args.clip_index if args.clip_index is not None else 5
        count = 0
        target_stem = None
        for _, _, _, metas in test_loader:
            for m in metas:
                if count == idx:
                    target_stem = m["stem"]
                    break
                count += 1
            if target_stem:
                break
        if target_stem is None:
            print(f"Could not find clip at index {idx}")
            return

    print(f"Using clip: {target_stem}")

    # --- Step 2: Decode raw frames + gaze ---
    parsed = parse_clip_stem(target_stem)
    session = parsed["session"]
    f0 = parsed["f0"]
    f1 = parsed["f1"]

    # Find clip video file
    clip_path = None
    for session_dir in CLIPS_ROOT.iterdir():
        candidate = session_dir / f"{target_stem}.mp4"
        if candidate.exists():
            clip_path = candidate
            break
    if clip_path is None:
        print(f"Could not find video for {target_stem}")
        return

    raw_frames = decode_clip_frames(clip_path)
    if not raw_frames:
        print("No frames decoded")
        return
    h, w = raw_frames[0].shape[:2]
    T_vid = len(raw_frames)

    # Pick frame
    frame_idx = args.frame_idx if args.frame_idx is not None else T_vid // 2
    frame_idx = min(frame_idx, T_vid - 1)
    raw_frame = raw_frames[frame_idx]

    # Gaze
    gaze_df = load_gaze_file(session)
    g_clip = slice_gaze_for_clip(gaze_df, f0, f1)
    g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))
    session_f = f0 + frame_idx
    gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
    gi_local = gi - g0_offset
    gi_local = int(np.clip(gi_local, 0, len(g_clip) - 1))
    gx, gy, _ = nearest_valid_xy(g_clip, gi_local, max_radius=30)
    has_gaze = gx is not None
    if has_gaze:
        gx_px = np.clip(gx * (w / CALIBRATION_W), 0, w - 1)
        gy_px = np.clip(gy * (h / CALIBRATION_H), 0, h - 1)

    # --- Step 3: For each model, compute CAM ---
    cam_overlays = {}  # arch -> (cam_2d_fullres, pred_label, gt_label, correct)

    for arch in ARCHS:
        print(f"\n--- {ARCH_LABELS[arch]} ---")
        _, tl, _, id2l, na_arch = make_loaders(
            split=args.split, arch=arch, batch_size=1, max_train=1,
        )

        model = build_model(arch, num_actions=na_arch, device=DEVICE)
        ckpt = weights_path(args.split, arch)
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            print(f"  Loaded {ckpt.name}")
        else:
            print(f"  WARNING: no checkpoint at {ckpt}, using pretrained weights")
        model.eval()

        cam_engine = build_cam_engine(arch, model)
        cam_thw, pred_idx, gt_idx = get_cam_for_clip(arch, model, cam_engine, tl, target_stem)

        if cam_thw is None:
            print(f"  Clip {target_stem} not found in {arch} test loader, skipping")
            cam_overlays[arch] = None
            continue

        # Upsample CAM to video resolution
        cam_up = F.interpolate(
            cam_thw.unsqueeze(0).unsqueeze(0),
            size=(T_vid, h, w),
            mode="trilinear", align_corners=False,
        ).squeeze().numpy()

        cam_2d = cam_up[frame_idx]
        # Normalize to [0, 1]
        cmin, cmax = cam_2d.min(), cam_2d.max()
        if cmax - cmin > 1e-8:
            cam_2d = (cam_2d - cmin) / (cmax - cmin)
        else:
            cam_2d = np.zeros_like(cam_2d)

        gt_label = id2l.get(gt_idx, f"cls{gt_idx}")
        pred_label = id2l.get(pred_idx, f"cls{pred_idx}")
        correct = pred_idx == gt_idx

        cam_overlays[arch] = (cam_2d, pred_label, gt_label, correct)

        # Free GPU memory
        del model, cam_engine
        torch.cuda.empty_cache()

    # --- Step 4: Build the composite figure ---
    # Layout: 4 rows (one per model) x 3 columns (Raw+Gaze | CAM Overlay | CAM+Gaze)
    valid_archs = [a for a in ARCHS if cam_overlays.get(a) is not None]
    n_rows = len(valid_archs)
    n_cols = 3

    fig = plt.figure(figsize=(10, 3.0 * n_rows + 0.6))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=0.15,
                  left=0.08, right=0.98, top=0.94, bottom=0.02)

    col_titles = ["Raw Frame + Gaze", "Saliency Map", "Saliency + Gaze"]

    for row_i, arch in enumerate(valid_archs):
        cam_2d, pred_label, gt_label, correct = cam_overlays[arch]

        # Make heatmap
        cam_u8 = (cam_2d * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        alpha = 0.45
        blended = (raw_frame.astype(float) * (1 - alpha) +
                   heatmap.astype(float) * alpha).clip(0, 255).astype(np.uint8)

        # Panel images
        panels = [raw_frame.copy(), blended.copy(), blended.copy()]

        # Draw gaze dot on panels 0 and 2
        if has_gaze:
            for pi in [0, 2]:
                cx, cy = int(gx_px), int(gy_px)
                # Outer ring (white for visibility)
                cv2.circle(panels[pi], (cx, cy), 14, (255, 255, 255), 2)
                # Green ring
                cv2.circle(panels[pi], (cx, cy), 12, (0, 255, 0), 2)
                # Green dot
                cv2.circle(panels[pi], (cx, cy), 3, (0, 255, 0), -1)

        for col_i in range(n_cols):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.imshow(panels[col_i])
            ax.set_xticks([])
            ax.set_yticks([])

            # Column titles on first row
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=11, fontweight="bold", pad=6)

            # Row labels on first column
            if col_i == 0:
                mark = "correct" if correct else "incorrect"
                color = "#2e7d32" if correct else "#c62828"
                label = f"{ARCH_LABELS[arch]}\n({mark})"
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              color=color, rotation=90, labelpad=10)

    # Suptitle with clip info
    first_arch = valid_archs[0]
    gt_label = cam_overlays[first_arch][2]
    fig.suptitle(f"Ground Truth: {gt_label}  |  Clip: {target_stem}",
                 fontsize=12, fontweight="bold", y=0.99)

    out_path = out_dir / "fig_teaser.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved teaser figure -> {out_path}")


if __name__ == "__main__":
    main()
