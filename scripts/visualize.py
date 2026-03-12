"""
visualize.py  –  Generate sample visual outputs:
  1. Gaze overlay on raw video frames
  2. Grad-CAM / Attention-CAM heatmaps on video frames
  3. Combined gaze + CAM overlay
  4. Animated GIFs showing CAM + gaze over time

Usage:
    python scripts/visualize.py [--model r3d18] [--split 1] [--num-clips 4] [--gifs]
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
from PIL import Image, ImageDraw, ImageFont

from gaze_cam.config import (
    SPLIT, ARCH, DEVICE, OUTPUT_DIR, CLIPS_ROOT,
    CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
    SUPPORTED_ARCHS, get_model_cfg,
    weights_path,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model, load_or_train
from gaze_cam.gradcam import build_cam_engine
from gaze_cam.gaze_utils import (
    load_gaze_file, slice_gaze_for_clip,
    parse_clip_stem, nearest_valid_xy,
)


VIS_DIR = OUTPUT_DIR / "visualizations"


def decode_clip_frames(clip_path, max_frames=None):
    """Decode all (or max_frames) RGB frames from a clip."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def make_cam_heatmap(cam_2d, h, w):
    """Convert a (Hc, Wc) float [0,1] map to a jet-colored (h, w, 3) RGB."""
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
    """Draw a gaze circle on image (in-place copy)."""
    out = frame_rgb.copy()
    cv2.circle(out, (int(gx), int(gy)), radius, color, thickness)
    cv2.circle(out, (int(gx), int(gy)), 3, color, -1)
    return out


def generate_gif(
    clip_path, meta, gt_label, pred_label,
    cam_thw, gaze_df, f0, f1,
    out_path, gif_fps=10, max_width=480,
):
    """
    Create an animated GIF with three side-by-side panels per frame:
      Left   = raw frame + gaze dot
      Center = Grad-CAM heatmap overlay
      Right  = combined (CAM + gaze)
    A text banner at the top shows GT vs Pred.
    """
    frames = decode_clip_frames(clip_path)
    if len(frames) < 2:
        return False

    h, w = frames[0].shape[:2]
    T_vid = len(frames)
    T_cam = cam_thw.shape[0]

    # Upsample CAM to full video resolution
    cam_t = cam_thw.unsqueeze(0).unsqueeze(0)
    cam_up = F.interpolate(
        cam_t, size=(T_vid, h, w),
        mode="trilinear", align_corners=False,
    ).squeeze().cpu().numpy()

    g_clip = slice_gaze_for_clip(gaze_df, f0, f1)
    g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))

    # Scale factor for manageable GIF size
    scale = max_width / (w * 3)
    new_w = int(w * scale)
    new_h = int(h * scale)
    banner_h = 36

    pil_frames = []
    correct = gt_label == pred_label
    mark = "CORRECT" if correct else "WRONG"
    title = f"GT: {gt_label}  |  Pred: {pred_label}  [{mark}]"

    for fi in range(T_vid):
        frame = frames[fi]
        cam_slice = cam_up[fi]
        heatmap = make_cam_heatmap(cam_slice, h, w)

        # Gaze for this frame
        session_f = f0 + fi
        gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
        gi_local = gi - g0_offset
        gi_local = int(np.clip(gi_local, 0, len(g_clip) - 1))
        gx, gy, _ = nearest_valid_xy(
            g_clip, gi_local, max_radius=30
        )
        has_gaze = gx is not None
        if has_gaze:
            gx_px = np.clip(gx * (w / CALIBRATION_W), 0, w - 1)
            gy_px = np.clip(gy * (h / CALIBRATION_H), 0, h - 1)

        # Panel 1: frame + gaze
        p1 = frame.copy()
        if has_gaze:
            p1 = draw_gaze_dot(p1, gx_px, gy_px)

        # Panel 2: CAM overlay
        p2 = blend(frame, heatmap)

        # Panel 3: CAM + gaze
        p3 = blend(frame, heatmap)
        if has_gaze:
            p3 = draw_gaze_dot(
                p3, gx_px, gy_px,
                color=(0, 255, 0), radius=12, thickness=3,
            )

        # Resize and concatenate
        p1s = cv2.resize(p1, (new_w, new_h))
        p2s = cv2.resize(p2, (new_w, new_h))
        p3s = cv2.resize(p3, (new_w, new_h))
        row = np.concatenate([p1s, p2s, p3s], axis=1)

        # Top banner: GT vs Pred label
        top_banner = np.full(
            (banner_h, row.shape[1], 3), 30, dtype=np.uint8
        )
        pil_top = Image.fromarray(top_banner)
        draw_top = ImageDraw.Draw(pil_top)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()

        col = (0, 255, 100) if correct else (255, 100, 100)
        draw_top.text((8, 8), title, fill=col, font=font)
        top_banner = np.array(pil_top)

        # Bottom banner: frame counter
        bot_banner = np.full(
            (banner_h, row.shape[1], 3), 30, dtype=np.uint8
        )
        pil_bot = Image.fromarray(bot_banner)
        draw_bot = ImageDraw.Draw(pil_bot)
        fc_text = f"frame {fi}/{T_vid-1}"
        draw_bot.text((8, 8), fc_text, fill=(200, 200, 200), font=font)
        bot_banner = np.array(pil_bot)

        combined = np.concatenate([top_banner, row, bot_banner], axis=0)

        pil_frames.append(Image.fromarray(combined))

    # Subsample if too many frames (keep GIF < ~200 frames)
    if len(pil_frames) > 150:
        step = max(1, len(pil_frames) // 150)
        pil_frames = pil_frames[::step]

    duration = max(30, int(1000 / gif_fps))
    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    return True


def visualize_clip(
    clip_idx, clip_path, meta, gt_label, pred_label,
    cam_thw, gaze_df, f0, f1,
    num_sample_frames=8,
):
    """
    Generate a multi-row figure for one clip:
      Row 1: Raw frames with gaze dot
      Row 2: Grad-CAM heatmap overlay
      Row 3: Combined gaze + CAM
    """
    frames = decode_clip_frames(clip_path)
    if len(frames) == 0:
        print(f"  Skipping {clip_path.name}: no frames decoded")
        return None

    h, w = frames[0].shape[:2]
    T_vid = len(frames)
    T_cam = cam_thw.shape[0]

    # Upsample CAM temporally+spatially to match video frames
    cam_t = cam_thw.unsqueeze(0).unsqueeze(0)  # (1,1,Tc,Hc,Wc)
    cam_up = F.interpolate(
        cam_t, size=(T_vid, h, w),
        mode="trilinear", align_corners=False,
    ).squeeze().cpu().numpy()  # (T_vid, h, w)

    # Prepare gaze data for this clip
    g_clip = slice_gaze_for_clip(gaze_df, f0, f1)

    # Pick evenly-spaced sample frames
    idxs = np.linspace(
        0, T_vid - 1, min(num_sample_frames, T_vid)
    ).astype(int)

    fig, axes = plt.subplots(
        3, len(idxs),
        figsize=(3.5 * len(idxs), 10),
    )
    if len(idxs) == 1:
        axes = axes.reshape(3, 1)

    correct = (gt_label == pred_label)
    mark = "CORRECT" if correct else "WRONG"
    fig.suptitle(
        f"Clip #{clip_idx}: {meta['stem']}\n"
        f"GT: {gt_label}  |  Pred: {pred_label}  [{mark}]",
        fontsize=11, fontweight="bold", y=0.99,
    )

    for col, fi in enumerate(idxs):
        frame = frames[fi]
        cam_slice = cam_up[fi]
        heatmap = make_cam_heatmap(cam_slice, h, w)

        # Gaze point for this frame
        session_f = f0 + fi
        gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
        gi_local = gi - int(round(f0 * (GAZE_HZ / VIDEO_FPS)))
        gi_local = int(np.clip(gi_local, 0, len(g_clip) - 1))
        gx, gy, _ = nearest_valid_xy(g_clip, gi_local, max_radius=30)

        # Scale gaze to frame resolution
        has_gaze = gx is not None
        if has_gaze:
            gx_px = gx * (w / CALIBRATION_W)
            gy_px = gy * (h / CALIBRATION_H)
            gx_px = np.clip(gx_px, 0, w - 1)
            gy_px = np.clip(gy_px, 0, h - 1)

        # Row 0: raw frame + gaze
        row0 = frame.copy()
        if has_gaze:
            row0 = draw_gaze_dot(row0, gx_px, gy_px)
        axes[0, col].imshow(row0)
        axes[0, col].set_title(f"f{fi}", fontsize=8)
        axes[0, col].axis("off")

        # Row 1: CAM overlay
        row1 = blend(frame, heatmap)
        axes[1, col].imshow(row1)
        axes[1, col].axis("off")

        # Row 2: CAM + gaze
        row2 = blend(frame, heatmap)
        if has_gaze:
            row2 = draw_gaze_dot(
                row2, gx_px, gy_px,
                color=(0, 255, 0), radius=12, thickness=3,
            )
        axes[2, col].imshow(row2)
        axes[2, col].axis("off")

    # Row labels
    axes[0, 0].set_ylabel("Gaze", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=10, fontweight="bold")
    axes[2, 0].set_ylabel("CAM + Gaze", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gaze + CAM visualizations"
    )
    parser.add_argument("--model", type=str, default=ARCH, choices=SUPPORTED_ARCHS,
                        help="Model architecture")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--num-clips", type=int, default=6,
                        help="Number of clips to visualize")
    parser.add_argument("--max-test", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gifs", action="store_true",
                        help="Also generate animated GIF overlays")
    parser.add_argument("--clips", nargs="*", default=None,
                        help="Specific clip stems to regenerate (bypasses random sampling)")
    args = parser.parse_args()

    arch = args.model
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}  Arch: {arch}")
    print(f"Outputs -> {VIS_DIR}\n")

    target_stems = set(args.clips) if args.clips else None

    # ── Load data & model ──
    print("Loading data...")
    effective_max_test = args.max_test if not target_stems else max(args.max_test, 2000)
    train_loader, test_loader, label_to_id, id_to_label, na = \
        make_loaders(
            split=args.split,
            batch_size=args.batch_size,
            arch=arch,
            max_train=10,
            max_test=effective_max_test,
        )

    print("Loading model...")
    model = build_model(arch, na)
    if target_stems:
        ckpt_path = weights_path(args.split, arch)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(DEVICE)
        print(f"\n  Loaded checkpoint: {ckpt_path.name}")
    else:
        model = load_or_train(
            model, train_loader, test_loader,
            label_to_id, id_to_label, na,
            arch=arch,
            split=args.split,
        )

    # ── CAM setup ──
    gcam = build_cam_engine(arch, model)
    model.eval()

    # ── Process clips ──
    clips_done = 0
    gaze_cache = {}
    num_target = len(target_stems) if target_stems else args.num_clips

    print(f"\nGenerating visualizations for {num_target} clips...\n")

    for x_batch, y_batch, paths, metas in test_loader:
        if isinstance(x_batch, (list, tuple)):
            x_batch = [t.to(DEVICE) for t in x_batch]
        else:
            x_batch = x_batch.to(DEVICE)

        cam_bthw, logits, _ = gcam.compute(x_batch, class_idx=None)
        preds = logits.argmax(1).cpu()

        for i in range(len(paths)):
            if clips_done >= num_target:
                break

            meta = metas[i]

            # If targeting specific clips, skip non-matching ones
            if target_stems and meta["stem"] not in target_stems:
                continue

            clip_path = Path(paths[i])
            gt_id = int(y_batch[i].item())
            pr_id = int(preds[i].item())
            gt_label = id_to_label.get(gt_id, str(gt_id))
            pr_label = id_to_label.get(pr_id, str(pr_id))

            session = meta["session"]
            f0, f1 = meta["f0"], meta["f1"]

            # Load gaze (cached per session)
            if session not in gaze_cache:
                try:
                    gaze_cache[session] = load_gaze_file(session)
                except FileNotFoundError:
                    print(f"  No gaze for {session}, skipping")
                    continue
            gaze_df = gaze_cache[session]

            cam_i = cam_bthw[i]  # (T, H, W)
            fig = visualize_clip(
                clips_done, clip_path, meta,
                gt_label, pr_label,
                cam_i, gaze_df, f0, f1,
            )

            if fig is not None:
                out_path = VIS_DIR / f"clip_{clips_done:02d}_{meta['stem']}.png"
                fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
                plt.close(fig)
                print(f"  [{clips_done+1}/{args.num_clips}] Saved: {out_path.name}")
                print(f"       GT={gt_label}  Pred={pr_label}")

                # Generate animated GIF
                if args.gifs:
                    gif_dir = VIS_DIR / "gifs"
                    gif_dir.mkdir(exist_ok=True)
                    gif_path = gif_dir / f"clip_{clips_done:02d}_{meta['stem']}.gif"
                    ok = generate_gif(
                        clip_path, meta, gt_label, pr_label,
                        cam_i, gaze_df, f0, f1,
                        gif_path,
                    )
                    if ok:
                        print(f"       GIF -> {gif_path.name}")

                clips_done += 1

        if clips_done >= num_target:
            break

    gcam.close()

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Done! {clips_done} visualizations saved to:")
    print(f"  {VIS_DIR}")
    if args.gifs:
        print(f"  {VIS_DIR / 'gifs'}")
    print(f"{'='*50}")
    print("\nEach image has 3 rows:")
    print("  Row 1: Raw frames + green gaze dot")
    print("  Row 2: Grad-CAM heatmap overlay")
    print("  Row 3: Combined (CAM + gaze dot)")
    if args.gifs:
        print("\nGIFs show 3 panels side-by-side animating over time:")
        print("  Left=Gaze  Center=Grad-CAM  Right=Combined")


if __name__ == "__main__":
    main()
