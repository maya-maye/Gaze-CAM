"""
Quick diagnostic: is the high shuffle-baseline NSS a code bug or center bias?

Tests 4 conditions on a handful of test clips:
  1. Actual alignment (clip's own gaze → own CAM)
  2. Shuffled gaze (another clip's gaze → this CAM)
  3. Uniform random gaze (should give NSS ≈ 0 if code is correct)
  4. Center-point gaze (always frame center → measures center bias of CAM)

If uniform-random ≈ 0 and center ≈ shuffle ≈ actual, the problem is center bias.
If uniform-random ≠ 0, there's a code bug.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaze_cam.config import (
    DEVICE, CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
    weights_path, get_model_cfg,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model, model_forward
from gaze_cam.gradcam import build_cam_engine
from gaze_cam.gaze_utils import load_gaze_file, slice_gaze_for_clip, nearest_valid_xy


# Use a fixed resolution for the diagnostic — avoids cv2 per-clip overhead
DIAG_H = 224
DIAG_W = 224


def compute_nss_frame(saliency_map, gx_pixel, gy_pixel):
    """NSS at a pixel-space gaze point. Returns NaN if zero-variance map."""
    h, w = saliency_map.shape
    gx_i = int(np.clip(round(gx_pixel), 0, w - 1))
    gy_i = int(np.clip(round(gy_pixel), 0, h - 1))
    mu = saliency_map.mean()
    sigma = saliency_map.std()
    if sigma < 1e-8:
        return np.nan
    return (saliency_map[gy_i, gx_i] - mu) / sigma


def upsample_cam(cam_t, frame_h, frame_w):
    """(T, Hc, Wc) tensor → (T, frame_h, frame_w) numpy."""
    c = cam_t.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(c, size=(cam_t.shape[0], frame_h, frame_w),
                       mode="trilinear", align_corners=False)
    return up.squeeze().cpu().numpy()


def main():
    arch = "r3d18"
    split = 1
    n_clips = 50  # enough for a quick test

    print(f"Loading data (arch={arch}, split={split}, max_test={n_clips})...")
    _, test_loader, label_to_id, id_to_label, na = \
        make_loaders(split=split, batch_size=4, arch=arch,
                     max_train=2, max_test=n_clips)

    print("Loading model...")
    model = build_model(arch, na)
    ckpt = torch.load(weights_path(split, arch), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE).eval()

    gcam = build_cam_engine(arch, model)
    gaze_cache = {}
    rng = np.random.default_rng(42)

    actual_nss, shuffle_nss, uniform_nss, center_nss = [], [], [], []
    # Collect cam+gaze for shuffle
    clip_data = []  # list of (cam_up, g_clip, f0, f1, fh, fw)

    print("\nProcessing clips...")
    for x_batch, y_batch, paths, metas in tqdm(test_loader, desc="clips"):
        if isinstance(x_batch, (list, tuple)):
            x_batch = [t.to(DEVICE) for t in x_batch]
        else:
            x_batch = x_batch.to(DEVICE)
        cam_bthw, logits, _ = gcam.compute(x_batch, class_idx=None)

        for i in range(len(paths)):
            meta = metas[i]
            session = meta["session"]
            f0, f1 = meta["f0"], meta["f1"]

            if session not in gaze_cache:
                try:
                    gaze_cache[session] = load_gaze_file(session)
                except FileNotFoundError:
                    continue
            g_clip = slice_gaze_for_clip(gaze_cache[session], f0, f1)
            if len(g_clip) < 2:
                continue

            cam_up = upsample_cam(cam_bthw[i].cpu(), DIAG_H, DIAG_W)
            clip_data.append((cam_up, g_clip, f0, f1, DIAG_H, DIAG_W))

    gcam.close()
    print(f"\nCollected {len(clip_data)} clips with valid gaze.\n")

    # Now compute 4 baselines
    n = len(clip_data)
    for idx in range(n):
        cam_up, g_clip, f0, f1, fh, fw = clip_data[idx]
        T_cam = cam_up.shape[0]
        T_vid = f1 - f0 + 1
        cam_indices = np.linspace(0, T_cam - 1, T_vid).astype(int)
        g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))

        # Pick a random donor for shuffle
        donor_idx = (idx + rng.integers(1, max(2, n - 1))) % n
        d_cam, d_gaze, d_f0, d_f1, d_fh, d_fw = clip_data[donor_idx]

        a_vals, s_vals, u_vals, c_vals = [], [], [], []

        for fi in range(min(T_vid, len(cam_indices))):
            cam_slice = cam_up[cam_indices[fi]]

            # ── 1. Actual gaze ──
            session_f = f0 + fi
            gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
            gi_local = int(np.clip(gi - g0_offset, 0, len(g_clip) - 1))
            gx, gy, _ = nearest_valid_xy(g_clip, gi_local, max_radius=30)
            if gx is not None:
                gx_px = gx * (fw / CALIBRATION_W)
                gy_px = gy * (fh / CALIBRATION_H)
                nss = compute_nss_frame(cam_slice, gx_px, gy_px)
                if not np.isnan(nss):
                    a_vals.append(nss)

            # ── 2. Donor (shuffle) gaze ──
            d_T_vid = d_f1 - d_f0 + 1
            d_fi = int(fi * d_T_vid / T_vid)  # map to donor timeline
            d_g0_off = int(round(d_f0 * (GAZE_HZ / VIDEO_FPS)))
            d_session_f = d_f0 + d_fi
            d_gi = int(round(d_session_f * (GAZE_HZ / VIDEO_FPS)))
            d_gi_local = int(np.clip(d_gi - d_g0_off, 0, len(d_gaze) - 1))
            dgx, dgy, _ = nearest_valid_xy(d_gaze, d_gi_local, max_radius=30)
            if dgx is not None:
                dgx_px = dgx * (fw / CALIBRATION_W)
                dgy_px = dgy * (fh / CALIBRATION_H)
                nss = compute_nss_frame(cam_slice, dgx_px, dgy_px)
                if not np.isnan(nss):
                    s_vals.append(nss)

            # ── 3. Uniform random gaze (in calibration space) ──
            rx = rng.uniform(0, CALIBRATION_W)
            ry = rng.uniform(0, CALIBRATION_H)
            rx_px = rx * (fw / CALIBRATION_W)
            ry_px = ry * (fh / CALIBRATION_H)
            nss = compute_nss_frame(cam_slice, rx_px, ry_px)
            if not np.isnan(nss):
                u_vals.append(nss)

            # ── 4. Center-point gaze ──
            cx = fw / 2.0
            cy = fh / 2.0
            nss = compute_nss_frame(cam_slice, cx, cy)
            if not np.isnan(nss):
                c_vals.append(nss)

        if a_vals: actual_nss.append(np.mean(a_vals))
        if s_vals: shuffle_nss.append(np.mean(s_vals))
        if u_vals: uniform_nss.append(np.mean(u_vals))
        if c_vals: center_nss.append(np.mean(c_vals))

    # Print results
    print("=" * 65)
    print("SHUFFLE BASELINE DIAGNOSTIC")
    print("=" * 65)
    for name, vals in [
        ("Actual gaze  (should be > 0)", actual_nss),
        ("Shuffle gaze  (investigating)", shuffle_nss),
        ("Uniform random (should be ≈ 0)", uniform_nss),
        ("Center point  (center bias)", center_nss),
    ]:
        arr = np.array(vals)
        print(f"  {name:35s}: {arr.mean():+.4f} ± {arr.std():.4f}  (n={len(arr)})")

    print()
    if abs(np.mean(uniform_nss)) < 0.15:
        print("✓ Uniform baseline ≈ 0 → NSS code is correct.")
        if np.mean(center_nss) > 0.5:
            print("  High center-point NSS → strong CENTER BIAS in CAMs.")
            print("  The high shuffle baseline is REAL — it's measuring center bias,")
            print("  not a code bug. We need a center-bias baseline metric.")
        else:
            print("  Center-point NSS is low → something else is going on.")
    else:
        print("✗ Uniform baseline ≠ 0 → POSSIBLE CODE BUG in NSS computation!")

    # Also check: where do gaze points typically land?
    all_gx, all_gy = [], []
    for _, g_clip, _, _, _, _ in clip_data:
        valid = g_clip.dropna(subset=["x", "y"])
        all_gx.extend(valid["x"].values)
        all_gy.extend(valid["y"].values)
    all_gx, all_gy = np.array(all_gx), np.array(all_gy)
    print(f"\n  Gaze spatial distribution (calibration space {CALIBRATION_W}×{CALIBRATION_H}):")
    print(f"    X: mean={all_gx.mean():.1f}, std={all_gx.std():.1f}  "
          f"(center={CALIBRATION_W/2:.0f})")
    print(f"    Y: mean={all_gy.mean():.1f}, std={all_gy.std():.1f}  "
          f"(center={CALIBRATION_H/2:.0f})")


if __name__ == "__main__":
    main()
