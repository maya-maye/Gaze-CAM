"""
analysis.py  –  Full gaze-vs-CAM alignment analysis pipeline.

Computes:
  1. Per-clip alignment metrics (NSS, AUC-Judd, KL divergence)
  2. Correct vs incorrect comparison + shuffle baseline + center-bias baseline
  3. Error bucketing by verb / noun
  4. Temporal lead/lag analysis
  5. Model randomization control (random-weight CAMs)
  6. Occlusion sensitivity (CAM faithfulness check)
  7. Publication-quality plots

Usage:
    python scripts/analysis.py [--model r3d18] [--split 1] [--max-test 500] [--lag-range 5]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import rel_entr
from tqdm import tqdm

from gaze_cam.config import (
    SPLIT, ARCH, DEVICE, OUTPUT_DIR, CLIPS_ROOT,
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

ANALYSIS_DIR = OUTPUT_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Reduced resolution for in-memory CAM cache to avoid OOM (full res = 480x640).
# Gaze coordinates are rescaled accordingly — no accuracy loss for point lookups.
CAM_CACHE_H = 120
CAM_CACHE_W = 160


def _model_dir(arch: str) -> Path:
    """Per-model output directory inside analysis/."""
    d = ANALYSIS_DIR / arch
    d.mkdir(parents=True, exist_ok=True)
    return d


def gaze_in_bounds(gx: float, gy: float) -> bool:
    """Return True if gaze coordinates are within the calibration area."""
    return 0 <= gx <= CALIBRATION_W and 0 <= gy <= CALIBRATION_H


# ═══════════════════════════════════════════════
#  METRIC FUNCTIONS
# ═══════════════════════════════════════════════

def cam_to_frame_map(cam_thw: torch.Tensor, frame_h: int, frame_w: int) -> np.ndarray:
    """
    Upsample a (T, H_cam, W_cam) CAM tensor to (T, frame_h, frame_w) numpy.
    """
    cam = cam_thw.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    up = F.interpolate(cam, size=(cam_thw.shape[0], frame_h, frame_w),
                       mode="trilinear", align_corners=False)
    return up.squeeze().cpu().numpy()  # (T, frame_h, frame_w)


def compute_nss(saliency_map: np.ndarray, gaze_x: float, gaze_y: float,
                frame_h: int, frame_w: int) -> float:
    """
    Normalized Scanpath Saliency: z-score of saliency at the gaze point.
    saliency_map: (H, W) — raw or upsampled CAM for one frame.
    gaze_x, gaze_y: gaze in calibration-space pixels.
    Returns NaN if saliency has zero variance or gaze is out of bounds.
    """
    if not gaze_in_bounds(gaze_x, gaze_y):
        return np.nan
    # Convert gaze from calibration space to frame space
    gx = gaze_x * (frame_w / CALIBRATION_W)
    gy = gaze_y * (frame_h / CALIBRATION_H)
    gx_i = int(np.clip(round(gx), 0, frame_w - 1))
    gy_i = int(np.clip(round(gy), 0, frame_h - 1))

    mu = saliency_map.mean()
    sigma = saliency_map.std()
    if sigma < 1e-8:
        return np.nan
    return (saliency_map[gy_i, gx_i] - mu) / sigma


def compute_auc_judd(saliency_map: np.ndarray, gaze_x: float, gaze_y: float,
                     frame_h: int, frame_w: int) -> float:
    """
    AUC-Judd: area under ROC with gaze point(s) as positives.
    Vectorized for speed.
    """
    if not gaze_in_bounds(gaze_x, gaze_y):
        return np.nan
    gx = gaze_x * (frame_w / CALIBRATION_W)
    gy = gaze_y * (frame_h / CALIBRATION_H)
    gx_i = int(np.clip(round(gx), 0, frame_w - 1))
    gy_i = int(np.clip(round(gy), 0, frame_h - 1))

    s_gaze = saliency_map[gy_i, gx_i]
    s_flat = saliency_map.flatten()

    # Fraction of map >= gaze saliency value gives the false positive rate
    # at the threshold where the true positive = 1.
    # Simple AUC: proportion of pixels with saliency < gaze saliency + 0.5 * equal
    n = len(s_flat)
    above = (s_flat > s_gaze).sum()
    equal = (s_flat == s_gaze).sum()
    # AUC = P(random negative has lower saliency than the positive)
    auc = (n - above - equal + 0.5 * equal) / max(1, n)
    return float(auc)


def clip_alignment_metrics(
    cam_thw: np.ndarray,
    gaze_df: pd.DataFrame,
    f0: int, f1: int,
    frame_h: int, frame_w: int,
    temporal_shift: int = 0,
    compute_auc: bool = True,
) -> dict:
    """
    Compute frame-averaged NSS and AUC for one clip.

    temporal_shift: shift gaze by this many video frames relative to CAM.
      shift=+2 means we compare gaze at frame t with CAM at frame t-2.
      Positive shift = model leads. Negative = model lags.
    compute_auc: if False, skip AUC (much faster for baseline/temporal).
    """
    T_cam = cam_thw.shape[0]
    T_vid = f1 - f0 + 1
    # Map CAM frames to video frames
    cam_indices = np.linspace(0, T_cam - 1, T_vid).astype(int)

    g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))

    nss_values = []
    auc_values = []
    valid_frames = 0
    prev_gx = prev_gy = None  # saccade detection state (per-clip)

    for fi in range(T_vid):
        # Gaze frame: apply temporal shift
        gaze_fi = fi + temporal_shift
        if gaze_fi < 0 or gaze_fi >= T_vid:
            continue

        # Get gaze coordinates (with shift applied)
        session_f_gaze = f0 + gaze_fi
        gi = int(round(session_f_gaze * (GAZE_HZ / VIDEO_FPS)))
        gi_local = gi - g0_offset
        gi_local = int(np.clip(gi_local, 0, len(gaze_df) - 1))

        gx, gy, _ = nearest_valid_xy(gaze_df, gi_local, max_radius=30)
        if gx is None or not gaze_in_bounds(gx, gy):
            continue

        # Saccade / blink detection: skip if gaze jumps > 100px
        if prev_gx is not None:
            dx = abs(gx - prev_gx)
            dy = abs(gy - prev_gy)
            if dx > 100 or dy > 100:
                prev_gx, prev_gy = gx, gy
                continue
        prev_gx, prev_gy = gx, gy

        # CAM for this video frame (no shift — CAM stays fixed)
        cam_slice = cam_thw[cam_indices[fi]]  # (H, W)

        nss = compute_nss(cam_slice, gx, gy, frame_h, frame_w)
        if compute_auc:
            auc = compute_auc_judd(cam_slice, gx, gy, frame_h, frame_w)

        if not np.isnan(nss):
            nss_values.append(nss)
        if compute_auc and not np.isnan(auc):
            auc_values.append(auc)
        valid_frames += 1

    return {
        "nss_mean": float(np.mean(nss_values)) if nss_values else np.nan,
        "auc_mean": float(np.mean(auc_values)) if auc_values else np.nan,
        "valid_frames": valid_frames,
        "total_frames": T_vid,
    }


def compute_kl_divergence(cam_2d: np.ndarray, gaze_x: float, gaze_y: float,
                          frame_h: int, frame_w: int,
                          gaze_sigma: float = 25.0) -> float:
    """
    KL divergence between CAM probability map and a Gaussian placed at gaze.
    Lower = more agreement.
    """
    if not gaze_in_bounds(gaze_x, gaze_y):
        return np.nan
    gx = gaze_x * (frame_w / CALIBRATION_W)
    gy = gaze_y * (frame_h / CALIBRATION_H)

    # Build gaze fixation map: Gaussian at gaze location
    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    gaze_map = np.exp(-((xx - gx)**2 + (yy - gy)**2) / (2 * gaze_sigma**2))

    # Normalize both to probability distributions
    eps = 1e-10
    p = cam_2d.copy().astype(np.float64)
    p = np.maximum(p, 0)
    p_sum = p.sum()
    if p_sum < eps:
        return np.nan
    p = p / p_sum + eps

    q = gaze_map.astype(np.float64) + eps
    q = q / q.sum()

    # KL(gaze || cam) — how much info lost using CAM to represent gaze distribution
    kl = np.sum(rel_entr(q, p))
    return float(kl)


def make_center_bias_map(frame_h: int, frame_w: int, sigma_frac: float = 0.25) -> np.ndarray:
    """
    Create a Gaussian center-bias saliency map.
    sigma_frac: fraction of frame dimension used as Gaussian sigma.
    """
    cy, cx = frame_h / 2.0, frame_w / 2.0
    sigma_y = frame_h * sigma_frac
    sigma_x = frame_w * sigma_frac
    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    g = np.exp(-(((xx - cx) / sigma_x)**2 + ((yy - cy) / sigma_y)**2) / 2)
    return g / g.max()  # normalize to [0, 1]


def compute_center_bias_nss(gaze_df: pd.DataFrame, f0: int, f1: int,
                            frame_h: int, frame_w: int) -> float:
    """
    NSS of a center-bias Gaussian w.r.t. gaze points.
    Measures how much alignment is explained by center bias alone.
    """
    center_map = make_center_bias_map(frame_h, frame_w)

    T_vid = f1 - f0 + 1
    g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))

    nss_vals = []
    for fi in range(T_vid):
        session_f = f0 + fi
        gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
        gi_local = int(np.clip(gi - g0_offset, 0, len(gaze_df) - 1))
        gx, gy, _ = nearest_valid_xy(gaze_df, gi_local, max_radius=30)
        if gx is None or not gaze_in_bounds(gx, gy):
            continue
        nss = compute_nss(center_map, gx, gy, frame_h, frame_w)
        if not np.isnan(nss):
            nss_vals.append(nss)

    return float(np.mean(nss_vals)) if nss_vals else np.nan


# ═══════════════════════════════════════════════
#  VERB / NOUN PARSING
# ═══════════════════════════════════════════════

def parse_verb_noun(label: str) -> tuple:
    """
    Split 'Take sponge' → ('Take', 'sponge').
    Handles multi-word verbs like 'Move Around'.
    """
    parts = label.strip().split()
    if len(parts) == 0:
        return ("", "")
    if len(parts) == 1:
        return (parts[0], "")
    # Special: 'Move Around X' or 'Inspect/Read X'
    if len(parts) >= 3 and parts[0] in ("Move", "Inspect/Read"):
        verb = " ".join(parts[:2])
        noun = " ".join(parts[2:])
    else:
        verb = parts[0]
        noun = " ".join(parts[1:])
    return (verb, noun)


def error_bucket(gt_label: str, pred_label: str) -> str:
    """
    Categorize a prediction into one of 4 buckets:
      correct, right_verb_wrong_noun, wrong_verb_right_noun, completely_wrong
    """
    if gt_label == pred_label:
        return "correct"
    gt_v, gt_n = parse_verb_noun(gt_label)
    pr_v, pr_n = parse_verb_noun(pred_label)
    if gt_v == pr_v:
        return "right_verb_wrong_noun"
    if gt_n == pr_n:
        return "wrong_verb_right_noun"
    return "completely_wrong"


# ═══════════════════════════════════════════════
#  MAIN ANALYSIS LOOP
# ═══════════════════════════════════════════════

def run_analysis(args):
    arch = args.model
    print(f"Device: {DEVICE}  Arch: {arch}")
    print(f"Outputs -> {ANALYSIS_DIR}\n")

    # ── Load data & model ──
    print("Loading data...")
    max_test = args.max_test if args.max_test > 0 else None
    train_loader, test_loader, label_to_id, id_to_label, na = \
        make_loaders(
            split=args.split,
            batch_size=args.batch_size,
            arch=arch,
            max_train=10,
            max_test=max_test,
        )

    print("Loading model...")
    model = build_model(arch, na)
    ckpt_path = weights_path(args.split, arch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    # Restore label maps from checkpoint
    label_to_id.update(ckpt.get("label_to_id", {}))
    id_to_label.update(ckpt.get("id_to_label", {}))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded: {ckpt_path.name}\n")

    # ── CAM setup ──
    gcam = build_cam_engine(arch, model)

    # ── Check for existing CSV (resume support) ──
    out_dir = _model_dir(arch)
    csv_path = out_dir / f"alignment_split{args.split}.csv"
    resume = csv_path.exists() and not getattr(args, "force", False)

    # ── Collect per-clip results ──
    gaze_cache = {}
    rows = []
    clip_cam_cache = {}  # stem -> (cam_thw_np, meta) for temporal analysis

    if resume:
        print(f"RESUMING — loading existing CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows from previous run")

    desc = "clips (CAM-only, resuming)" if resume else "clips"
    print(f"\n{'Rebuilding CAM cache' if resume else 'Computing per-clip alignment metrics'}...\n")
    t0 = time.time()

    for batch_idx, (x_batch, y_batch, paths, metas) in enumerate(tqdm(test_loader, desc=desc)):
        if isinstance(x_batch, (list, tuple)):
            x_batch = [t.to(DEVICE) for t in x_batch]
        else:
            x_batch = x_batch.to(DEVICE)
        cam_bthw, logits, _ = gcam.compute(x_batch, class_idx=None)
        preds = logits.argmax(1).cpu()

        for i in range(len(paths)):
            meta = metas[i]
            stem = meta["stem"]
            gt_id = int(y_batch[i].item())
            pr_id = int(preds[i].item())
            gt_label = id_to_label.get(gt_id, str(gt_id))
            pr_label = id_to_label.get(pr_id, str(pr_id))
            correct = gt_id == pr_id

            session = meta["session"]
            f0_clip, f1_clip = meta["f0"], meta["f1"]

            # Load gaze
            if session not in gaze_cache:
                try:
                    gaze_cache[session] = load_gaze_file(session)
                except FileNotFoundError:
                    continue
            gaze_df = gaze_cache[session]
            g_clip = slice_gaze_for_clip(gaze_df, f0_clip, f1_clip)
            if len(g_clip) < 2:
                continue

            # Get frame dimensions from first video frame
            clip_path = Path(paths[i])
            cap = cv2.VideoCapture(str(clip_path))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            frame_h, frame_w = frame.shape[:2]

            # Upsample CAM
            cam_i = cam_bthw[i].cpu()  # (T_model, H_cam, W_cam)

            # Store a reduced-resolution copy in the cache to avoid OOM
            # (500 clips × ~100 frames × 480×640 × 8 bytes ≈ 125 GB)
            cam_cached = cam_to_frame_map(cam_i, CAM_CACHE_H, CAM_CACHE_W)
            cam_cached = cam_cached.astype(np.float32)
            clip_cam_cache[stem] = (cam_cached, g_clip, f0_clip, f1_clip, CAM_CACHE_H, CAM_CACHE_W)

            # Skip metric computation on resume — we already have the CSV
            if resume:
                continue

            # Full-resolution CAM for per-clip metrics (not kept in memory)
            cam_up = cam_to_frame_map(cam_i, frame_h, frame_w)

            # Compute alignment (shift=0)
            metrics = clip_alignment_metrics(
                cam_up, g_clip, f0_clip, f1_clip,
                frame_h, frame_w, temporal_shift=0,
            )

            # Parse verb/noun and error bucket
            gt_v, gt_n = parse_verb_noun(gt_label)
            pr_v, pr_n = parse_verb_noun(pr_label)
            bucket = error_bucket(gt_label, pr_label)

            # Compute per-frame KL divergence
            T_cam = cam_up.shape[0]
            T_vid = f1_clip - f0_clip + 1
            cam_idx = np.linspace(0, T_cam - 1, T_vid).astype(int)
            g0_off = int(round(f0_clip * (GAZE_HZ / VIDEO_FPS)))
            kl_vals = []
            for fi in range(T_vid):
                sf = f0_clip + fi
                gi = int(round(sf * (GAZE_HZ / VIDEO_FPS)))
                gi_l = int(np.clip(gi - g0_off, 0, len(g_clip) - 1))
                gx, gy, _ = nearest_valid_xy(g_clip, gi_l, max_radius=30)
                if gx is None or not gaze_in_bounds(gx, gy):
                    continue
                kl = compute_kl_divergence(cam_up[cam_idx[fi]], gx, gy, frame_h, frame_w)
                if not np.isnan(kl):
                    kl_vals.append(kl)

            # Center-bias baseline NSS
            cb_nss = compute_center_bias_nss(g_clip, f0_clip, f1_clip, frame_h, frame_w)

            row = {
                "stem": stem,
                "session": session,
                "gt_label": gt_label,
                "pred_label": pr_label,
                "gt_verb": gt_v,
                "gt_noun": gt_n,
                "pred_verb": pr_v,
                "pred_noun": pr_n,
                "correct": int(correct),
                "bucket": bucket,
                "nss": metrics["nss_mean"],
                "auc": metrics["auc_mean"],
                "kl_divergence": float(np.mean(kl_vals)) if kl_vals else np.nan,
                "center_bias_nss": cb_nss,
                "valid_frames": metrics["valid_frames"],
                "total_frames": metrics["total_frames"],
            }
            rows.append(row)

    gcam.close()
    elapsed = time.time() - t0

    if not resume:
        print(f"\nProcessed {len(rows)} clips in {elapsed:.1f}s")
        if len(rows) == 0:
            print("ERROR: No clips with valid gaze data found.")
            return
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved per-clip results -> {csv_path}")
    else:
        print(f"\nRebuilt CAM cache for {len(clip_cam_cache)} clips in {elapsed:.1f}s")

    # ── Print summary ──
    print_summary(df)

    # ── Shuffle baseline ──
    print("\n" + "="*60)
    print("SHUFFLE BASELINE (random gaze-to-clip assignment)")
    print("="*60)
    shuffle_nss = compute_shuffle_baseline(df, clip_cam_cache, n_shuffles=args.n_shuffles)

    # ── Center-bias baseline summary ──
    cb = df["center_bias_nss"].dropna()
    if len(cb) > 0:
        print("\n" + "="*60)
        print("CENTER-BIAS BASELINE")
        print("="*60)
        print(f"  Center-bias NSS: {cb.mean():.4f} ± {cb.std():.4f} (n={len(cb)})")
        print(f"  Model CAM NSS:   {df['nss'].dropna().mean():.4f}")
        excess = df['nss'].dropna().mean() - cb.mean()
        print(f"  Excess over center-bias: {excess:+.4f}")

    # ── KL divergence summary ──
    kl = df["kl_divergence"].dropna()
    if len(kl) > 0:
        print(f"\n  KL divergence (gaze||cam): {kl.mean():.4f} ± {kl.std():.4f}")
        c_kl = df[df["correct"] == 1]["kl_divergence"].dropna()
        i_kl = df[df["correct"] == 0]["kl_divergence"].dropna()
        if len(c_kl) > 0 and len(i_kl) > 0:
            print(f"    Correct:   {c_kl.mean():.4f} ± {c_kl.std():.4f}")
            print(f"    Incorrect: {i_kl.mean():.4f} ± {i_kl.std():.4f}")
            if len(c_kl) >= 5 and len(i_kl) >= 5:
                u, p = stats.mannwhitneyu(c_kl, i_kl, alternative="two-sided")
                print(f"    Mann-Whitney U: p={p:.6f}")

    # ── Model randomization control ──
    if not args.skip_randomization:
        print("\n" + "="*60)
        print("MODEL RANDOMIZATION CONTROL (random-weight CAMs)")
        print("="*60)
        rand_nss = run_randomization_control(
            arch, na, test_loader, gaze_cache, clip_cam_cache,
        )
    else:
        rand_nss = None

    # ── Occlusion sensitivity ──
    if not args.skip_occlusion:
        print("\n" + "="*60)
        print("OCCLUSION SENSITIVITY (CAM faithfulness)")
        print("="*60)
        run_occlusion_sensitivity(model, arch, test_loader, clip_cam_cache, out_dir)
    else:
        print("\n  [Skipped occlusion sensitivity — use --skip-occlusion=false]")

    # ── Temporal lead/lag ──
    print("\n" + "="*60)
    print("TEMPORAL LEAD / LAG ANALYSIS")
    print("="*60)
    lag_df = compute_temporal_shifts(clip_cam_cache, args.lag_range, out_dir)

    # ── Generate all plots ──
    print("\n" + "="*60)
    print(f"GENERATING PLOTS -> {out_dir}")
    print("="*60)
    plot_correct_vs_incorrect(df, shuffle_nss, rand_nss, out_dir)
    plot_error_buckets(df, shuffle_nss, out_dir)
    plot_temporal_lag(lag_df, args.lag_range, out_dir)
    plot_verb_alignment(df, out_dir=out_dir)
    plot_noun_alignment(df, out_dir=out_dir)

    print(f"\nAll outputs saved to: {out_dir}")


# ═══════════════════════════════════════════════
#  SUMMARY PRINTER
# ═══════════════════════════════════════════════

def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)

    total = len(df)
    correct = df[df["correct"] == 1]
    incorrect = df[df["correct"] == 0]

    print(f"\nTotal clips analyzed: {total}")
    print(f"  Correct:   {len(correct)} ({100*len(correct)/total:.1f}%)")
    print(f"  Incorrect: {len(incorrect)} ({100*len(incorrect)/total:.1f}%)")

    print(f"\n{'Condition':<30} {'NSS (mean+/-std)':<22} {'AUC (mean+/-std)':<22} {'N':>5}")
    print("-"*80)
    for name, subset in [("ALL", df), ("Correct", correct), ("Incorrect", incorrect)]:
        valid = subset.dropna(subset=["nss"])
        if len(valid) == 0:
            continue
        nss_m, nss_s = valid["nss"].mean(), valid["nss"].std()
        auc_m, auc_s = valid["auc"].mean(), valid["auc"].std()
        print(f"  {name:<28} {nss_m:+.4f} +/- {nss_s:.4f}    {auc_m:.4f} +/- {auc_s:.4f}    {len(valid):>5}")

    # Statistical test
    c_nss = correct["nss"].dropna()
    i_nss = incorrect["nss"].dropna()
    if len(c_nss) >= 5 and len(i_nss) >= 5:
        u_stat, p_val = stats.mannwhitneyu(c_nss, i_nss, alternative="two-sided")
        # Effect size: rank-biserial correlation
        n1, n2 = len(c_nss), len(i_nss)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        print(f"\n  Mann-Whitney U test (NSS, correct vs incorrect):")
        print(f"    U = {u_stat:.1f},  p = {p_val:.6f},  rank-biserial r = {r_rb:+.4f}")
        if p_val < 0.05:
            print(f"    -> Significant at p < 0.05")
        else:
            print(f"    -> NOT significant at p < 0.05")

    # Error buckets
    print(f"\n{'Bucket':<30} {'NSS (mean+/-std)':<22} {'N':>5}")
    print("-"*60)
    for bucket in ["correct", "right_verb_wrong_noun", "wrong_verb_right_noun", "completely_wrong"]:
        sub = df[df["bucket"] == bucket].dropna(subset=["nss"])
        if len(sub) == 0:
            continue
        nss_m, nss_s = sub["nss"].mean(), sub["nss"].std()
        print(f"  {bucket:<28} {nss_m:+.4f} +/- {nss_s:.4f}    {len(sub):>5}")

    # Per-noun summary (top/bottom 5)
    noun_stats = df.groupby("gt_noun")["nss"].agg(["mean", "count"])
    noun_stats = noun_stats[noun_stats["count"] >= 3].sort_values("mean")
    if len(noun_stats) >= 4:
        n = min(5, len(noun_stats) // 2)
        print(f"\n  Top {n} nouns by NSS:")
        for noun, row in noun_stats.tail(n).iterrows():
            print(f"    {noun:<25} {row['mean']:+.4f}  (n={int(row['count'])})")
        print(f"  Bottom {n} nouns by NSS:")
        for noun, row in noun_stats.head(n).iterrows():
            print(f"    {noun:<25} {row['mean']:+.4f}  (n={int(row['count'])})")


# ═══════════════════════════════════════════════
#  SHUFFLE BASELINE
# ═══════════════════════════════════════════════

def compute_shuffle_baseline(df: pd.DataFrame, cam_cache: dict,
                             n_shuffles: int = 50) -> list:
    """
    Randomly reassign gaze from other clips to each clip's CAM.
    Returns list of mean-NSS values across shuffles (one per shuffle).
    """
    stems = list(cam_cache.keys())
    if len(stems) < 3:
        print("  Not enough clips for shuffle baseline.")
        return []

    shuffle_means = []
    rng = np.random.default_rng(42)

    for si in tqdm(range(n_shuffles), desc="shuffle baseline"):
        nss_vals = []
        shuffled_stems = rng.permutation(stems)
        # Pair each clip's CAM with a different clip's gaze
        for idx, stem in enumerate(stems):
            donor_stem = shuffled_stems[idx]
            if donor_stem == stem:
                # Shift by one to avoid self-pairing
                donor_stem = shuffled_stems[(idx + 1) % len(shuffled_stems)]

            if stem not in cam_cache or donor_stem not in cam_cache:
                continue

            cam_up, _, f0, f1, fh, fw = cam_cache[stem]
            _, donor_gaze, donor_f0, donor_f1, _, _ = cam_cache[donor_stem]

            # Use donor gaze with current clip's CAM (NSS only for speed)
            metrics = clip_alignment_metrics(
                cam_up, donor_gaze, donor_f0, donor_f1,
                fh, fw, temporal_shift=0, compute_auc=False,
            )
            if not np.isnan(metrics["nss_mean"]):
                nss_vals.append(metrics["nss_mean"])

        if nss_vals:
            shuffle_means.append(np.mean(nss_vals))

    if shuffle_means:
        sm = np.array(shuffle_means)
        print(f"  Shuffle baseline NSS: {sm.mean():.4f} +/- {sm.std():.4f}")
        print(f"  Actual NSS (all):     {df['nss'].dropna().mean():.4f}")
        print(f"  Actual NSS (correct): {df[df['correct']==1]['nss'].dropna().mean():.4f}")
    return shuffle_means


# ═══════════════════════════════════════════════
#  TEMPORAL LEAD / LAG
# ═══════════════════════════════════════════════

def compute_temporal_shifts(cam_cache: dict, lag_range: int = 5,
                            out_dir: Path = ANALYSIS_DIR) -> pd.DataFrame:
    """
    Compute alignment at different temporal shifts.
    shift = +k: gaze is shifted +k frames ahead → model leads
    shift = -k: gaze is shifted -k frames back → model lags

    Optimized: precompute z-scored CAM maps and gaze lookups once,
    then just do a lookup per shift.
    """
    shifts = list(range(-lag_range, lag_range + 1))

    # Precompute per-clip: z-scored CAM frames + gaze points per video frame
    precomputed = {}
    for stem, (cam_up, g_clip, f0, f1, fh, fw) in cam_cache.items():
        T_cam = cam_up.shape[0]
        T_vid = f1 - f0 + 1
        cam_indices = np.linspace(0, T_cam - 1, T_vid).astype(int)
        g0_offset = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))

        # Z-score each CAM frame
        z_maps = []
        for fi in range(T_vid):
            cam_slice = cam_up[cam_indices[fi]]
            mu = cam_slice.mean()
            sigma = cam_slice.std()
            if sigma < 1e-8:
                z_maps.append(None)
            else:
                z_maps.append((cam_slice - mu) / sigma)

        # Get gaze pixel coords for each video frame
        gaze_coords = []
        prev_gx, prev_gy = None, None
        for fi in range(T_vid):
            session_f = f0 + fi
            gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
            gi_local = gi - g0_offset
            gi_local = int(np.clip(gi_local, 0, len(g_clip) - 1))
            gx, gy, _ = nearest_valid_xy(g_clip, gi_local, max_radius=30)
            if gx is None or not gaze_in_bounds(gx, gy):
                gaze_coords.append(None)
                continue
            # Saccade filter
            if prev_gx is not None and (abs(gx - prev_gx) > 100 or abs(gy - prev_gy) > 100):
                gaze_coords.append(None)
            else:
                gx_i = int(np.clip(round(gx * (fw / CALIBRATION_W)), 0, fw - 1))
                gy_i = int(np.clip(round(gy * (fh / CALIBRATION_H)), 0, fh - 1))
                gaze_coords.append((gy_i, gx_i))
            prev_gx, prev_gy = gx, gy

        precomputed[stem] = (z_maps, gaze_coords, T_vid)

    # Now compute NSS at each shift as simple lookups
    results = []
    for shift in shifts:
        nss_vals = []
        for stem, (z_maps, gaze_coords, T_vid) in precomputed.items():
            frame_nss = []
            for fi in range(T_vid):
                gaze_fi = fi + shift
                if gaze_fi < 0 or gaze_fi >= T_vid:
                    continue
                z_map = z_maps[fi]
                gc = gaze_coords[gaze_fi]
                if z_map is None or gc is None:
                    continue
                gy_i, gx_i = gc
                frame_nss.append(float(z_map[gy_i, gx_i]))
            if frame_nss:
                nss_vals.append(np.mean(frame_nss))

        mean_nss = np.mean(nss_vals) if nss_vals else np.nan
        std_nss = np.std(nss_vals) if nss_vals else np.nan
        sem_nss = std_nss / np.sqrt(len(nss_vals)) if nss_vals else np.nan
        results.append({
            "shift": shift,
            "nss_mean": mean_nss,
            "nss_std": std_nss,
            "nss_sem": sem_nss,
            "n_clips": len(nss_vals),
        })
        label = ""
        if shift > 0:
            label = " (model leads)"
        elif shift < 0:
            label = " (model lags)"
        print(f"  shift={shift:+d}{label}: NSS={mean_nss:.4f} +/- {sem_nss:.4f}  (n={len(nss_vals)})")

    lag_df = pd.DataFrame(results)
    lag_csv = out_dir / "temporal_shifts.csv"
    lag_df.to_csv(lag_csv, index=False)
    print(f"  Saved -> {lag_csv.name}")

    # Report peak
    if not lag_df.empty:
        peak = lag_df.loc[lag_df["nss_mean"].idxmax()]
        print(f"\n  Peak alignment at shift={int(peak['shift']):+d} "
              f"({abs(int(peak['shift']))*1000/VIDEO_FPS:.0f} ms), "
              f"NSS={peak['nss_mean']:.4f}")

    return lag_df


# ═══════════════════════════════════════════════
#  MODEL RANDOMIZATION CONTROL
# ═══════════════════════════════════════════════

def run_randomization_control(arch, na, test_loader, gaze_cache, clip_cam_cache):
    """
    Compute CAMs using a randomly-initialized model (no training).
    If alignment is similar to trained → CAM alignment is spurious.
    If alignment drops to ~0 → CAM alignment is meaningful.
    """
    rand_model = build_model(arch, na)
    rand_model.to(DEVICE).eval()
    rand_gcam = build_cam_engine(arch, rand_model)

    nss_vals = []
    stems_done = set()

    for x_batch, y_batch, paths, metas in tqdm(test_loader, desc="random-model CAMs"):
        if isinstance(x_batch, (list, tuple)):
            x_batch = [t.to(DEVICE) for t in x_batch]
        else:
            x_batch = x_batch.to(DEVICE)
        cam_bthw, _, _ = rand_gcam.compute(x_batch, class_idx=None)

        for i in range(len(paths)):
            meta = metas[i]
            stem = meta["stem"]
            if stem in stems_done or stem not in clip_cam_cache:
                continue
            stems_done.add(stem)

            _, g_clip, f0, f1, fh, fw = clip_cam_cache[stem]
            cam_i = cam_bthw[i].cpu()
            cam_up = cam_to_frame_map(cam_i, fh, fw)

            metrics = clip_alignment_metrics(
                cam_up, g_clip, f0, f1, fh, fw,
                temporal_shift=0, compute_auc=False,
            )
            if not np.isnan(metrics["nss_mean"]):
                nss_vals.append(metrics["nss_mean"])

    rand_gcam.close()
    del rand_model
    gc.collect()
    torch.cuda.empty_cache()

    if nss_vals:
        arr = np.array(nss_vals)
        print(f"  Random-model NSS:  {arr.mean():.4f} ± {arr.std():.4f} (n={len(arr)})")
    else:
        print("  No valid clips for randomization control.")
    return nss_vals


# ═══════════════════════════════════════════════
#  OCCLUSION SENSITIVITY
# ═══════════════════════════════════════════════

def run_occlusion_sensitivity(model, arch, test_loader, clip_cam_cache, out_dir,
                              patch_frac: float = 0.25, n_clips: int = 50):
    """
    For each clip, mask the highest-CAM region and the lowest-CAM region,
    and compare the prediction confidence drop.
    If masking high-CAM drops confidence more → CAM is faithful.
    """
    model.eval()
    high_drops, low_drops = [], []
    count = 0

    for x_batch, y_batch, paths, metas in tqdm(test_loader, desc="occlusion sensitivity"):
        if count >= n_clips:
            break
        if isinstance(x_batch, (list, tuple)):
            x_batch_dev = [t.to(DEVICE) for t in x_batch]
        else:
            x_batch_dev = x_batch.to(DEVICE)

        with torch.no_grad():
            logits_orig = model_forward(arch, model, x_batch_dev)
            probs_orig = torch.softmax(logits_orig, dim=1)

        for i in range(len(paths)):
            if count >= n_clips:
                break
            meta = metas[i]
            stem = meta["stem"]
            if stem not in clip_cam_cache:
                continue

            cam_up, _, _, _, fh, fw = clip_cam_cache[stem]
            gt = int(y_batch[i].item())
            orig_conf = float(probs_orig[i, gt].item())

            # Average CAM across time to get spatial importance map
            cam_spatial = cam_up.mean(axis=0)  # (H, W)

            # Resize cam_spatial to input tensor spatial size
            if isinstance(x_batch, (list, tuple)):
                inp = x_batch[0][i:i+1]  # slow pathway for slowfast
            else:
                inp = x_batch[i:i+1]
            _, _, T_inp, H_inp, W_inp = inp.shape
            cam_resized = cv2.resize(cam_spatial, (W_inp, H_inp))

            # Determine mask size
            ph = max(1, int(H_inp * patch_frac))
            pw = max(1, int(W_inp * patch_frac))

            # High-CAM region: center of a patch_frac×patch_frac window around max
            flat_idx = np.argmax(cam_resized)
            cy, cx = divmod(flat_idx, W_inp)
            y0h = max(0, cy - ph // 2)
            x0h = max(0, cx - pw // 2)
            y1h = min(H_inp, y0h + ph)
            x1h = min(W_inp, x0h + pw)

            # Low-CAM region: center of a window around min
            flat_idx_lo = np.argmin(cam_resized)
            cy_lo, cx_lo = divmod(flat_idx_lo, W_inp)
            y0l = max(0, cy_lo - ph // 2)
            x0l = max(0, cx_lo - pw // 2)
            y1l = min(H_inp, y0l + ph)
            x1l = min(W_inp, x0l + pw)

            # Mask high-CAM region (zero out)
            if isinstance(x_batch, (list, tuple)):
                masked_high = [t[i:i+1].clone().to(DEVICE) for t in x_batch]
                for t in masked_high:
                    t[:, :, :, y0h:y1h, x0h:x1h] = 0
                masked_low = [t[i:i+1].clone().to(DEVICE) for t in x_batch]
                for t in masked_low:
                    t[:, :, :, y0l:y1l, x0l:x1l] = 0
            else:
                masked_high = x_batch[i:i+1].clone().to(DEVICE)
                masked_high[:, :, :, y0h:y1h, x0h:x1h] = 0
                masked_low = x_batch[i:i+1].clone().to(DEVICE)
                masked_low[:, :, :, y0l:y1l, x0l:x1l] = 0

            with torch.no_grad():
                logits_h = model_forward(arch, model, masked_high)
                logits_l = model_forward(arch, model, masked_low)
                conf_h = float(torch.softmax(logits_h, dim=1)[0, gt].item())
                conf_l = float(torch.softmax(logits_l, dim=1)[0, gt].item())

            high_drops.append(orig_conf - conf_h)
            low_drops.append(orig_conf - conf_l)
            count += 1

    if high_drops:
        hd = np.array(high_drops)
        ld = np.array(low_drops)
        print(f"  Mask high-CAM region → avg confidence drop: {hd.mean():.4f} ± {hd.std():.4f}")
        print(f"  Mask low-CAM region  → avg confidence drop: {ld.mean():.4f} ± {ld.std():.4f}")
        print(f"  Difference (high-low): {(hd.mean() - ld.mean()):.4f}")
        if hd.mean() > ld.mean():
            print("  ✓ Masking high-CAM hurts more → CAM is faithful")
        else:
            print("  ✗ Masking low-CAM hurts more → CAM may not be faithful")

        # Save to CSV
        occ_df = pd.DataFrame({
            "high_cam_drop": high_drops,
            "low_cam_drop": low_drops,
        })
        occ_df.to_csv(out_dir / "occlusion_sensitivity.csv", index=False)


# ═══════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════

def plot_correct_vs_incorrect(df: pd.DataFrame, shuffle_nss: list,
                              rand_nss: list | None = None,
                              out_dir: Path = ANALYSIS_DIR):
    """Box/violin plot: NSS for correct vs incorrect + baselines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    correct_nss = df[df["correct"] == 1]["nss"].dropna().values
    incorrect_nss = df[df["correct"] == 0]["nss"].dropna().values

    data = [correct_nss, incorrect_nss]
    labels_list = ["Correct", "Incorrect"]
    colors = ["#4CAF50", "#F44336"]

    if shuffle_nss:
        data.append(np.array(shuffle_nss))
        labels_list.append("Shuffle\nBaseline")
        colors.append("#9E9E9E")

    # Center-bias reference
    cb = df["center_bias_nss"].dropna().values
    if len(cb) > 0:
        data.append(cb)
        labels_list.append("Center\nBias")
        colors.append("#FF9800")

    if rand_nss:
        data.append(np.array(rand_nss))
        labels_list.append("Random\nWeights")
        colors.append("#795548")

    # Filter out empty arrays (can happen with small test sets)
    valid = [(d, l, c) for d, l, c in zip(data, labels_list, colors) if len(d) > 0]
    if not valid:
        plt.close(fig)
        print("  Skipped correct_vs_incorrect plot (not enough data)")
        return
    data, labels_list, colors = zip(*valid)
    data, labels_list, colors = list(data), list(labels_list), list(colors)

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels_list)
    ax.set_ylabel("NSS (Normalized Scanpath Saliency)")
    ax.set_title("Grad-CAM × Gaze Alignment: Correct vs Incorrect Predictions")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Chance (NSS=0)")
    ax.legend(loc="upper right")

    # Add counts
    for i, d in enumerate(data):
        ax.text(i, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={len(d)}", ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    out = out_dir / "fig_correct_vs_incorrect.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_error_buckets(df: pd.DataFrame, shuffle_nss: list | None = None,
                      out_dir: Path = ANALYSIS_DIR):
    """Bar chart: NSS by error bucket — zoomed to show real differences."""
    bucket_order = ["correct", "right_verb_wrong_noun", "wrong_verb_right_noun", "completely_wrong"]
    bucket_labels = ["Correct", "Right Verb\nWrong Noun", "Wrong Verb\nRight Noun", "Completely\nWrong"]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    means, sems, counts = [], [], []
    for b in bucket_order:
        sub = df[df["bucket"] == b]["nss"].dropna()
        means.append(sub.mean() if len(sub) > 0 else 0)
        sems.append(sub.sem() if len(sub) > 1 else 0)
        counts.append(len(sub))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(bucket_order))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Mean NSS")
    ax.set_title("Gaze Alignment by Error Type (Verb / Noun)")

    # Shuffle baseline reference line
    if shuffle_nss:
        shuf_mean = float(np.mean(shuffle_nss))
        ax.axhline(shuf_mean, color="#9E9E9E", linestyle="--", linewidth=1.5,
                   label=f"Shuffle baseline (NSS={shuf_mean:.2f})")
        ax.legend(loc="lower left", fontsize=9)

    # Zoom y-axis to where the data lives — show the differences
    all_vals = means + ([shuf_mean] if shuffle_nss else [])
    lo = min(all_vals) - max(sems) - 0.08
    hi = max(all_vals) + max(sems) + 0.08
    ax.set_ylim(lo, hi)

    for i, (m, c) in enumerate(zip(means, counts)):
        ax.text(i, m + sems[i] + 0.01, f"n={c}", ha="center", fontsize=9, color="gray")

    fig.tight_layout()
    out = out_dir / "fig_error_buckets.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_temporal_lag(lag_df: pd.DataFrame, lag_range: int,
                     out_dir: Path = ANALYSIS_DIR):
    """Line plot: NSS vs temporal shift."""
    if lag_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    shifts = lag_df["shift"].values
    nss_mean = lag_df["nss_mean"].values
    nss_sem = lag_df["nss_sem"].values

    ax.fill_between(shifts, nss_mean - nss_sem, nss_mean + nss_sem, alpha=0.2, color="#2196F3")
    ax.plot(shifts, nss_mean, "o-", color="#2196F3", linewidth=2, markersize=6)

    # Mark peak
    peak_idx = np.nanargmax(nss_mean)
    ax.plot(shifts[peak_idx], nss_mean[peak_idx], "r*", markersize=15,
            label=f"Peak: shift={shifts[peak_idx]:+d}")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Temporal Shift (frames)\n← Model lags human | Model leads human →")
    ax.set_ylabel("Mean NSS")
    ax.set_title(f"Temporal Lead/Lag Analysis (±{lag_range} frames, {1000/VIDEO_FPS:.0f} ms/frame)")
    ax.legend()
    ax.set_xticks(range(-lag_range, lag_range + 1))

    fig.tight_layout()
    out = out_dir / "fig_temporal_lag.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_verb_alignment(df: pd.DataFrame, top_n: int = 10, out_dir: Path = ANALYSIS_DIR):
    """Horizontal bar chart: top/bottom verbs by mean NSS."""
    verb_stats = df.groupby("gt_verb")["nss"].agg(["mean", "sem", "count"])
    verb_stats = verb_stats[verb_stats["count"] >= 3].sort_values("mean")

    if len(verb_stats) < 4:
        print("  Not enough verb categories for verb alignment plot.")
        return

    # Take top and bottom N
    n = min(top_n, len(verb_stats) // 2)
    selected = pd.concat([verb_stats.head(n), verb_stats.tail(n)])

    fig, ax = plt.subplots(figsize=(9, max(5, len(selected) * 0.4)))
    y_pos = range(len(selected))
    colors = ["#F44336" if m < 0 else "#4CAF50" for m in selected["mean"]]

    ax.barh(y_pos, selected["mean"], xerr=selected["sem"],
            capsize=3, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v} (n={int(c)})" for v, c in zip(selected.index, selected["count"])])
    ax.set_xlabel("Mean NSS")
    ax.set_title(f"Gaze Alignment by Verb (Top/Bottom {n})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out = out_dir / "fig_verb_alignment.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_noun_alignment(df: pd.DataFrame, top_n: int = 10, out_dir: Path = ANALYSIS_DIR):
    """Horizontal bar chart: top/bottom nouns by mean NSS."""
    noun_stats = df.groupby("gt_noun")["nss"].agg(["mean", "sem", "count"])
    noun_stats = noun_stats[noun_stats["count"] >= 3].sort_values("mean")

    if len(noun_stats) < 4:
        print("  Not enough noun categories for noun alignment plot.")
        return

    n = min(top_n, len(noun_stats) // 2)
    selected = pd.concat([noun_stats.head(n), noun_stats.tail(n)])

    fig, ax = plt.subplots(figsize=(9, max(5, len(selected) * 0.4)))
    y_pos = range(len(selected))
    colors = ["#F44336" if m < 0 else "#2196F3" for m in selected["mean"]]

    ax.barh(y_pos, selected["mean"], xerr=selected["sem"],
            capsize=3, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{v} (n={int(c)})" for v, c in zip(selected.index, selected["count"])])
    ax.set_xlabel("Mean NSS")
    ax.set_title(f"Gaze Alignment by Noun (Top/Bottom {n})")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out = out_dir / "fig_noun_alignment.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gaze-CAM alignment analysis")
    parser.add_argument("--model", type=str, default=ARCH, choices=SUPPORTED_ARCHS,
                        help="Model architecture")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lag-range", type=int, default=5,
                        help="Max temporal shift in frames (±K)")
    parser.add_argument("--n-shuffles", type=int, default=50,
                        help="Number of shuffle iterations for baseline")
    parser.add_argument("--skip-randomization", action="store_true",
                        help="Skip the model-randomization control")
    parser.add_argument("--skip-occlusion", action="store_true",
                        help="Skip occlusion sensitivity analysis")
    parser.add_argument("--force", action="store_true",
                        help="Force full recompute even if CSV exists")
    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
