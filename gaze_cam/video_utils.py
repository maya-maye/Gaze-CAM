"""
Video loading, preprocessing, and visualization helpers.

Supports multiple model architectures:
  - r3d18        → (3, T, H, W) tensor at 112×112
  - slowfast_r50 → tuple (slow, fast) at 224×224
  - timesformer  → (T, C, H, W) tensor at 224×224 via HF processor
  - vivit        → (T, C, H, W) tensor at 224×224 via HF processor
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt
from pathlib import Path

from gaze_cam.config import (
    NUM_FRAMES, INPUT_SIZE, ARCH,
    KINETICS_MEAN, KINETICS_STD,
    CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
    get_model_cfg,
)


# ──────────────────────────────────────────────
# Normalization tensors (lazy, created once)
# ──────────────────────────────────────────────

_MEAN = torch.tensor(KINETICS_MEAN).view(1, 3, 1, 1, 1)
_STD = torch.tensor(KINETICS_STD).view(1, 3, 1, 1, 1)


# ──────────────────────────────────────────────
# Raw frame loading (shared by all architectures)
# ──────────────────────────────────────────────

def load_raw_frames(clip_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """
    Decode an mp4 with OpenCV and uniformly sample *num_frames* frames.
    Returns (T, H, W, 3) uint8 numpy array in RGB order.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {clip_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Too few frames ({len(frames)}) in: {clip_path}")

    frames = np.asarray(frames)  # (T_all, H, W, 3)
    T = frames.shape[0]
    idx = np.linspace(0, T - 1, num_frames).round().astype(int)
    idx = np.clip(idx, 0, T - 1)
    return frames[idx]  # (num_frames, H, W, 3)


# ──────────────────────────────────────────────
# R3D-18 preprocessing
# ──────────────────────────────────────────────

def preprocess_bcthw(x_bcthw: torch.Tensor, out_size: int = INPUT_SIZE) -> torch.Tensor:
    """
    Resize + Kinetics-normalise a (B, C, T, H, W) float [0,1] tensor.
    """
    B, C, T, H, W = x_bcthw.shape
    x = x_bcthw.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.view(B, T, C, out_size, out_size).permute(0, 2, 1, 3, 4).contiguous()
    mean = _MEAN.to(x.device, x.dtype)
    std = _STD.to(x.device, x.dtype)
    return (x - mean) / std


def _preprocess_r3d18(frames_rgb: np.ndarray, input_size: int = 112) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (3, T, H, W) float, normalised."""
    x = torch.from_numpy(frames_rgb).float() / 255.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    x = preprocess_bcthw(x, out_size=input_size)
    return x[0]  # (3, T, H, W)


# ──────────────────────────────────────────────
# SlowFast preprocessing
# ──────────────────────────────────────────────

def _preprocess_slowfast(frames_rgb: np.ndarray, input_size: int = 224,
                         alpha: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (T, H, W, 3) uint8 → (slow, fast) where
        fast = (3, T, H, W)  and  slow = (3, T//alpha, H, W)
    Both are Kinetics-normalised floats.
    """
    x = torch.from_numpy(frames_rgb).float() / 255.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
    x = preprocess_bcthw(x, out_size=input_size)[0]  # (3, T, H, W)

    fast = x                        # (3, T, H, W)
    slow = x[:, ::alpha, :, :]      # (3, T//alpha, H, W)
    return slow, fast


# ──────────────────────────────────────────────
# HuggingFace processor cache
# ──────────────────────────────────────────────

_HF_PROCESSOR_CACHE: dict = {}


def get_hf_processor(arch: str):
    """Return a cached HuggingFace image/video processor for *arch*."""
    if arch in _HF_PROCESSOR_CACHE:
        return _HF_PROCESSOR_CACHE[arch]

    if arch == "timesformer":
        from transformers import VideoMAEImageProcessor
        proc = VideoMAEImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
    elif arch == "vivit":
        from transformers import VivitImageProcessor
        proc = VivitImageProcessor.from_pretrained(
            "google/vivit-b-16x2-kinetics400"
        )
    else:
        proc = None

    _HF_PROCESSOR_CACHE[arch] = proc
    return proc


def _preprocess_transformer(frames_rgb: np.ndarray, arch: str) -> torch.Tensor:
    """
    (T, H, W, 3) uint8 → (T, C, H, W) float tensor via HuggingFace processor.
    """
    processor = get_hf_processor(arch)
    # Processor expects a list of PIL-like arrays: list of (H, W, 3) numpy
    frame_list = [frames_rgb[t] for t in range(frames_rgb.shape[0])]
    inputs = processor(frame_list, return_tensors="pt")
    # pixel_values shape: (1, T, C, H, W)
    return inputs["pixel_values"].squeeze(0)  # (T, C, H, W)


# ──────────────────────────────────────────────
# Unified clip loader
# ──────────────────────────────────────────────

def load_clip_tensor(clip_path: Path,
                     num_frames: int = NUM_FRAMES,
                     input_size: int = INPUT_SIZE,
                     arch: str = ARCH):
    """
    Decode an mp4 and return a preprocessed tensor (or tuple) ready
    for the specified architecture.

    Returns:
      r3d18        → Tensor (3, T, H, W)
      slowfast_r50 → tuple  (slow (3, Ts, H, W), fast (3, Tf, H, W))
      timesformer  → Tensor (T, C, H, W)
      vivit        → Tensor (T, C, H, W)
    """
    frames = load_raw_frames(clip_path, num_frames)

    if arch == "r3d18":
        return _preprocess_r3d18(frames, input_size)
    elif arch == "slowfast_r50":
        cfg = get_model_cfg(arch)
        return _preprocess_slowfast(frames, input_size, cfg.get("slowfast_alpha", 4))
    elif arch in ("timesformer", "vivit"):
        return _preprocess_transformer(frames, arch)
    else:
        # Fallback: treat like r3d18
        return _preprocess_r3d18(frames, input_size)


# ──────────────────────────────────────────────
# Gaze overlay visualisation
# ──────────────────────────────────────────────

def overlay_gaze_on_frame(
    frame_bgr: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    cal_w: int = CALIBRATION_W,
    cal_h: int = CALIBRATION_H,
    radius: int = 8,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Draw a gaze dot on a BGR frame, rescaling from calibration resolution.
    Returns the annotated frame (a copy).
    """
    h, w = frame_bgr.shape[:2]
    x = int(np.clip(round(gaze_x * (w / cal_w)), 0, w - 1))
    y = int(np.clip(round(gaze_y * (h / cal_h)), 0, h - 1))
    out = frame_bgr.copy()
    cv2.circle(out, (x, y), radius, color, -1)
    return out


def save_gaze_overlay_grid(
    clip_path: Path,
    g_clip,
    f0: int,
    out_path: Path,
    frame_indices=(0, 10, 30, 60, 100),
):
    """
    Save a grid of frames with gaze overlays for visual inspection.
    """
    from gaze_cam.gaze_utils import nearest_valid_xy

    cap = cv2.VideoCapture(str(clip_path))
    images = []

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        session_f = f0 + i
        gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
        gi_local = gi - int(round(f0 * (GAZE_HZ / VIDEO_FPS)))
        gi_local = int(np.clip(gi_local, 0, len(g_clip) - 1))

        x, y, gi_used = nearest_valid_xy(g_clip, gi_local, max_radius=30)
        if x is not None:
            frame = overlay_gaze_on_frame(frame, x, y)

        images.append((i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cap.release()

    if not images:
        return

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (idx, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(f"frame {idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    print(f"Saved overlay grid -> {out_path}")
