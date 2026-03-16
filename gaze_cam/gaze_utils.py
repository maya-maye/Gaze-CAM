"""
Gaze data loading, parsing, and alignment utilities for EGTEA Gaze+.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

from gaze_cam.config import (
    GAZE_DATA_DIR,
    CALIBRATION_W, CALIBRATION_H,
    VIDEO_FPS, GAZE_HZ,
)


# ──────────────────────────────────────────────
# Timecode helpers
# ──────────────────────────────────────────────

def timecode_to_frame30(tc: str) -> int:
    """Convert 'HH:MM:SS:FF' timecode (30 Hz frame counter) to absolute frame number."""
    hh, mm, ss, ff = str(tc).split(":")
    return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 30 + int(ff)


# ──────────────────────────────────────────────
# Clip metadata parsing
# ──────────────────────────────────────────────

_CLIP_RE = re.compile(r"^(?P<session>.+?)-(?P<t0>\d+)-(?P<t1>\d+)-F(?P<f0>\d+)-F(?P<f1>\d+)$")


def parse_clip_stem(stem: str):
    """
    Parse an EGTEA clip filename stem like
    'P02-R04-ContinentalBreakfast-544830-552337-F013057-F013275'
    Returns (session, t0_ms, t1_ms, f0, f1, prefix) or None.
    """
    m = _CLIP_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    session = d["session"]
    t0 = int(d["t0"])
    t1 = int(d["t1"])
    f0 = int(d["f0"])
    f1 = int(d["f1"])
    prefix = f"{session}-{t0}-{t1}"
    return session, t0, t1, f0, f1, prefix


# ──────────────────────────────────────────────
# Load gaze file into a clean DataFrame
# ──────────────────────────────────────────────

def load_gaze_file(session_key: str, gaze_dir: Path = None) -> pd.DataFrame:
    """
    Load gaze data for *session_key* (e.g. 'P02-R04-ContinentalBreakfast').
    Auto-detects header row and handles both binocular (B POR) and
    monocular (L POR / R POR) column formats.
    Returns a DataFrame with added columns: frame30, x, y, t_sec.
    """
    if gaze_dir is None:
        gaze_dir = GAZE_DATA_DIR / "gaze_data"

    gaze_path = gaze_dir / f"{session_key}.txt"
    if not gaze_path.exists():
        raise FileNotFoundError(f"Gaze file not found: {gaze_path}")

    # Auto-detect header: find the first line that doesn't start with ##
    skip = 0
    with open(gaze_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if not line.startswith("##"):
                skip = i
                break

    gaze = pd.read_csv(gaze_path, sep="\t", skiprows=skip, header=0)

    # Resolve X/Y columns: prefer binocular, fall back to L or R
    x_col = y_col = None
    for prefix in ["B", "L", "R"]:
        xc = f"{prefix} POR X [px]"
        yc = f"{prefix} POR Y [px]"
        if xc in gaze.columns and yc in gaze.columns:
            x_col, y_col = xc, yc
            break

    if x_col is None:
        raise ValueError(
            f"No POR X/Y columns found in {gaze_path}. "
            f"Columns: {list(gaze.columns)}"
        )

    gaze["x"] = pd.to_numeric(gaze[x_col], errors="coerce")
    gaze["y"] = pd.to_numeric(gaze[y_col], errors="coerce")

    # Parse Frame column: timecode "HH:MM:SS:FF" or plain integers
    sample_frame = str(gaze["Frame"].iloc[0]).strip()
    if ":" in sample_frame:
        gaze["frame30"] = gaze["Frame"].apply(timecode_to_frame30)
    else:
        gaze["frame30"] = pd.to_numeric(
            gaze["Frame"], errors="coerce"
        ).fillna(0).astype(int)

    # Time in seconds relative to start
    t_base_us = gaze["Time"].iloc[0]
    gaze["t_sec"] = (gaze["Time"] - t_base_us) / 1e6

    # Store which POR columns were used so downstream code can reference them
    gaze.attrs["x_col"] = x_col
    gaze.attrs["y_col"] = y_col

    return gaze


# ──────────────────────────────────────────────
# Slice gaze to a clip window
# ──────────────────────────────────────────────

def slice_gaze_for_clip(gaze: pd.DataFrame, f0: int, f1: int) -> pd.DataFrame:
    """
    Slice the full-session gaze DataFrame to the frame range [f0, f1]
    of a video clip, converting from video frame IDs to gaze indices.
    """
    g0 = int(round(f0 * (GAZE_HZ / VIDEO_FPS)))
    g1 = int(round(f1 * (GAZE_HZ / VIDEO_FPS)))
    g0 = max(g0, 0)
    g1 = min(g1, len(gaze) - 1)

    g_clip = gaze.iloc[g0:g1 + 1].copy()
    # x/y already set by load_gaze_file; re-coerce just in case
    x_col = gaze.attrs.get("x_col", "B POR X [px]")
    y_col = gaze.attrs.get("y_col", "B POR Y [px]")
    if x_col in g_clip.columns:
        g_clip["x"] = pd.to_numeric(g_clip[x_col], errors="coerce")
        g_clip["y"] = pd.to_numeric(g_clip[y_col], errors="coerce")
    return g_clip


# ──────────────────────────────────────────────
# Gaze point retrieval with nearest-valid fallback
# ──────────────────────────────────────────────

def nearest_valid_xy(df: pd.DataFrame, gi: int, max_radius: int = 20):
    """
    Return (x, y, gi_used) for the nearest row to *gi* that has valid x/y.
    Returns (None, None, None) if nothing valid is found within *max_radius*.
    """
    n = len(df)
    gi = int(np.clip(gi, 0, n - 1))

    if pd.notna(df.iloc[gi]["x"]) and pd.notna(df.iloc[gi]["y"]):
        return float(df.iloc[gi]["x"]), float(df.iloc[gi]["y"]), gi

    for r in range(1, max_radius + 1):
        lo, hi = gi - r, gi + r
        if lo >= 0 and pd.notna(df.iloc[lo]["x"]) and pd.notna(df.iloc[lo]["y"]):
            return float(df.iloc[lo]["x"]), float(df.iloc[lo]["y"]), lo
        if hi < n and pd.notna(df.iloc[hi]["x"]) and pd.notna(df.iloc[hi]["y"]):
            return float(df.iloc[hi]["x"]), float(df.iloc[hi]["y"]), hi

    return None, None, None


def gaze_xy_for_clip_frame(
    gaze: pd.DataFrame,
    clip_frame_idx: int,
    f0: int,
    frame_w: int,
    frame_h: int,
    cal_w: int = CALIBRATION_W,
    cal_h: int = CALIBRATION_H,
):
    """
    Get the (x, y) gaze coordinate in *pixel* space for clip
    frame *clip_frame_idx*, rescaled from the gaze calibration
    resolution to (frame_w, frame_h).
    Returns None if the gaze sample is invalid.
    """
    session_f = f0 + clip_frame_idx
    gi = int(round(session_f * (GAZE_HZ / VIDEO_FPS)))
    gi = int(np.clip(gi, 0, len(gaze) - 1))

    x = float(gaze.iloc[gi]["x"])
    y = float(gaze.iloc[gi]["y"])

    if not (0 <= x <= cal_w and 0 <= y <= cal_h):
        return None

    x = x * (frame_w / cal_w)
    y = y * (frame_h / cal_h)
    return int(round(x)), int(round(y))


def gaze_validity_ratio(g_clip: pd.DataFrame) -> float:
    """Fraction of gaze samples that fall within the calibration area."""
    valid = g_clip["x"].between(0, CALIBRATION_W) & g_clip["y"].between(0, CALIBRATION_H)
    return float(valid.mean())
