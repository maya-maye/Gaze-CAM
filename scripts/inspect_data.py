"""
inspect_data.py  –  Quick data-sanity checks (sections 1-3 of the original notebook).

Usage:
    python scripts/inspect_data.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from gaze_cam.config import (
    ACTION_ANNOTATION_DIR, GAZE_DATA_DIR, VIDEO_CLIPS_DIR,
    CLIPS_ROOT,
)
from gaze_cam.gaze_utils import load_gaze_file, slice_gaze_for_clip, parse_clip_stem, gaze_validity_ratio


def main():
    # ── 1. Directory sanity checks ──
    print("=== Directory checks ===")
    print("  annotations:", ACTION_ANNOTATION_DIR.exists(), ACTION_ANNOTATION_DIR)
    print("  gaze:       ", GAZE_DATA_DIR.exists(), GAZE_DATA_DIR)
    print("  videos:     ", VIDEO_CLIPS_DIR.exists(), VIDEO_CLIPS_DIR)

    ann_files = list(ACTION_ANNOTATION_DIR.glob("*"))[:10]
    print(f"  annotation files ({len(ann_files)}):", [p.name for p in ann_files])

    gaze_files = list(GAZE_DATA_DIR.rglob("*.txt"))[:10]
    print(f"  gaze files ({len(gaze_files)}):", [p.name for p in gaze_files])

    clips_root = CLIPS_ROOT if CLIPS_ROOT.exists() else VIDEO_CLIPS_DIR
    videos = list(clips_root.rglob("*.mp4"))
    print(f"  video clips found: {len(videos)}")

    if not videos:
        print("  WARNING: no .mp4 files found. Check VIDEO_CLIPS_DIR in config.py")
        return

    # ── 2. Pick a random clip ──
    vid = random.choice(videos)
    print(f"\n=== Random clip: {vid.name} ===")

    parsed = parse_clip_stem(vid.stem)
    if parsed is None:
        print("  Could not parse clip stem.")
        return

    session, t0, t1, f0, f1, prefix = parsed
    print(f"  session={session}  t0={t0}ms  t1={t1}ms  f0={f0}  f1={f1}")

    # ── 3. Load gaze + slice ──
    try:
        gaze = load_gaze_file(session)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    print(f"\n=== Gaze data for {session} ===")
    print(f"  gaze shape: {gaze.shape}")
    print(gaze[["Time", "t_sec", "Frame", "frame30", "x", "y"]].head())

    g_clip = slice_gaze_for_clip(gaze, f0, f1)
    validity = gaze_validity_ratio(g_clip)
    print(f"\n  clip gaze rows: {len(g_clip)}  valid gaze: {validity:.1%}")


if __name__ == "__main__":
    main()
