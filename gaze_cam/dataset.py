"""
EGTEA Gaze+ PyTorch Dataset and label-loading utilities.
Supports multi-architecture preprocessing (r3d18, slowfast_r50, timesformer, vivit).
"""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from gaze_cam.config import (
    CLIPS_ROOT,
    RAW_ANNOTATION_CSV,
    CACHE_DIR,
    NUM_FRAMES, BATCH_SIZE, NUM_WORKERS, ARCH,
    get_model_cfg,
)
from gaze_cam.gaze_utils import parse_clip_stem
from gaze_cam.video_utils import load_clip_tensor


# ──────────────────────────────────────────────
# Split file loader
# ──────────────────────────────────────────────

def load_split_stems(path: Path) -> list[str]:
    """Load clip stems from a train/test split file."""
    stems = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                stems.append(s.split()[0].strip())
    return stems


# ──────────────────────────────────────────────
# Label loading
# ──────────────────────────────────────────────

def load_action_labels(csv_path: Path = RAW_ANNOTATION_CSV):
    """
    Parse action_labels.csv and return:
      - label_to_id  : dict  str -> int
      - id_to_label  : dict  int -> str
      - prefix_to_label : dict  'session-t0-t1' -> action label str
      - sess_t0t1_to_label : dict  (session, t0, t1) -> action label str
    """
    # The header line starts with "# " so we read it manually
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().lstrip("# ").strip()
    col_names = [c.strip() for c in header_line.split(";")]

    df = pd.read_csv(
        csv_path, sep=";", engine="python",
        comment="#", header=None, names=col_names,
    )
    df.columns = [c.strip() for c in df.columns]

    COL_PREFIX = "Clip Prefix (Unique)"
    COL_SESSION = "Video Session"
    COL_T0 = "Starting Time (ms)"
    COL_T1 = "Ending Time (ms)"
    COL_LABEL = "Action Label"

    for c in [COL_PREFIX, COL_SESSION, COL_T0, COL_T1, COL_LABEL]:
        assert c in df.columns, f"Missing column '{c}'. Have: {list(df.columns)}"

    df[COL_PREFIX] = df[COL_PREFIX].astype(str).str.strip()
    df[COL_SESSION] = df[COL_SESSION].astype(str).str.strip()
    df[COL_LABEL] = df[COL_LABEL].astype(str).str.strip()
    df[COL_T0] = pd.to_numeric(df[COL_T0], errors="coerce")
    df[COL_T1] = pd.to_numeric(df[COL_T1], errors="coerce")
    df = df.dropna(subset=[COL_T0, COL_T1]).copy()
    df[COL_T0] = df[COL_T0].astype(int)
    df[COL_T1] = df[COL_T1].astype(int)

    labels = sorted(df[COL_LABEL].unique().tolist())
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    prefix_to_label = dict(zip(df[COL_PREFIX], df[COL_LABEL]))
    sess_t0t1_to_label = {
        (r[COL_SESSION], int(r[COL_T0]), int(r[COL_T1])): r[COL_LABEL]
        for _, r in df.iterrows()
    }

    return label_to_id, id_to_label, prefix_to_label, sess_t0t1_to_label


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class EGTEAActionDataset(Dataset):
    """
    Yields (clip_tensor, label_id, clip_path_str, meta_dict) for each
    clip whose stem is in the allowed split.

    *clip_tensor* format depends on *arch*:
      r3d18        → Tensor (3, T, H, W)
      slowfast_r50 → tuple  (slow, fast)
      timesformer  → Tensor (T, C, H, W)
      vivit        → Tensor (T, C, H, W)
    """

    def __init__(
        self,
        clips_root: Path,
        allowed_stems: set,
        prefix_to_label: dict,
        sess_t0t1_to_label: dict,
        label_to_id: dict,
        num_frames: int = NUM_FRAMES,
        input_size: int = 112,
        arch: str = ARCH,
        max_items: int | None = None,
        use_cache: bool = True,
    ):
        mp4s = sorted(clips_root.rglob("*.mp4"))
        mp4_by_stem = {p.stem: p for p in mp4s}

        self.items: list[tuple] = []
        self.num_frames = num_frames
        self.input_size = input_size
        self.arch = arch
        self.use_cache = use_cache
        self.cache_dir = CACHE_DIR / arch
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        miss_no_file = miss_parse = miss_label = 0

        for stem in allowed_stems:
            p = mp4_by_stem.get(stem)
            if p is None:
                miss_no_file += 1
                continue

            parsed = parse_clip_stem(stem)
            if parsed is None:
                miss_parse += 1
                continue

            session, t0, t1, f0, f1, prefix = parsed

            label = prefix_to_label.get(prefix)
            if label is None:
                label = sess_t0t1_to_label.get((session, t0, t1))
            if label is None:
                miss_label += 1
                continue

            y = label_to_id[label]
            meta = dict(stem=stem, session=session, t0=t0, t1=t1,
                        f0=f0, f1=f1, prefix=prefix, label=label)
            self.items.append((p, y, meta))

            if max_items is not None and len(self.items) >= max_items:
                break

        print(f"[dataset] kept={len(self.items)}  "
              f"miss_no_file={miss_no_file}  miss_parse={miss_parse}  miss_label={miss_label}")

        if len(self.items) == 0:
            raise ValueError(
                "No items found. Check that your video clips exist under "
                f"{clips_root} and match the split file stems."
            )

    def __len__(self):
        return len(self.items)

    def precache(self):
        """Build the tensor cache single-threaded (called before DataLoader starts)."""
        if not self.use_cache:
            return
        n = len(self.items)
        to_cache = [(i, m["stem"]) for i, (_, _, m) in enumerate(self.items)
                     if not (self.cache_dir / f"{m['stem']}.pt").exists()]
        if not to_cache:
            print(f"[cache] {self.arch}: all {n} clips already cached")
            return
        print(f"[cache] {self.arch}: caching {len(to_cache)}/{n} clips ...")
        from tqdm import tqdm
        for i, stem in tqdm(to_cache, desc=f"caching {self.arch}"):
            p, _, meta = self.items[i]
            cache_path = self.cache_dir / f"{stem}.pt"
            x = load_clip_tensor(p, num_frames=self.num_frames,
                                 input_size=self.input_size, arch=self.arch)
            # Atomic write: write temp then replace to avoid partial files
            tmp = cache_path.with_suffix(".tmp")
            torch.save(x, tmp)
            import os
            os.replace(str(tmp), str(cache_path))
        print(f"[cache] {self.arch}: done")

    def __getitem__(self, idx):
        p, y, meta = self.items[idx]

        if self.use_cache:
            cache_path = self.cache_dir / f"{meta['stem']}.pt"
            if cache_path.exists():
                x = torch.load(cache_path, weights_only=True)
            else:
                # Fallback: decode if cache miss (shouldn't happen after precache)
                x = load_clip_tensor(p, num_frames=self.num_frames,
                                     input_size=self.input_size, arch=self.arch)
        else:
            x = load_clip_tensor(p, num_frames=self.num_frames,
                                 input_size=self.input_size, arch=self.arch)

        return x, torch.tensor(y, dtype=torch.long), str(p), meta


def collate_fn(batch):
    xs, ys, paths, metas = zip(*batch)

    # SlowFast returns tuples of (slow, fast)
    if isinstance(xs[0], tuple):
        slows = torch.stack([x[0] for x in xs])
        fasts = torch.stack([x[1] for x in xs])
        return [slows, fasts], torch.stack(ys, 0), list(paths), list(metas)

    return torch.stack(xs, 0), torch.stack(ys, 0), list(paths), list(metas)


def make_loaders(
    split: int = 1,
    batch_size: int | None = None,
    arch: str = ARCH,
    max_train: int | None = None,
    max_test: int | None = None,
):
    """
    Build train and test DataLoaders for the given split and architecture.
    Returns (train_loader, test_loader, label_to_id, id_to_label, num_actions).
    """
    from gaze_cam.config import train_split_path, test_split_path

    cfg = get_model_cfg(arch)
    num_frames = cfg["num_frames"]
    input_size = cfg["input_size"]
    if batch_size is None:
        batch_size = cfg["batch_size"]

    label_to_id, id_to_label, prefix_to_label, sess_t0t1_to_label = load_action_labels()
    num_actions = len(label_to_id)

    train_stems = set(load_split_stems(train_split_path(split)))
    test_stems = set(load_split_stems(test_split_path(split)))

    clips_root = CLIPS_ROOT if CLIPS_ROOT.exists() else CLIPS_ROOT.parent
    assert clips_root.exists(), f"Clips root not found: {clips_root}"

    train_ds = EGTEAActionDataset(
        clips_root, train_stems, prefix_to_label, sess_t0t1_to_label,
        label_to_id, num_frames=num_frames, input_size=input_size,
        arch=arch, max_items=max_train,
    )
    test_ds = EGTEAActionDataset(
        clips_root, test_stems, prefix_to_label, sess_t0t1_to_label,
        label_to_id, num_frames=num_frames, input_size=input_size,
        arch=arch, max_items=max_test,
    )

    # Pre-cache all clips (single-threaded, one-time cost)
    train_ds.precache()
    test_ds.precache()

    # SlowFast tensors are very large (two pathways, 32 frames @ 224px).
    # Using fewer workers + no persistent_workers prevents gradual OOM on Windows.
    nw = min(NUM_WORKERS, 4) if arch == "slowfast_r50" else NUM_WORKERS
    _persist = nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False, prefetch_factor=2 if _persist else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=False, prefetch_factor=2 if _persist else None,
    )

    print(f"[{arch}] Train items: {len(train_ds)}  Test items: {len(test_ds)}  "
          f"Num actions: {num_actions}  "
          f"Frames: {num_frames}  Size: {input_size}")

    return train_loader, test_loader, label_to_id, id_to_label, num_actions
