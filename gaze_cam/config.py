"""
Central configuration for Gaze-CAM project.
All paths, hyperparameters, and constants live here.
"""

from pathlib import Path
import torch

# ──────────────────────────────────────────────
# Paths  (relative to repo root)
# ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = REPO_ROOT / "data" / "EGTEA Gaze+"
ACTION_ANNOTATION_DIR = DATA_ROOT / "action_annotation"
GAZE_DATA_DIR = DATA_ROOT / "gaze_data"
VIDEO_CLIPS_DIR = DATA_ROOT / "video_clips"
CLIPS_ROOT = VIDEO_CLIPS_DIR / "cropped_clips"

RAW_ANNOTATION_CSV = ACTION_ANNOTATION_DIR / "raw_annotations" / "action_labels.csv"

# Output directory for checkpoints, predictions, etc.
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Preprocessed tensor cache (avoids re-decoding mp4 every epoch)
CACHE_DIR = REPO_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Defaults  (overridden per-model via MODEL_CONFIGS)
# ──────────────────────────────────────────────
SPLIT = 1
ARCH = "r3d18"                # default architecture
NUM_FRAMES = 16
INPUT_SIZE = 112
BATCH_SIZE = 16
EPOCHS = 30
LR = 3e-4
NUM_WORKERS = 8               # parallel data loading (uses imageio, not OpenCV)
MAX_TRAIN_ITEMS = None        # set int to limit, None for all
MAX_TEST_ITEMS = None         # set int to limit, None for all

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Per-model configuration
# ──────────────────────────────────────────────
MODEL_CONFIGS = {
    "r3d18": {
        "num_frames": 16,
        "input_size": 112,
        "batch_size": 16,
        "epochs": 30,
        "lr": 3e-4,
    },
    "slowfast_r50": {
        "num_frames": 32,          # fast pathway frames
        "slowfast_alpha": 4,       # slow = num_frames // alpha
        "input_size": 224,
        "batch_size": 16,
        "epochs": 30,
        "lr": 5e-4,
    },
    "timesformer": {
        "num_frames": 8,
        "input_size": 224,
        "batch_size": 8,
        "epochs": 30,
        "lr": 2e-5,
    },
    "vivit": {
        "num_frames": 32,
        "input_size": 224,
        "batch_size": 1,
        "accum_steps": 4,          # effective batch = 1 * 4 = 4
        "epochs": 30,
        "lr": 2e-5,
    },
}

SUPPORTED_ARCHS = list(MODEL_CONFIGS.keys())


def get_model_cfg(arch: str = ARCH) -> dict:
    """Return the per-model config dict for *arch*."""
    if arch not in MODEL_CONFIGS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {SUPPORTED_ARCHS}")
    return MODEL_CONFIGS[arch]


# ──────────────────────────────────────────────
# Gaze constants
# ──────────────────────────────────────────────
CALIBRATION_W = 1280
CALIBRATION_H = 960
VIDEO_FPS = 24.0
GAZE_HZ = 30.0
GAZE_HEADER_SKIP_ROWS = 33  # lines to skip before header in gaze .txt

# ──────────────────────────────────────────────
# Normalization (Kinetics pretrained)
# ──────────────────────────────────────────────
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


def weights_path(split: int = SPLIT, arch: str = ARCH) -> Path:
    """Path to the saved model checkpoint."""
    return OUTPUT_DIR / f"{arch}_egtea_split{split}.pt"


def predictions_path(split: int = SPLIT, arch: str = ARCH) -> Path:
    """Path to saved prediction CSV."""
    return OUTPUT_DIR / f"{arch}_split{split}_predictions.csv"


def train_split_path(split: int = SPLIT) -> Path:
    return ACTION_ANNOTATION_DIR / f"train_split{split}.txt"


def test_split_path(split: int = SPLIT) -> Path:
    return ACTION_ANNOTATION_DIR / f"test_split{split}.txt"
