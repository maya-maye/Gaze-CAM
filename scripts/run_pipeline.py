"""
run_pipeline.py  –  Run the full Gaze-CAM pipeline end-to-end.

    1. Data inspection / sanity checks
    2. Train (or load) a model on EGTEA action labels
    3. Evaluate + generate predictions CSV
    4. Run CAM and save overlay images

Usage:
    python scripts/run_pipeline.py [--model r3d18] [--split 1] [--epochs 30] [--save-cam-grids]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from gaze_cam.config import (
    SPLIT, ARCH, DEVICE,
    MAX_TRAIN_ITEMS, MAX_TEST_ITEMS,
    SUPPORTED_ARCHS, get_model_cfg,
)


def main():
    parser = argparse.ArgumentParser(description="Full Gaze-CAM pipeline")
    parser.add_argument("--model", type=str, default=ARCH, choices=SUPPORTED_ARCHS,
                        help="Model architecture")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: per-model config)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_ITEMS)
    parser.add_argument("--max-test", type=int, default=MAX_TEST_ITEMS)
    parser.add_argument("--save-cam-grids", action="store_true")
    args = parser.parse_args()

    arch = args.model
    cfg = get_model_cfg(arch)
    print(f"Device: {DEVICE}")
    print(f"Arch={arch}  Split={args.split}  "
          f"Epochs={args.epochs or cfg['epochs']}  "
          f"Batch={args.batch_size or cfg['batch_size']}\n")

    # ─── Step 1: Data inspection ───
    print("=" * 60)
    print("STEP 1: Data Inspection")
    print("=" * 60)
    from scripts.inspect_data import main as inspect_main
    inspect_main()

    # ─── Step 2: Build loaders ───
    print("\n" + "=" * 60)
    print("STEP 2: Build Datasets & Loaders")
    print("=" * 60)
    from gaze_cam.dataset import make_loaders
    train_loader, test_loader, label_to_id, id_to_label, num_actions = make_loaders(
        split=args.split,
        batch_size=args.batch_size,
        arch=arch,
        max_train=args.max_train,
        max_test=args.max_test,
    )

    # ─── Step 3: Train / load model ───
    print("\n" + "=" * 60)
    print(f"STEP 3: Train / Load {arch}")
    print("=" * 60)
    from gaze_cam.model import build_model, load_or_train
    model = build_model(arch, num_actions)
    model = load_or_train(
        model, train_loader, test_loader,
        label_to_id, id_to_label, num_actions,
        arch=arch,
        split=args.split,
    )

    # ─── Step 4: Predictions ───
    print("\n" + "=" * 60)
    print("STEP 4: Predictions")
    print("=" * 60)
    from scripts.evaluate import run_predictions
    run_predictions(model, test_loader, id_to_label, args.split, arch=arch)

    # ─── Step 5: CAM ───
    print("\n" + "=" * 60)
    print("STEP 5: CAM Analysis")
    print("=" * 60)
    from scripts.evaluate import run_gradcam_batch
    run_gradcam_batch(model, test_loader, id_to_label,
                      arch=arch, save_grids=args.save_cam_grids)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
