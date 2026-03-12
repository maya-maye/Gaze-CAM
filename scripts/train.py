"""
Train a video model on EGTEA Gaze+ action labels.

Usage:
    python scripts/train.py [--model r3d18] [--split 1] [--epochs 30] [--batch-size 16]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from gaze_cam.config import (
    SPLIT, ARCH, DEVICE, MAX_TRAIN_ITEMS, MAX_TEST_ITEMS,
    SUPPORTED_ARCHS, get_model_cfg,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model, train_model, eval_top1


def main():
    parser = argparse.ArgumentParser(description="Train a model on EGTEA action labels")
    parser.add_argument("--model", type=str, default=ARCH, choices=SUPPORTED_ARCHS,
                        help="Model architecture")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: per-model config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: per-model config)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: per-model config)")
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_ITEMS)
    parser.add_argument("--max-test", type=int, default=MAX_TEST_ITEMS)
    args = parser.parse_args()

    arch = args.model
    cfg = get_model_cfg(arch)

    max_train = args.max_train if args.max_train and args.max_train > 0 else None
    max_test = args.max_test if args.max_test and args.max_test > 0 else None

    print(f"Device: {DEVICE}")
    print(f"Arch: {arch}  Split: {args.split}  "
          f"Epochs: {args.epochs or cfg['epochs']}  "
          f"Batch: {args.batch_size or cfg['batch_size']}  "
          f"LR: {args.lr or cfg['lr']}")

    # Build loaders
    train_loader, test_loader, label_to_id, id_to_label, num_actions = make_loaders(
        split=args.split,
        batch_size=args.batch_size,
        arch=arch,
        max_train=max_train,
        max_test=max_test,
    )

    # Build model
    model = build_model(arch, num_actions)

    # Train
    model = train_model(
        model, train_loader, test_loader,
        label_to_id, id_to_label, num_actions,
        arch=arch,
        split=args.split,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Final accuracy
    acc = eval_top1(model, test_loader, arch)
    print(f"\nFinal test top-1 accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
