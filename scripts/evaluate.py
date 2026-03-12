"""
Evaluate a trained model and run Grad-CAM / Attention-CAM analysis.

Usage:
    python scripts/evaluate.py [--model r3d18] [--split 1] [--max-batches 5] [--save-cam-grids]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np

from gaze_cam.config import (
    SPLIT, ARCH, DEVICE, OUTPUT_DIR,
    MAX_TEST_ITEMS, predictions_path,
    SUPPORTED_ARCHS, get_model_cfg,
)
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model, load_or_train, predict_table
from gaze_cam.gradcam import build_cam_engine, upsample_cam_to_frames, save_cam_grid


def run_predictions(model, test_loader, id_to_label, split, arch=ARCH, max_batches=5):
    """Run inference and save a predictions CSV."""
    df = predict_table(model, test_loader, id_to_label, arch=arch, max_batches=max_batches)
    print(df.head(10).to_string(index=False))
    print(f"\nAccuracy (sampled): {df['correct'].mean():.4f}")

    out_csv = predictions_path(split, arch)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions -> {out_csv}")
    return df


def run_gradcam_batch(model, test_loader, id_to_label, arch=ARCH, save_grids=False):
    """
    Run Grad-CAM / Attention-CAM on one batch from the test set and print results.
    Optionally saves overlay images.
    """
    cam_engine = build_cam_engine(arch, model)
    print(f"CAM engine: {type(cam_engine).__name__} for {arch}")

    model.eval()

    x, y, paths, metas = next(iter(test_loader))
    if isinstance(x, (list, tuple)):
        x = [t.to(DEVICE) for t in x]
    else:
        x = x.to(DEVICE)

    cam_bthw, logits, used_cls = cam_engine.compute(x, class_idx=None)
    pred = logits.argmax(1).cpu()
    y_cpu = y.cpu()

    print(f"\nCAM shape   : {tuple(cam_bthw.shape)}")

    for i in range(min(8, len(paths))):
        stem = metas[i]["stem"]
        gt_id = int(y_cpu[i].item())
        pr_id = int(pred[i].item())
        print(
            f"  {i:02d}  {stem}  "
            f"gt={id_to_label.get(gt_id, gt_id)}  "
            f"pred={id_to_label.get(pr_id, pr_id)}  "
            f"correct={gt_id == pr_id}"
        )

    # Save CAM overlay grids
    if save_grids:
        cam_dir = OUTPUT_DIR / "cam_grids"
        cam_dir.mkdir(exist_ok=True)

        for i in range(min(4, len(paths))):
            clip_path = Path(paths[i])
            cap = cv2.VideoCapture(str(clip_path))
            frames_rgb = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames_rgb) == 0:
                continue

            h, w = frames_rgb[0].shape[:2]
            cam_i = cam_bthw[i]
            cam_up = upsample_cam_to_frames(cam_i, h, w)

            T_cam = cam_up.shape[0]
            T_vid = len(frames_rgb)
            select = np.linspace(0, min(T_cam, T_vid) - 1, min(8, T_cam)).astype(int)

            sel_frames, sel_cams, sel_titles = [], [], []

            for t in select:
                vid_t = int(t * T_vid / T_cam) if T_cam != T_vid else t
                vid_t = min(vid_t, T_vid - 1)
                sel_frames.append(frames_rgb[vid_t])
                sel_cams.append(cam_up[t])

                stem = metas[i]["stem"]
                gt_str = id_to_label.get(int(y_cpu[i].item()), "?")
                pr_str = id_to_label.get(int(pred[i].item()), "?")
                sel_titles.append(f"t={t} gt={gt_str}\npred={pr_str}")

            save_cam_grid(
                sel_frames, sel_cams, sel_titles,
                cam_dir / f"cam_{metas[i]['stem']}.png",
            )

    cam_engine.close()
    return cam_bthw


def main():
    parser = argparse.ArgumentParser(description="Evaluate + CAM analysis")
    parser.add_argument("--model", type=str, default=ARCH, choices=SUPPORTED_ARCHS,
                        help="Model architecture")
    parser.add_argument("--split", type=int, default=SPLIT)
    parser.add_argument("--max-batches", type=int, default=5,
                        help="Max batches for prediction table (None = all)")
    parser.add_argument("--save-cam-grids", action="store_true",
                        help="Save CAM overlay images to outputs/cam_grids/")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=MAX_TEST_ITEMS)
    args = parser.parse_args()

    arch = args.model
    print(f"Device: {DEVICE}  Arch: {arch}")

    train_loader, test_loader, label_to_id, id_to_label, num_actions = make_loaders(
        split=args.split,
        batch_size=args.batch_size,
        arch=arch,
        max_test=args.max_test,
    )

    model = build_model(arch, num_actions)
    model = load_or_train(
        model, train_loader, test_loader,
        label_to_id, id_to_label, num_actions,
        arch=arch,
        split=args.split,
    )

    # Predictions
    print("\n=== Predictions ===")
    run_predictions(model, test_loader, id_to_label, args.split,
                    arch=arch, max_batches=args.max_batches)

    # CAM
    print("\n=== CAM Analysis ===")
    run_gradcam_batch(model, test_loader, id_to_label,
                      arch=arch, save_grids=args.save_cam_grids)

    print("\nDone.")


if __name__ == "__main__":
    main()
