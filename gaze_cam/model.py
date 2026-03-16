"""
Multi-architecture model construction, training loop, and evaluation.

Supported architectures:
  - r3d18        : torchvision R3D-18 (3-D ResNet)
  - slowfast_r50 : pytorchvideo SlowFast R-50
  - timesformer  : HuggingFace TimeSformer-Base
  - vivit        : HuggingFace ViViT-B-16x2
"""

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from gaze_cam.config import (
    DEVICE, EPOCHS, LR, SPLIT, ARCH,
    weights_path, get_model_cfg, SUPPORTED_ARCHS,
)


# ──────────────────────────────────────────────
# Model builder
# ──────────────────────────────────────────────

def build_model(
    arch: str = ARCH,
    num_actions: int = 106,
    device: str = DEVICE,
) -> nn.Module:
    """Build a pretrained model and replace its head for *num_actions* classes."""

    if arch == "r3d18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_actions)

    elif arch == "slowfast_r50":
        from pytorchvideo.models.hub import slowfast_r50
        model = slowfast_r50(pretrained=True)
        # Head is blocks[6].proj  (in_features=2304)
        model.blocks[6].proj = nn.Linear(2304, num_actions)

    elif arch == "timesformer":
        from transformers import TimesformerForVideoClassification
        model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=num_actions,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",   # need real attn probs for CAM
        )

    elif arch == "vivit":
        from transformers import VivitForVideoClassification
        model = VivitForVideoClassification.from_pretrained(
            "google/vivit-b-16x2-kinetics400",
            num_labels=num_actions,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",   # SDPA discards attn probs
        )

    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {SUPPORTED_ARCHS}")

    return model.to(device)


# ──────────────────────────────────────────────
# Architecture helpers
# ──────────────────────────────────────────────

def get_target_layer(arch: str, model: nn.Module):
    """Return the module to hook for Grad-CAM (or *None* for transformers)."""
    if arch == "r3d18":
        return model.layer4[-1].conv2

    elif arch == "slowfast_r50":
        # Last Conv3d with 2048 out-channels in the slow pathway (blocks[4])
        last_slow = None
        for _name, mod in model.blocks[4].named_modules():
            if isinstance(mod, nn.Conv3d) and mod.weight.shape[0] == 2048:
                last_slow = mod
        if last_slow is None:
            # Fallback: last Conv3d anywhere in blocks[4]
            for _name, mod in model.blocks[4].named_modules():
                if isinstance(mod, nn.Conv3d):
                    last_slow = mod
        return last_slow

    elif arch in ("timesformer", "vivit"):
        return None  # transformer models use attention-based CAM

    raise ValueError(f"Unknown arch '{arch}'")


def model_forward(arch: str, model: nn.Module, x) -> torch.Tensor:
    """
    Unified forward pass that always returns a logits tensor.

    *x* can be:
      - Tensor (B, C, T, H, W)           for r3d18
      - list [slow_tensor, fast_tensor]   for slowfast_r50
      - Tensor (B, T, C, H, W)           for timesformer / vivit
    """
    if arch in ("timesformer", "vivit"):
        out = model(pixel_values=x)
        return out.logits
    else:
        # r3d18 or slowfast_r50 (list input is handled natively by slowfast)
        return model(x)


def replace_head(arch: str, model: nn.Module, num_actions: int, device: str = DEVICE):
    """Replace the classification head in-place (used when loading a checkpoint
    whose num_actions differs from the current label set)."""
    if arch == "r3d18":
        model.fc = nn.Linear(model.fc.in_features, num_actions).to(device)
    elif arch == "slowfast_r50":
        model.blocks[6].proj = nn.Linear(2304, num_actions).to(device)
    elif arch == "timesformer":
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_actions).to(device)
        model.config.num_labels = num_actions
    elif arch == "vivit":
        in_f = model.classifier.in_features
        model.classifier = nn.Linear(in_f, num_actions).to(device)
        model.config.num_labels = num_actions


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.inference_mode()
def eval_top1(model: nn.Module, loader, arch: str = ARCH, device: str = DEVICE) -> float:
    """Return top-1 accuracy on *loader*."""
    model.eval()
    correct = total = 0
    for x, y, _, _ in loader:
        x = _to_device(x, device)
        y = y.to(device, non_blocking=True)
        logits = model_forward(arch, model, x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.inference_mode()
def predict_table(
    model: nn.Module,
    loader,
    id_to_label: dict,
    arch: str = ARCH,
    device: str = DEVICE,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """
    Run inference on *loader* and return a DataFrame with columns:
    stem, path, gt_id, pred_id, correct, gt_str, pred_str
    """
    model.eval()
    rows = []
    t0 = time.time()

    for bi, (x, y, paths, metas) in enumerate(tqdm(loader, desc="inference")):
        if bi == 0:
            print(f"  First batch arrived after {time.time() - t0:.2f}s")

        x = _to_device(x, device)
        logits = model_forward(arch, model, x)
        pred = logits.argmax(1).cpu()
        y_cpu = y.cpu()

        for i in range(len(paths)):
            gt_id = int(y_cpu[i].item())
            pr_id = int(pred[i].item())
            rows.append(dict(
                stem=metas[i]["stem"],
                path=paths[i],
                gt_id=gt_id,
                pred_id=pr_id,
                correct=int(gt_id == pr_id),
                gt_str=id_to_label.get(gt_id, str(gt_id)),
                pred_str=id_to_label.get(pr_id, str(pr_id)),
            ))

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader,
    test_loader,
    label_to_id: dict,
    id_to_label: dict,
    num_actions: int,
    arch: str = ARCH,
    split: int = SPLIT,
    epochs: int | None = None,
    lr: float | None = None,
    device: str = DEVICE,
) -> nn.Module:
    """
    Fine-tune *model* on EGTEA action labels.
    If *epochs* or *lr* are None, use the per-model defaults from MODEL_CONFIGS.
    """
    cfg = get_model_cfg(arch)
    if epochs is None:
        epochs = cfg["epochs"]
    if lr is None:
        lr = cfg["lr"]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(device == "cuda"))
    use_amp = (device == "cuda")

    # Enable TF32 for ~15% free speed on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Resume from checkpoint if it exists
    start_epoch = 1
    ckpt_path = weights_path(split, arch)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
        start_epoch = ckpt.get("epoch", 0) + 1
        prev_acc = ckpt.get("test_top1", 0)
        print(f"  Resuming from epoch {start_epoch} (prev acc {prev_acc:.4f})")

    # Gradient accumulation: ViViT needs batch=1 on GPU to avoid OOM,
    # but we simulate a larger effective batch by accumulating gradients.
    accum_steps = cfg.get("accum_steps", 1)
    if accum_steps > 1:
        print(f"  Gradient accumulation: {accum_steps} steps")

    print(f"\nTraining {arch} on {device} for epochs {start_epoch}-{epochs} ...")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)

        for step, (x, y, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            x = _to_device(x, device)
            y = y.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                logits = model_forward(arch, model, x)
                loss = F.cross_entropy(logits, y)
                if accum_steps > 1:
                    loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += loss.item() * y.size(0) * (accum_steps if accum_steps > 1 else 1)
            seen += y.size(0)

        acc = eval_top1(model, test_loader, arch, device)
        print(f"  Epoch {epoch}: train_loss={running / max(1, seen):.4f}  test_top1={acc:.4f}")

        # Prevent gradual memory buildup that causes Windows to kill the process
        gc.collect()
        torch.cuda.empty_cache()

        # Save checkpoint every epoch so progress is never lost
        out = weights_path(split, arch)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            model_state=model.state_dict(),
            num_actions=num_actions,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            arch=arch,
            split=split,
            epoch=epoch,
            test_top1=acc,
        ), out)
        print(f"  Checkpoint saved -> {out}  (epoch {epoch}, acc {acc:.4f})")

    return model


def load_or_train(
    model: nn.Module,
    train_loader,
    test_loader,
    label_to_id: dict,
    id_to_label: dict,
    num_actions: int,
    arch: str = ARCH,
    split: int = SPLIT,
    device: str = DEVICE,
) -> nn.Module:
    """
    If a checkpoint already exists for this arch/split, load it.
    Otherwise train from scratch.
    """
    ckpt_path = weights_path(split, arch)

    if ckpt_path.exists():
        print(f"\nFound checkpoint, loading: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        ckpt_num = int(ckpt.get("num_actions", num_actions))
        if ckpt_num != num_actions:
            print(f"  WARNING: checkpoint num_actions={ckpt_num} vs current={num_actions}. Using checkpoint.")
            num_actions = ckpt_num
            replace_head(arch, model, num_actions, device)

        model.load_state_dict(ckpt["model_state"], strict=True)

        # Restore label maps from checkpoint
        label_to_id.update(ckpt.get("label_to_id", {}))
        id_to_label.update(ckpt.get("id_to_label", {}))

        acc = eval_top1(model, test_loader, arch, device)
        print(f"  Loaded. test_top1={acc:.4f}")
        return model

    return train_model(
        model, train_loader, test_loader,
        label_to_id, id_to_label, num_actions,
        arch=arch, split=split, device=device,
    )


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _to_device(x, device: str):
    """Move model input to *device*. Handles both tensors and [slow, fast] lists."""
    if isinstance(x, (list, tuple)):
        return [t.to(device, non_blocking=True) for t in x]
    return x.to(device, non_blocking=True)
