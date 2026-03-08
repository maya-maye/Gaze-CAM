"""
SlowFast R50 — EGTEA fine-tuned action recognition + Grad-CAM.

Usage:
    from slowfast_model import SlowFastWrapper

    sf = SlowFastWrapper("slowfast_r50_egtea_actionlabel_split1_v3.pt", device="cuda")

    # Predict action label for a clip
    label, label_id, probs = sf.predict("path/to/clip.mp4")

    # Get Grad-CAM heatmap (T, H, W) in [0, 1]
    cam, label, label_id = sf.gradcam("path/to/clip.mp4")

    # Access internals
    sf.model          # the nn.Module
    sf.id_to_label    # {int: str}
    sf.label_to_id    # {str: int}
    sf.num_actions    # int
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────

KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KINETICS_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

NUM_FRAMES_FAST = 32
ALPHA = 4
INPUT_SIZE = 224


# ── Video preprocessing ───────────────────────────────────────────────────

def _preprocess_bcthw(x, out_size=INPUT_SIZE):
    """Resize + Kinetics normalize a (B, C, T, H, W) float tensor in [0,1]."""
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x = x.view(B, T, C, out_size, out_size).permute(0, 2, 1, 3, 4).contiguous()
    mean = KINETICS_MEAN.to(x.device, x.dtype)
    std  = KINETICS_STD.to(x.device, x.dtype)
    return (x - mean) / std


def load_clip(clip_path, num_frames_fast=NUM_FRAMES_FAST, alpha=ALPHA, out_size=INPUT_SIZE):
    """
    Read an mp4 and return SlowFast pathway tensors.

    Returns:
        slow: (3, T_slow, H, W)  e.g. (3, 8, 224, 224)
        fast: (3, T_fast, H, W)  e.g. (3, 32, 224, 224)
    """
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        raise ValueError(f"Too few frames ({len(frames)}): {clip_path}")

    frames = np.asarray(frames)
    T_total = frames.shape[0]

    fast_idx = np.linspace(0, T_total - 1, num_frames_fast).round().astype(int)
    fast_idx = np.clip(fast_idx, 0, T_total - 1)
    slow_idx = fast_idx[::alpha]

    def _to_tensor(f):
        x = torch.from_numpy(f).float() / 255.0
        x = x.permute(3, 0, 1, 2).unsqueeze(0)   # (1, 3, T, H, W)
        x = _preprocess_bcthw(x, out_size=out_size)
        return x[0]                                 # (3, T, H, W)

    return _to_tensor(frames[slow_idx]), _to_tensor(frames[fast_idx])


# ── Grad-CAM for 3D models ────────────────────────────────────────────────

class GradCAM3D:
    """Grad-CAM for any 3D conv feature map (B, C, T, H, W)."""

    def __init__(self, model, target_module):
        self.model = model
        self.acts = None
        self.grads = None
        self._hook = target_module.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def close(self):
        self._hook.remove()

    @torch.enable_grad()
    def compute(self, pathways, class_idx=None):
        """
        Returns:
            cam:       (B, T, H, W) in [0, 1]
            logits:    (B, num_actions)
            class_idx: (B,)
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(pathways)
        if class_idx is None:
            class_idx = logits.argmax(1)

        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        w = self.grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1))     # (B, T, H, W)

        flat = cam.view(cam.size(0), -1)
        flat = flat - flat.min(1, keepdim=True).values
        flat = flat / (flat.max(1, keepdim=True).values + 1e-6)
        return flat.view_as(cam).detach(), logits.detach(), class_idx.detach()


def _find_target_layer(model):
    """Find the last Conv3d in blocks[4] with 2048 out-channels (slow pathway)."""
    last_slow = last_any = None
    for name, mod in model.blocks[4].named_modules():
        if isinstance(mod, nn.Conv3d):
            last_any = mod
            if mod.weight.shape[0] == 2048:
                last_slow = mod
    return last_slow or last_any


# ── Main wrapper ───────────────────────────────────────────────────────────

class SlowFastWrapper:
    """
    One-stop interface for SlowFast inference and Grad-CAM.

    Args:
        weights_path: path to .pt checkpoint saved by the training notebook
        device: "cuda" or "cpu"
    """

    def __init__(self, weights_path, device="cuda"):
        from pytorchvideo.models.hub import slowfast_r50

        self.device = torch.device(device)
        ckpt = torch.load(weights_path, map_location=self.device)

        self.num_actions = int(ckpt["num_actions"])
        self.id_to_label = {int(k): str(v) for k, v in ckpt["id_to_label"].items()}
        self.label_to_id = {str(v): int(k) for k, v in ckpt["id_to_label"].items()}

        self.model = slowfast_r50(pretrained=False)
        self.model.blocks[6].proj = nn.Linear(2304, self.num_actions)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        # Enable gradients for Grad-CAM
        for p in self.model.parameters():
            p.requires_grad = True

        target = _find_target_layer(self.model)
        self.gcam = GradCAM3D(self.model, target)

    def _to_pathways(self, clip_path):
        """Load clip and return batched pathways on device."""
        slow, fast = load_clip(clip_path)
        return [slow.unsqueeze(0).to(self.device),
                fast.unsqueeze(0).to(self.device)]

    @torch.inference_mode()
    def predict(self, clip_path):
        """
        Returns:
            label:    str, e.g. "Put-down/Cup"
            label_id: int
            probs:    (num_actions,) numpy array of softmax probabilities
        """
        pathways = self._to_pathways(clip_path)
        logits = self.model(pathways)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        label_id = int(probs.argmax())
        return self.id_to_label.get(label_id, str(label_id)), label_id, probs

    def gradcam(self, clip_path, class_idx=None):
        """
        Returns:
            cam:      (T, H, W) numpy array in [0, 1]
            label:    str, predicted (or specified) class label
            label_id: int
        """
        pathways = self._to_pathways(clip_path)
        idx = None if class_idx is None else torch.tensor([class_idx]).to(self.device)
        cam, logits, used_cls = self.gcam.compute(pathways, class_idx=idx)
        label_id = int(used_cls[0].item())
        return (cam[0].cpu().numpy(),
                self.id_to_label.get(label_id, str(label_id)),
                label_id)

    def batch_predict(self, clip_paths):
        """
        Predict multiple clips. Returns list of (label, label_id, probs).
        """
        results = []
        for p in clip_paths:
            results.append(self.predict(p))
        return results
