"""
Grad-CAM and Attention-CAM engines for video classification models.

- GradCAM3D    : gradient-weighted activations from a Conv3d layer
                  (works with R3D-18 and SlowFast)
- AttentionCAM : CLS→patch attention from the last transformer layer
                  (works with TimeSformer and ViViT)
- build_cam_engine() : factory — returns the right engine for an arch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# GradCAM3D  (Conv-based models)
# ══════════════════════════════════════════════

class GradCAM3D:
    """
    Grad-CAM for layers that output (B, C, T, H, W) feature maps.
    Supports both single-tensor inputs (R3D-18) and list inputs (SlowFast).
    """

    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target = target_module
        self._acts = None
        self._grads = None
        self._hook_handle = self.target.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self._acts = out
        if out.requires_grad:
            out.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def close(self):
        """Remove the forward hook."""
        self._hook_handle.remove()

    @torch.enable_grad()
    def compute(self, model_input, class_idx: torch.Tensor = None):
        """
        Run forward + backward and return a normalised Grad-CAM heatmap.

        Parameters
        ----------
        model_input : Tensor (B,3,T,H,W) **or** list [slow, fast]
        class_idx   : Tensor (B,) or None  (None ⇒ predicted class)

        Returns
        -------
        cam       : Tensor (B, T, H, W)   – normalised saliency [0,1]
        logits    : Tensor (B, num_classes)
        class_idx : Tensor (B,)
        """
        # Ensure autograd graph
        if isinstance(model_input, (list, tuple)):
            model_input = [t.detach().requires_grad_(True) for t in model_input]
        else:
            model_input = model_input.detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        self._acts = None
        self._grads = None

        logits = self.model(model_input)

        if class_idx is None:
            class_idx = logits.argmax(1)

        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        score.backward()

        acts = self._acts
        grads = self._grads

        if acts is None or grads is None:
            raise RuntimeError(
                "GradCAM hooks did not capture activations/gradients. "
                "Make sure the target_module is actually used in the forward pass "
                "and that torch.no_grad / inference_mode is NOT wrapping this call."
            )

        w = grads.mean(dim=(2, 3, 4), keepdim=True)   # (B, C, 1, 1, 1)
        cam = (w * acts).sum(dim=1)                    # (B, T, H, W)
        cam = F.relu(cam)

        B, T, H, W = cam.shape
        flat = cam.view(B, -1)
        flat = flat - flat.min(dim=1, keepdim=True).values
        flat = flat / (flat.max(dim=1, keepdim=True).values + 1e-8)
        cam = flat.view(B, T, H, W)

        return cam.detach(), logits.detach(), class_idx.detach()


# ══════════════════════════════════════════════
# AttentionCAM  (Transformer-based models)
# ══════════════════════════════════════════════

class AttentionCAM:
    """
    Attention-based saliency for ViT video models (TimeSformer, ViViT).
    Uses CLS→patch attention from the last encoder layer.

    * **TimeSformer** — divided space-time attention returns per-frame
      spatial attention with shape (B*T, heads, S+1, S+1).  We extract
      CLS→patch, average over heads, and reshape to (B, T, H_p, W_p).
    * **ViViT (HF)** — the stock HuggingFace implementation discards
      attention weights, so we register a forward hook on the last
      VivitSelfAttention to capture them.  Full-sequence attention is
      (B, heads, N+1, N+1) where N = T_patches * S_patches.
    """

    def __init__(self, model: nn.Module, arch: str):
        self.model = model
        self.arch = arch

        # ── ViViT: set up hook to capture attention probs ──
        self._attn_probs = None
        self._hook_handle = None
        if arch == "vivit":
            # model.vivit.encoder.layer[-1].attention.attention is
            # VivitSelfAttention; its forward returns (output, attn_probs)
            target = model.vivit.encoder.layer[-1].attention.attention
            self._hook_handle = target.register_forward_hook(self._vivit_hook)

    def _vivit_hook(self, module, inp, out):
        """Capture the attention_probs tensor that VivitAttention discards."""
        # VivitSelfAttention.forward → (context_layer, attention_probs)
        if isinstance(out, tuple) and len(out) >= 2:
            self._attn_probs = out[1]

    def close(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ─────── helpers ───────

    def _spatial_grid(self, cfg):
        """Return (H_p, W_p) — spatial patch grid for the architecture."""
        if self.arch == "timesformer":
            ps = getattr(cfg, "patch_size", 16)
            img = getattr(cfg, "image_size", 224)
            return img // ps, img // ps
        if self.arch == "vivit":
            tub = getattr(cfg, "tubelet_size", [2, 16, 16])
            if isinstance(tub, (list, tuple)):
                ps = tub[1]
            else:
                ps = tub
            img = getattr(cfg, "image_size", 224)
            return img // ps, img // ps
        raise ValueError(self.arch)

    def _temporal_patches(self, cfg):
        """Return T_patches — number of temporal tokens."""
        if self.arch == "timesformer":
            return cfg.num_frames
        if self.arch == "vivit":
            tub = getattr(cfg, "tubelet_size", [2, 16, 16])
            t_t = tub[0] if isinstance(tub, (list, tuple)) else tub
            return cfg.num_frames // t_t
        raise ValueError(self.arch)

    # ─────── main entry point ───────

    @torch.no_grad()
    def compute(self, pixel_values: torch.Tensor, class_idx: torch.Tensor = None):
        """
        Parameters
        ----------
        pixel_values : Tensor — shape depends on arch
            timesformer : (B, C, T, H, W)
            vivit       : (B, T, C, H, W)
        class_idx    : Tensor (B,) or None

        Returns
        -------
        cam       : Tensor (B, T_out, H_out, W_out)  – normalised [0,1]
        logits    : Tensor (B, num_classes)
        class_idx : Tensor (B,)
        """
        self.model.eval()
        self._attn_probs = None  # reset for vivit hook

        out = self.model(pixel_values=pixel_values, output_attentions=True)
        logits = out.logits

        if class_idx is None:
            class_idx = logits.argmax(1)

        cfg = self.model.config
        H_p, W_p = self._spatial_grid(cfg)
        T_patches = self._temporal_patches(cfg)

        if self.arch == "timesformer":
            cam = self._cam_timesformer(out, pixel_values, H_p, W_p, T_patches)
        elif self.arch == "vivit":
            cam = self._cam_vivit(H_p, W_p, T_patches)
        else:
            raise ValueError(f"Unsupported arch: {self.arch}")

        # Normalise per sample
        B = cam.shape[0]
        flat = cam.reshape(B, -1)
        flat = flat - flat.min(dim=1, keepdim=True).values
        flat = flat / (flat.max(dim=1, keepdim=True).values + 1e-8)
        cam = flat.view(B, T_patches, H_p, W_p)

        return cam.detach(), logits.detach(), class_idx.detach()

    # ─────── arch-specific attention extraction ───────

    def _cam_timesformer(self, out, pixel_values, H_p, W_p, T_patches):
        """
        TimeSformer divided space-time attention.
        Last-layer attention shape: (B*T, heads, S+1, S+1)
        where S = H_p * W_p  (196 for 224/16).
        """
        attn = out.attentions[-1]          # (B*T, heads, S+1, S+1)
        BT, heads, seq, _ = attn.shape
        B = pixel_values.shape[0]
        T = BT // B
        S = H_p * W_p

        # CLS (index 0) → spatial patches (indices 1..S)
        cls_attn = attn[:, :, 0, 1:S + 1]  # (B*T, heads, S)
        cls_attn = cls_attn.mean(dim=1)     # (B*T, S)
        cam = cls_attn.view(B, T, H_p, W_p)

        # T might differ from T_patches if model pads frames; trim/pad
        if T != T_patches:
            cam = cam[:, :T_patches]
        return cam

    def _cam_vivit(self, H_p, W_p, T_patches):
        """
        ViViT full-sequence attention (captured via hook).
        Attention shape: (B, heads, N+1, N+1)
        where N = T_patches * H_p * W_p.
        """
        attn = self._attn_probs
        if attn is None:
            raise RuntimeError(
                "ViViT attention hook captured nothing. Make sure the model "
                "has a VivitSelfAttention layer and the hook is registered."
            )

        # (B, heads, N+1, N+1)
        S = H_p * W_p
        N = T_patches * S
        cls_attn = attn[:, :, 0, 1:N + 1]   # (B, heads, N)
        cls_attn = cls_attn.mean(dim=1)      # (B, N)
        B = cls_attn.shape[0]
        cam = cls_attn.view(B, T_patches, H_p, W_p)
        return cam


# ══════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════

def build_cam_engine(arch: str, model: nn.Module):
    """
    Return the appropriate CAM engine for *arch*.
    For conv models returns GradCAM3D; for transformers returns AttentionCAM.
    """
    if arch in ("timesformer", "vivit"):
        return AttentionCAM(model, arch)

    # Conv models: r3d18, slowfast_r50
    from gaze_cam.model import get_target_layer
    target = get_target_layer(arch, model)
    if target is None:
        raise RuntimeError(f"No GradCAM target layer found for arch '{arch}'")
    return GradCAM3D(model, target)


# ──────────────────────────────────────────────
# Convenience: upsample CAM to original frame resolution
# ──────────────────────────────────────────────

def upsample_cam_to_frames(cam_thw: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """
    Upsample a single-sample CAM (T, H, W) to (T, target_h, target_w) numpy array.
    """
    cam = cam_thw.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    cam = F.interpolate(cam, size=(cam_thw.shape[0], target_h, target_w),
                        mode="trilinear", align_corners=False)
    return cam.squeeze().cpu().numpy()


# ──────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────

def cam_overlay_on_frame(frame_rgb: np.ndarray, cam_2d: np.ndarray, alpha: float = 0.4):
    """
    Overlay a 2-D heatmap (H, W) in [0, 1] on an RGB frame.
    Returns an RGB uint8 image.
    """
    heatmap = (cam_2d * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if frame_rgb.shape[:2] != cam_2d.shape[:2]:
        heatmap = cv2.resize(heatmap, (frame_rgb.shape[1], frame_rgb.shape[0]))

    blended = (frame_rgb.astype(float) * (1 - alpha) + heatmap.astype(float) * alpha)
    return blended.clip(0, 255).astype(np.uint8)


def save_cam_grid(
    frames_rgb: list[np.ndarray],
    cam_maps: list[np.ndarray],
    titles: list[str],
    out_path: Path,
    cols: int = 4,
):
    """Save a grid of frames with CAM overlay."""
    n = len(frames_rgb)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        if i < n:
            overlay = cam_overlay_on_frame(frames_rgb[i], cam_maps[i])
            axes[i].imshow(overlay)
            axes[i].set_title(titles[i], fontsize=9)
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"Saved CAM grid -> {out_path}")
