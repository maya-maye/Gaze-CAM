"""Quick script: find which 'Put eating_utensil' clips SlowFast gets correct."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from gaze_cam.config import DEVICE, weights_path
from gaze_cam.dataset import make_loaders
from gaze_cam.model import build_model
from gaze_cam.gradcam import build_cam_engine

arch = "slowfast_r50"
split = 1

_, test_loader, label_to_id, id_to_label, na = make_loaders(
    split=split, batch_size=4, arch=arch, max_train=2, max_test=None,
)

model = build_model(arch, na)
ckpt = torch.load(weights_path(split, arch), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=True)
id_to_label.update(ckpt.get("id_to_label", {}))
model.to(DEVICE).eval()

print(f"Device: {DEVICE}, scanning {len(test_loader.dataset)} clips for 'Put eating_utensil'...\n")

results = []
for x_batch, y_batch, paths, metas in test_loader:
    if isinstance(x_batch, (list, tuple)):
        x_batch = [t.to(DEVICE) for t in x_batch]
    else:
        x_batch = x_batch.to(DEVICE)
    with torch.no_grad():
        logits = model(x_batch)
    preds = logits.argmax(1).cpu()
    for i in range(len(paths)):
        gt_id = int(y_batch[i].item())
        pr_id = int(preds[i].item())
        gt_label = id_to_label.get(gt_id, str(gt_id))
        if "put" in gt_label.lower() and "eating_utensil" in gt_label.lower():
            pr_label = id_to_label.get(pr_id, str(pr_id))
            correct = gt_id == pr_id
            mark = "CORRECT" if correct else "WRONG"
            results.append((mark, gt_label, pr_label, metas[i]["stem"]))
            print(f"  {mark:8s} GT={gt_label:25s} Pred={pr_label:25s} {metas[i]['stem']}")

print(f"\nTotal 'Put eating_utensil' clips: {len(results)}")
print(f"Correct: {sum(1 for r in results if r[0]=='CORRECT')}")
