# Presentation Slides

## Images
- All figures are in `docs/slides/figures/`
- All GIFs are in `docs/slides/gifs/`
- GIFs animate natively in Google Slides and PowerPoint (just drag them in)

---

## SLIDE 1 — TITLE

Do Video Models Look Where Humans Look?
Comparing Grad-CAM Saliency with Eye-Gaze on EGTEA Gaze+

Author 1, Author 2
University of California, Santa Barbara

---

## SLIDE 2 — MOTIVATION

- Video models achieve high accuracy — but what evidence do they use?
- A correct prediction might rely on background cues, not the action itself
- Human eye-gaze gives us ground truth for task-relevant attention
- Key question: Do model saliency maps align with where humans actually look?

IMAGE: one of the GIFs from gifs/ folder (e.g. clip_00_P13-R01-PastaSalad)

---

## SLIDE 3 — TEASER

Overview: Saliency vs. Gaze Across Models & Actions

Green circle = human gaze | Heatmap = model saliency | ✓ correct | ✗ incorrect

IMAGE: fig_teaser.png (full width)

---

## SLIDE 4 — DATASET

Dataset: EGTEA Gaze+

- Egocentric cooking videos (head-mounted camera)
- 86 sessions, 7 recipes
- 15,484 clips, 1,170 action classes (verb–noun)
- Eye-gaze recorded at 30 Hz
- Video at 24 fps, 640 × 480
- Split 1: 8,299 train / 2,022 test
- Subject-disjoint splits
- Gaze used only for evaluation — models never see gaze during training
- ~490–498 clips per model after gaze filtering

---

## SLIDE 5 — FOUR ARCHITECTURES

| Model | Type | Frames | Resolution | Saliency Method |
|---|---|---|---|---|
| R3D-18 | 3D CNN | 16 | 112² | Grad-CAM (layer4) |
| SlowFast R50 | Dual-path CNN | 32 | 224² | Grad-CAM (slow path) |
| TimeSformer | Transformer | 8 | 224² | Attention rollout |
| ViViT | Transformer | 32 | 224² | Attention rollout |

- All pretrained on Kinetics-400, fine-tuned on EGTEA Gaze+ for 30 epochs
- CNNs: Grad-CAM on last conv layer
- Transformers: CLS-to-patch attention from last encoder layer

---

## SLIDE 6 — PIPELINE

Video Clip → Model → Prediction + Saliency Map

Gaze Recording → Gaze Coordinates

↓

Compare saliency map vs. gaze location per frame

↓

NSS, AUC, KL Divergence

---

## SLIDE 7 — METRICS

NSS (Normalized Scanpath Saliency)
- Z-scored saliency value at the gaze point
- NSS = 0 means chance, higher = better

AUC-Judd
- Fraction of map with lower saliency than gaze point
- 0.5 = chance, 1.0 = perfect

KL Divergence
- Distributional distance between saliency and fixation map
- Lower = better

Center-bias baseline: fixed 2D Gaussian at image center → NSS ≈ 0.87–0.89, AUC ≈ 0.73

---

## SLIDE 8 — CLASSIFICATION ACCURACY

| Model | Accuracy (%) | Test Clips |
|---|---|---|
| R3D-18 | 48.7 | 493 |
| SlowFast R50 | 59.8 | 498 |
| TimeSformer | 60.0 | 493 |
| ViViT | **64.6** | 494 |

- Transformers outperform CNNs on 1,170 fine-grained actions
- ViViT best accuracy, R3D-18 lowest
- But does higher accuracy = better gaze alignment?

---

## SLIDE 9 — GAZE–SALIENCY ALIGNMENT

| Model | NSS | AUC | KL ↓ | CB NSS |
|---|---|---|---|---|
| R3D-18 | 1.079 | 0.774 | 3.37 | 0.867 |
| SlowFast R50 | **0.206** | **0.566** | 3.54 | 0.846 |
| TimeSformer | **1.469** | **0.799** | 3.18 | 0.911 |
| ViViT | 1.017 | 0.783 | 3.13 | 0.906 |

- TimeSformer: best gaze alignment (NSS = 1.47)
- SlowFast: BELOW center-bias baseline despite 59.8% accuracy
- Accuracy ≠ gaze alignment

---

## SLIDE 10 — ACCURACY VS. ALIGNMENT

The Dissociation

SlowFast: 2nd-highest accuracy, worst gaze alignment by far.

IMAGE: fig_comparison_split1.png

---

## SLIDE 11 — QUALITATIVE EXAMPLE

Row 1: Raw frames + gaze | Row 2: Grad-CAM heatmap | Row 3: Overlay

For live demo: play the animated GIFs from the gifs/ folder!

IMAGE: sample_strip.png
GIF: any clip GIF for live demo

---

## SLIDE 12 — CORRECT VS. INCORRECT PREDICTIONS

| Model | ΔNSS | p |
|---|---|---|
| R3D-18 | +0.26 | <0.001 |
| SlowFast | +0.06 | 0.314 |
| TimeSformer | +0.44 | 0.005 |
| ViViT | +0.12 | 0.211 |

- R3D-18 & TimeSformer: correct predictions → higher gaze alignment
- SlowFast & ViViT: not significant

IMAGE: fig_correct_vs_incorrect.png

---

## SLIDE 13 — ERROR TYPE BREAKDOWN

- Correct verb → high alignment, even if noun is wrong
- Verb recognition is more closely tied to gaze than noun recognition

IMAGE: fig_error_buckets.png

---

## SLIDE 14 — CENTER-BIAS BASELINE

| Model | Model NSS | CB NSS | t | p |
|---|---|---|---|---|
| R3D-18 | 1.079 | 0.867 | 5.13 | <0.001 |
| SlowFast R50 | 0.206 | 0.846 | -12.15 | <0.001 |
| TimeSformer | 1.469 | 0.911 | 6.47 | <0.001 |
| ViViT | 1.017 | 0.906 | 1.89 | 0.059 |

- R3D-18 & TimeSformer significantly exceed center bias
- SlowFast significantly below center bias
- ViViT: marginal (p = 0.059)

---

## SLIDE 15 — OCCLUSION SENSITIVITY

Are saliency maps faithful?

| Model | High-CAM Δp | Low-CAM Δp | t | p |
|---|---|---|---|---|
| R3D-18 | 0.088 | 0.006 | 2.15 | 0.036 |
| SlowFast R50 | -0.002 | -0.003 | 0.08 | 0.937 |
| TimeSformer | **0.209** | 0.002 | 3.58 | <0.001 |
| ViViT | 0.081 | 0.013 | 1.66 | 0.104 |

- Mask top-25% CAM region vs. bottom-25%
- TimeSformer: masking high-CAM drops confidence by 0.21 — faithful
- SlowFast: no effect — saliency is not causally meaningful

---

## SLIDE 16 — MODEL RANDOMIZATION

Sanity check: trained vs. randomly initialized model

| Model | Trained NSS | Random NSS |
|---|---|---|
| R3D-18 | 1.079 | 1.097 |
| SlowFast R50 | 0.206 | 1.026 |

- R3D-18: random ≈ trained → CNN inductive bias contributes to alignment
- This is why center-bias and occlusion controls are essential

---

## SLIDE 17 — TEMPORAL LEAD/LAG

- Shift gaze by δ ∈ [-5, +5] frames
- Peak at δ > 0 = model "leads" gaze
- All models: minimal temporal effect (<0.04 NSS range)
- Models process whole clips → temporally averaged attention

IMAGE: fig_temporal_lag.png

---

## SLIDE 18 — PER-VERB ALIGNMENT

- Localized actions (Cut, Stir) → higher alignment
- Distributed actions (Move, Operate) → lower alignment

IMAGE: fig_verb_alignment.png

---

## SLIDE 19 — KEY TAKEAWAYS

1. Accuracy ≠ interpretability. SlowFast achieves 59.8% accuracy but falls below a center-bias baseline in gaze alignment.

2. TimeSformer wins on alignment. Highest NSS (1.47), passes occlusion test, exceeds center bias. Its attention maps are both accurate and meaningful.

3. Correct predictions correlate with gaze for R3D-18 and TimeSformer — but not universally.

4. CNN inductive bias inflates raw scores. Randomization control shows R3D-18's alignment partly comes from architecture, not learning.

5. Verb matters more than noun for predicting gaze alignment.

---

## SLIDE 20 — LIMITATIONS & FUTURE WORK

Limitations:
- Single split (Split 1 only)
- Grad-CAM on slow pathway only for SlowFast
- Single gaze point per frame
- Strong center-bias confound in egocentric video

Future Directions:
- Alternative saliency methods (integrated gradients, SHAP)
- Multi-split evaluation
- Gaze as a training signal, not just evaluation
- Frame-level temporal gaze modeling

Questions?

---

## IMAGE PLACEMENT SUMMARY

| Slide | Image/GIF |
|---|---|
| 2 | GIF (any clip from gifs/) |
| 3 | fig_teaser.png |
| 10 | fig_comparison_split1.png |
| 11 | sample_strip.png + GIF for live demo |
| 12 | fig_correct_vs_incorrect.png |
| 13 | fig_error_buckets.png |
| 17 | fig_temporal_lag.png |
| 18 | fig_verb_alignment.png |
