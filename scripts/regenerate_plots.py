#!/usr/bin/env python3
"""
regenerate_plots.py — Regenerate all figures from existing CSVs without re-running inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from gaze_cam.config import OUTPUT_DIR, SUPPORTED_ARCHS


ANALYSIS_DIR = OUTPUT_DIR / "analysis"
COLORS = {"r3d18": "#2196F3", "slowfast_r50": "#4CAF50",
          "timesformer": "#FF9800", "vivit": "#9C27B0"}


def plot_correct_vs_incorrect_combined(all_dfs, out_dir):
    """Bar chart: correct vs incorrect NSS for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(all_dfs.keys())
    x = np.arange(len(models))
    width = 0.35
    
    correct_nss = []
    incorrect_nss = []
    correct_std = []
    incorrect_std = []
    
    for arch in models:
        df = all_dfs[arch]
        c = df[df["correct"] == 1]["nss"].dropna()
        ic = df[df["correct"] == 0]["nss"].dropna()
        correct_nss.append(c.mean())
        incorrect_nss.append(ic.mean())
        correct_std.append(c.std() / np.sqrt(len(c)))  # SE
        incorrect_std.append(ic.std() / np.sqrt(len(ic)))
    
    bars1 = ax.bar(x - width/2, correct_nss, width, yerr=correct_std,
                   label='Correct', color='#4CAF50', capsize=5)
    bars2 = ax.bar(x + width/2, incorrect_nss, width, yerr=incorrect_std,
                   label='Incorrect', color='#f44336', capsize=5)
    
    ax.set_ylabel('Mean NSS')
    ax.set_title('Gaze–CAM Alignment: Correct vs Incorrect Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Set y-axis limit with headroom for annotations
    ymax = max(max(correct_nss), max(incorrect_nss)) + 0.35
    ax.set_ylim(top=ymax)
    
    # Add significance stars
    for i, arch in enumerate(models):
        df = all_dfs[arch]
        c = df[df["correct"] == 1]["nss"].dropna()
        ic = df[df["correct"] == 0]["nss"].dropna()
        if len(c) >= 5 and len(ic) >= 5:
            _, p = stats.mannwhitneyu(c, ic, alternative="two-sided")
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                star = "n.s."
            y = max(correct_nss[i], incorrect_nss[i]) + 0.15
            ax.text(i, y, star, ha='center', fontsize=12)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_correct_vs_incorrect.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_correct_vs_incorrect.png'}")


def plot_error_buckets_combined(all_dfs, out_dir):
    """Grouped bar chart: NSS by error bucket for all models."""
    buckets = ["correct", "right_verb_wrong_noun", "wrong_verb_right_noun", "completely_wrong"]
    bucket_labels = ["Correct", "Right V\nWrong N", "Wrong V\nRight N", "Completely\nWrong"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(all_dfs.keys())
    x = np.arange(len(buckets))
    width = 0.2
    offsets = np.linspace(-1.5*width, 1.5*width, len(models))
    
    for i, arch in enumerate(models):
        df = all_dfs[arch]
        means = []
        stds = []
        for bucket in buckets:
            sub = df[df["bucket"] == bucket]["nss"].dropna()
            means.append(sub.mean() if len(sub) > 0 else 0)
            stds.append(sub.std() / np.sqrt(len(sub)) if len(sub) > 1 else 0)
        ax.bar(x + offsets[i], means, width, yerr=stds, label=arch,
               color=COLORS[arch], capsize=3)
    
    ax.set_ylabel('Mean NSS')
    ax.set_title('Gaze–CAM Alignment by Error Type')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.legend(loc='upper right')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_error_buckets.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_error_buckets.png'}")


def plot_verb_alignment_combined(all_dfs, out_dir, top_n=10):
    """Horizontal bar chart: top/bottom verbs by NSS (averaged across models)."""
    # Combine all dataframes
    combined = pd.concat([df.assign(model=arch) for arch, df in all_dfs.items()])
    
    verb_stats = combined.groupby("gt_verb")["nss"].agg(["mean", "std", "count"])
    verb_stats = verb_stats[verb_stats["count"] >= 20]  # Need at least 20 samples
    verb_stats = verb_stats.sort_values("mean")
    
    # Get top and bottom verbs
    bottom = verb_stats.head(top_n)
    top = verb_stats.tail(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bottom verbs (lowest NSS)
    ax1.barh(range(len(bottom)), bottom["mean"], xerr=bottom["std"]/np.sqrt(bottom["count"]),
             color='#f44336', capsize=3)
    ax1.set_yticks(range(len(bottom)))
    ax1.set_yticklabels(bottom.index)
    ax1.set_xlabel('Mean NSS')
    ax1.set_title(f'Bottom {top_n} Verbs by Alignment')
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Top verbs (highest NSS)
    ax2.barh(range(len(top)), top["mean"], xerr=top["std"]/np.sqrt(top["count"]),
             color='#4CAF50', capsize=3)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top.index)
    ax2.set_xlabel('Mean NSS')
    ax2.set_title(f'Top {top_n} Verbs by Alignment')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle('Per-Verb Gaze–CAM Alignment (All Models)', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_verb_alignment.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_verb_alignment.png'}")


def plot_accuracy_table(all_dfs, out_dir):
    """Classification accuracy breakdown: overall, verb, noun."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    models = list(all_dfs.keys())
    
    data = []
    for arch in models:
        df = all_dfs[arch]
        overall = 100 * df["correct"].mean()
        verb_acc = 100 * (df["gt_verb"] == df["pred_verb"]).mean()
        noun_acc = 100 * (df["gt_noun"] == df["pred_noun"]).mean()
        data.append([arch, f"{overall:.1f}%", f"{verb_acc:.1f}%", f"{noun_acc:.1f}%"])
    
    table = ax.table(
        cellText=data,
        colLabels=["Model", "Overall Acc", "Verb Acc", "Noun Acc"],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    fig.suptitle('Classification Accuracy Breakdown', fontsize=14, y=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_accuracy_table.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_accuracy_table.png'}")


def plot_center_bias_comparison(all_dfs, out_dir):
    """Model NSS vs Center-Bias NSS comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(all_dfs.keys())
    x = np.arange(len(models))
    width = 0.35
    
    model_nss = []
    cb_nss = []
    
    for arch in models:
        df = all_dfs[arch]
        model_nss.append(df["nss"].dropna().mean())
        cb_nss.append(df["center_bias_nss"].dropna().mean())
    
    ax.bar(x - width/2, model_nss, width, label='Model CAM', color='#2196F3')
    ax.bar(x + width/2, cb_nss, width, label='Center Bias', color='#9E9E9E')
    
    ax.set_ylabel('Mean NSS')
    ax.set_title('Model CAM vs Center-Bias Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Set y-axis limit with headroom for annotations
    ymax = max(max(model_nss), max(cb_nss)) + 0.25
    ax.set_ylim(top=ymax)
    
    # Add delta annotations
    for i in range(len(models)):
        delta = model_nss[i] - cb_nss[i]
        y = max(model_nss[i], cb_nss[i]) + 0.05
        ax.text(i, y, f"Δ={delta:+.2f}", ha='center', fontsize=10)
    
    fig.tight_layout()
    fig.savefig(out_dir / "fig_center_bias.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_center_bias.png'}")


def main():
    print("Loading CSVs...")
    all_dfs = {}
    for arch in SUPPORTED_ARCHS:
        csv = ANALYSIS_DIR / arch / f"alignment_split1.csv"
        if csv.exists():
            all_dfs[arch] = pd.read_csv(csv)
            print(f"  {arch}: {len(all_dfs[arch])} clips")
    
    if len(all_dfs) == 0:
        print("No CSVs found!")
        return
    
    print("\nGenerating combined figures...")
    plot_correct_vs_incorrect_combined(all_dfs, ANALYSIS_DIR)
    plot_error_buckets_combined(all_dfs, ANALYSIS_DIR)
    plot_verb_alignment_combined(all_dfs, ANALYSIS_DIR)
    plot_accuracy_table(all_dfs, ANALYSIS_DIR)
    plot_center_bias_comparison(all_dfs, ANALYSIS_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
