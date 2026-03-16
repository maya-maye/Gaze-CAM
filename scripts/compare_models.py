"""
compare_models.py – Cross-architecture comparison of gaze-CAM alignment.

Loads per-model alignment CSVs produced by analysis.py and generates
a unified comparison table + figure across all 4 architectures.

Usage:
    python scripts/compare_models.py [--split 1]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gaze_cam.config import OUTPUT_DIR, SUPPORTED_ARCHS, SPLIT


ANALYSIS_DIR = OUTPUT_DIR / "analysis"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=SPLIT)
    args = parser.parse_args()

    all_dfs = {}
    for arch in SUPPORTED_ARCHS:
        csv = ANALYSIS_DIR / arch / f"alignment_split{args.split}.csv"
        if csv.exists():
            all_dfs[arch] = pd.read_csv(csv)
            print(f"  Loaded {arch}: {len(all_dfs[arch])} clips")
        else:
            print(f"  Missing: {csv}")

    if len(all_dfs) < 2:
        print("Need at least 2 models to compare. Run analysis.py first.")
        return

    # ── Summary table ──
    print("\n" + "=" * 90)
    print("CROSS-ARCHITECTURE COMPARISON")
    print("=" * 90)

    header = (f"{'Model':<16} {'NSS':>8} {'AUC':>8} {'KL div':>8} "
              f"{'CB NSS':>8} {'Acc%':>7} {'N':>5}")
    print(header)
    print("-" * 90)

    summary_rows = []
    for arch, df in all_dfs.items():
        v = df.dropna(subset=["nss"])
        acc = 100 * df["correct"].mean()
        row = {
            "model": arch,
            "nss_mean": v["nss"].mean(),
            "nss_std": v["nss"].std(),
            "auc_mean": v["auc"].mean() if "auc" in v else np.nan,
            "kl_mean": (v["kl_divergence"].mean()
                        if "kl_divergence" in v else np.nan),
            "cb_nss": (v["center_bias_nss"].mean()
                       if "center_bias_nss" in v else np.nan),
            "accuracy": acc,
            "n": len(v),
        }
        summary_rows.append(row)
        kl_str = f"{row['kl_mean']:.4f}" if not np.isnan(row['kl_mean']) else "  N/A"
        cb_str = f"{row['cb_nss']:.4f}" if not np.isnan(row['cb_nss']) else "  N/A"
        print(f"  {arch:<14} {row['nss_mean']:+.4f}  "
              f"{row['auc_mean']:.4f}  {kl_str:>8}  "
              f"{cb_str:>8}  {acc:5.1f}%  {row['n']:>5}")

    # ── Correct vs Incorrect per model ──
    print(f"\n{'Model':<16} {'Correct NSS':>12} {'Incorrect NSS':>14} {'Delta':>8}")
    print("-" * 60)
    for arch, df in all_dfs.items():
        c = df[df["correct"] == 1]["nss"].dropna()
        ic = df[df["correct"] == 0]["nss"].dropna()
        if len(c) > 0 and len(ic) > 0:
            delta = c.mean() - ic.mean()
            print(f"  {arch:<14} {c.mean():+.4f}      "
                  f"{ic.mean():+.4f}        {delta:+.4f}")

    # Save summary CSV
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = ANALYSIS_DIR / f"comparison_split{args.split}.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"\nSaved: {sum_csv}")

    # ── Comparison bar plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = [r["model"] for r in summary_rows]
    x = range(len(models))
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(models)]

    # NSS
    nss_vals = [r["nss_mean"] for r in summary_rows]
    nss_stds = [r["nss_std"] for r in summary_rows]
    axes[0].bar(x, nss_vals, yerr=nss_stds, capsize=5,
                color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Mean NSS")
    axes[0].set_title("Gaze-CAM Alignment (NSS)")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    # AUC
    auc_vals = [r["auc_mean"] for r in summary_rows]
    axes[1].bar(x, auc_vals, color=bar_colors,
                edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Mean AUC-Judd")
    axes[1].set_title("Gaze-CAM Alignment (AUC)")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5,
                    label="Chance")
    axes[1].legend()

    # Accuracy
    acc_vals = [r["accuracy"] for r in summary_rows]
    axes[2].bar(x, acc_vals, color=bar_colors,
                edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("Action Recognition Accuracy")

    fig.suptitle("Cross-Architecture Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    out = ANALYSIS_DIR / f"fig_comparison_split{args.split}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
