"""
run_all_analysis.py – Run analysis for all 4 architectures sequentially,
then produce the cross-architecture comparison.

Usage:
    python scripts/run_all_analysis.py [--split 1] [--max-test 500]
"""

import subprocess
import sys
from pathlib import Path

MODELS = ["r3d18", "slowfast_r50", "timesformer", "vivit"]
PYTHON = sys.executable


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--max-test", type=int, default=500)
    parser.add_argument("--n-shuffles", type=int, default=10)
    parser.add_argument("--skip-occlusion", action="store_true")
    parser.add_argument("--skip-randomization", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"  RUNNING ANALYSIS: {model}")
        print(f"{'='*70}\n")
        cmd = [
            PYTHON, str(scripts_dir / "analysis.py"),
            "--model", model,
            "--split", str(args.split),
            "--max-test", str(args.max_test),
            "--batch-size", "4",
            "--n-shuffles", str(args.n_shuffles),
        ]
        if args.skip_occlusion:
            cmd.append("--skip-occlusion")
        if args.skip_randomization:
            cmd.append("--skip-randomization")
        if args.inference_only:
            cmd.append("--inference-only")
        # ViViT needs smaller batch for CAM computation
        if model == "vivit":
            cmd[cmd.index("--batch-size") + 1] = "1"
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n  WARNING: {model} analysis exited with code {result.returncode}")

    # Cross-architecture comparison
    print(f"\n{'='*70}")
    print("  CROSS-ARCHITECTURE COMPARISON")
    print(f"{'='*70}\n")
    subprocess.run([
        PYTHON, str(scripts_dir / "compare_models.py"),
        "--split", str(args.split),
    ])


if __name__ == "__main__":
    main()
