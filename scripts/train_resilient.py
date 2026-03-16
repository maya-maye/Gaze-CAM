"""
Crash-resilient training wrapper.

Keeps re-launching training until all epochs complete.
If the process crashes (OOM, CUDA error, etc.) it just restarts
from the last saved checkpoint automatically.

Usage:
    python scripts/train_resilient.py --model vivit --split 1
    python scripts/train_resilient.py --model vivit --split 1 --max-retries 50
"""

import subprocess
import sys
import time
from pathlib import Path

# Re-use the same argument parser as train.py so all flags work
TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"
PYTHON = sys.executable


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crash-resilient training wrapper")
    parser.add_argument("--max-retries", type=int, default=100,
                        help="Max number of restarts before giving up")
    # Capture all remaining args to forward to train.py
    args, train_args = parser.parse_known_args()

    max_retries = args.max_retries
    attempt = 0

    while attempt < max_retries:
        attempt += 1
        print(f"\n{'='*60}")
        print(f"  ATTEMPT {attempt}/{max_retries}  —  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        cmd = [PYTHON, str(TRAIN_SCRIPT)] + train_args
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"\n{'='*60}")
            print(f"  TRAINING COMPLETED SUCCESSFULLY after {attempt} attempt(s)")
            print(f"{'='*60}")
            return 0

        print(f"\n  ** Process crashed (exit code {result.returncode}). "
              f"Restarting in 10s ... **\n")
        # Brief pause to let GPU memory fully release
        time.sleep(10)

    print(f"\n  GAVE UP after {max_retries} retries.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
