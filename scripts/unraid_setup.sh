#!/bin/bash
# Run this in the Unraid web terminal (or SSH)
# It pulls a PyTorch Docker image, installs deps, and runs the full analysis.

PROJECT="/mnt/user/appdata/Github/Gaze CAM"

# Verify the project exists
if [ ! -f "$PROJECT/scripts/run_all_analysis.py" ]; then
    echo "ERROR: Project not found at $PROJECT"
    echo "Check that the path matches your Unraid share layout."
    ls -la "/mnt/user/appdata/Github/" 2>/dev/null
    exit 1
fi

echo "Project found at: $PROJECT"
echo "Starting Docker container with GPU..."

# Run the analysis inside a PyTorch container with GPU passthrough
# - mounts the project folder as /workspace
# - installs extra Python deps
# - runs the full 4-model analysis with 10 shuffles + occlusion sensitivity
docker run --gpus all --rm \
  -v "$PROJECT":/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  bash -c '
    echo "=== Inside Docker container ==="
    echo "GPU:"
    python -c "import torch; print(torch.cuda.get_device_name(0)); print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB\")"

    echo ""
    echo "=== Installing dependencies ==="
    pip install --quiet opencv-python-headless tqdm pandas scipy matplotlib pytorchvideo transformers accelerate Pillow imageio

    echo ""
    echo "=== Running full analysis (4 models, 500 clips each) ==="
    python scripts/run_all_analysis.py --split 1 --max-test 500 --n-shuffles 10

    echo ""
    echo "=== DONE ==="
    echo "Results saved to outputs/analysis/"
    ls -la outputs/analysis/
  '
