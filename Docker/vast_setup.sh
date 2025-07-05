#!/bin/bash
# Vast.ai provisioning script - 10X Engineer approach

echo "=== Vast.ai Environment Setup ==="

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/downloaded_models

# Create directories
mkdir -p /workspace/downloaded_models
mkdir -p /workspace/output
mkdir -p /workspace/log

echo "=== Verifying GLIBCXX compatibility ==="
ldd --version
echo "GLIBCXX versions available:"
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5

echo "=== Testing environment ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test GPU access
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "🎯 GPU access: Available"
    python3 -c "import torch; print(f'GPU devices: {torch.cuda.device_count()}')"
else
    echo "⚠️  GPU access: Not available (CPU mode)"
fi

echo "=== Running full compatibility test ==="
cd /workspace/code
python3 test_glibcxx.py

echo "=== Environment setup complete! ==="