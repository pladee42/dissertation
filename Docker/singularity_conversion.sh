#!/bin/bash
# Convert Docker image to Singularity format for Sheffield HPC - 10X Engineer approach

# Configuration
DOCKER_IMAGE="${DOCKER_REGISTRY:-docker.io/your-username}/dissertation-env:latest"
SINGULARITY_IMAGE="dissertation-env.sif"

echo "=== Converting Docker to Singularity for Sheffield HPC ==="
echo "Docker image: $DOCKER_IMAGE"
echo "Singularity image: $SINGULARITY_IMAGE"

# Check if singularity is available
if ! command -v singularity &> /dev/null; then
    echo "❌ Singularity not found. On Sheffield HPC, run: module load Singularity"
    exit 1
fi

# Convert Docker to Singularity
echo "Converting $DOCKER_IMAGE to $SINGULARITY_IMAGE..."
singularity build $SINGULARITY_IMAGE docker://$DOCKER_IMAGE

# Test Singularity image
echo "=== Testing Singularity image ==="
singularity exec $SINGULARITY_IMAGE python3 -c "
import sys
print(f'Python version: {sys.version}')
import torch
print(f'PyTorch version: {torch.__version__}')
import vllm
print('vLLM import successful')
"

# Test GLIBCXX compatibility in Singularity
echo "=== Testing GLIBCXX compatibility in Singularity ==="
singularity exec $SINGULARITY_IMAGE ldd --version
echo "Available GLIBCXX versions:"
singularity exec $SINGULARITY_IMAGE strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | tail -5

echo ""
echo "🎯 Singularity conversion complete!"
echo "📋 Next steps:"
echo "   1. Transfer to Sheffield HPC: scp $SINGULARITY_IMAGE username@sharc.sheffield.ac.uk:~/"
echo "   2. Submit jobs using: sbatch slurm_container_job.sh"
echo ""
echo "Singularity image ready: $SINGULARITY_IMAGE"