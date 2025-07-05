#!/bin/bash
#SBATCH --job-name=dissertation_container
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=container_job_%j.out
#SBATCH --error=container_job_%j.err

echo "=== Starting containerized job on Sheffield HPC ==="

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$PWD/downloaded_models

# Create directories
mkdir -p downloaded_models output log

# Path to Singularity image
SINGULARITY_IMAGE="dissertation-env.sif"

# Verify Singularity image exists
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "Error: Singularity image $SINGULARITY_IMAGE not found!"
    exit 1
fi

# Test environment within container
echo "=== Testing containerized environment ==="
singularity exec --nv \
    --bind $PWD/downloaded_models:/workspace/downloaded_models \
    --bind $PWD/output:/workspace/output \
    --bind $PWD/log:/workspace/log \
    --bind $PWD:/workspace/code \
    $SINGULARITY_IMAGE \
    python3 /workspace/code/test_glibcxx.py

# Run your actual job
echo "=== Running containerized dissertation code ==="
singularity exec --nv \
    --bind $PWD/downloaded_models:/workspace/downloaded_models \
    --bind $PWD/output:/workspace/output \
    --bind $PWD/log:/workspace/log \
    --bind $PWD:/workspace/code \
    $SINGULARITY_IMAGE \
    bash -c "
        cd /workspace/code && \
        python -m runner --email_generation=medium --topic='Children Hospital Cancer Treatment Fund'
    "

echo "=== Containerized job completed ==="