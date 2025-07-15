#!/bin/bash
#SBATCH --job-name=dpo_small
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/logs/train_small_%j.out
#SBATCH --error=outputs/logs/train_small_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# A100 80GB optimized - Small models only (TinyLlama, StableLM)
# Resource allocation: 1 GPU, 4 CPUs, 32GB RAM, 4 hours
# Most efficient for small models (1.1B - 1.6B parameters)

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate environment
source activate dis-venv3

# Environment variables
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set variables
DATA_FILE=${1:-"outputs/datasets/dpo_data.jsonl"}
OUTPUT_BASE_DIR="outputs/models"
CACHE_DIR="../downloaded_models"

# Set HF_HOME to use cached models from main pipeline
export HF_HOME="$CACHE_DIR"

echo "Starting A100-optimized small models DPO training job..."
echo "Data file: $DATA_FILE"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Resources: 1x A100 80GB, 4 CPUs, 32GB RAM"
echo "Target models: TinyLlama (1.1B) + StableLM (1.6B)"

# Train small models (TinyLlama + StableLM)
python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --models "tinyllama,stablelm" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Small models training completed!"
echo "Resource efficiency: Minimal allocation for 1-2B parameter models"