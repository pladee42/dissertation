#!/bin/bash
#SBATCH --job-name=dpo_medium
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/train_medium_%j.out
#SBATCH --error=outputs/logs/train_medium_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# A100 80GB optimized - Medium models (Vicuna, Phi-3, Llama-3-8B)
# Resource allocation: 1 GPU, 8 CPUs, 64GB RAM, 8 hours
# Optimized for medium-sized models (4B - 8B parameters)

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

echo "Starting A100-optimized medium models DPO training job..."
echo "Data file: $DATA_FILE"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Resources: 1x A100 80GB, 8 CPUs, 64GB RAM"
echo "Target models: Vicuna (7B) + Phi-3 (4B) + Llama-3 (8B)"

# Train medium models (Vicuna + Phi-3 + Llama-3)
python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --models "vicuna,phi3,llama3" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Medium models training completed!"
echo "Resource efficiency: Balanced allocation for 4-8B parameter models"