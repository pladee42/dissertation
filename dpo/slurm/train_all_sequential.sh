#!/bin/bash
#SBATCH --job-name=dpo_all_seq
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --output=outputs/logs/train_all_seq_%j.out
#SBATCH --error=outputs/logs/train_all_seq_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# A100 80GB optimized - Sequential training of all 5 models
# Resource allocation: 1 GPU, 12 CPUs, 96GB RAM, 16 hours
# Trains all models one by one on single A100 for maximum efficiency

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

echo "Starting A100-optimized sequential multi-model DPO training job..."
echo "Data file: $DATA_FILE"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Resources: 1x A100 80GB, 12 CPUs, 96GB RAM"
echo "Training strategy: Sequential (all 5 models one by one)"
echo "Models: tinyllama → vicuna → phi3 → llama3 → stablelm"

# Train all 5 models sequentially on single A100
python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --models "tinyllama,vicuna,phi3,llama3,stablelm" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR"

echo "All models training completed!"
echo "Sequential training maximizes A100 80GB utilization - each model uses full GPU capacity"