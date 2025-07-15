#!/bin/bash
#SBATCH --job-name=dpo_training
#SBATCH --time=12:00:00
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/logs/train_dpo_%j.out
#SBATCH --error=outputs/logs/train_dpo_%j.err

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
OUTPUT_DIR="outputs/models/dpo_model_$(date +%Y%m%d_%H%M%S)"

echo "Starting DPO training job..."
echo "Data file: $DATA_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

# Run training
python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --config configs/training_config.yaml

echo "DPO training completed!"