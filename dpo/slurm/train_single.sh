#!/bin/bash
#SBATCH --job-name=dpo_single
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/train_single_%j.out
#SBATCH --error=outputs/logs/train_single_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Optimized for A100 80GB GPUs - Single model training
# Resource allocation: 1 GPU, 8 CPUs, 64GB RAM, 6 hours
# Suitable for all model sizes (1.1B - 8B parameters)

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
MODEL_NAME=${1:-"tinyllama"}
DATA_FILE=${2:-"outputs/datasets/dpo_data.jsonl"}
OUTPUT_BASE_DIR="outputs/models"
CACHE_DIR="../downloaded_models"

# Set HF_HOME to use cached models from main pipeline
export HF_HOME="$CACHE_DIR"

echo "Starting A100-optimized single-model DPO training job..."
echo "Model: $MODEL_NAME"
echo "Data file: $DATA_FILE"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Resources: 1x A100 80GB, 8 CPUs, 64GB RAM"

# Validate model name
if [[ ! "$MODEL_NAME" =~ ^(tinyllama|vicuna|phi3|llama3|stablelm)$ ]]; then
    echo "Error: Invalid model name '$MODEL_NAME'"
    echo "Valid models: tinyllama, vicuna, phi3, llama3, stablelm"
    exit 1
fi

# Train specific model
python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR"

echo "Model $MODEL_NAME training completed!"
echo "A100 resource utilization: Single GPU optimal for models up to 8B parameters"