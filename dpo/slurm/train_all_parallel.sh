#!/bin/bash
#SBATCH --job-name=dpo_all_par
#SBATCH --time=06:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=20
#SBATCH --output=outputs/logs/train_all_par_%j.out
#SBATCH --error=outputs/logs/train_all_par_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# A100 80GB optimized - Parallel training of all 5 models
# Resource allocation: 5 GPUs, 20 CPUs, 160GB RAM, 6 hours
# Trains all models simultaneously for fastest completion

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

echo "Starting A100-optimized parallel multi-model DPO training job..."
echo "Data file: $DATA_FILE"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Cache directory: $CACHE_DIR"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Resources: 5x A100 80GB, 20 CPUs, 160GB RAM"
echo "Training strategy: Parallel (all 5 models simultaneously)"
echo "Models: tinyllama + vicuna + phi3 + llama3 + stablelm"

# Create individual background processes for each model
echo "Starting parallel training processes..."

# TinyLlama on GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "tinyllama" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR" &
TINYLLAMA_PID=$!

# Vicuna on GPU 1  
CUDA_VISIBLE_DEVICES=1 python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "vicuna" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR" &
VICUNA_PID=$!

# Phi-3 on GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "phi3" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR" &
PHI3_PID=$!

# Llama-3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "llama3" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR" &
LLAMA3_PID=$!

# StableLM on GPU 4
CUDA_VISIBLE_DEVICES=4 python scripts/train_dpo.py \
    --data-file "$DATA_FILE" \
    --model "stablelm" \
    --output-dir "$OUTPUT_BASE_DIR" \
    --cache-dir "$CACHE_DIR" &
STABLELM_PID=$!

echo "All training processes started. Waiting for completion..."

# Wait for all processes to complete
wait $TINYLLAMA_PID
echo "✅ TinyLlama training completed"

wait $VICUNA_PID  
echo "✅ Vicuna training completed"

wait $PHI3_PID
echo "✅ Phi-3 training completed"

wait $LLAMA3_PID
echo "✅ Llama-3 training completed"

wait $STABLELM_PID
echo "✅ StableLM training completed"

echo "🎉 All models training completed!"
echo "Parallel training strategy: 5x faster completion using dedicated A100 per model"