#!/bin/bash
#SBATCH --job-name=test_checklist_modes
#SBATCH --output=log/test_modes_%j.out
#SBATCH --error=log/test_modes_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Test SLURM job script for single topic across all 3 checklist modes
# Usage: sbatch test_modes.sh

echo "=== Testing Checklist Modes on HPC ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate conda environment
source activate agent-env

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=./downloaded_models

# Test topic
TOPIC="HPC Mode Testing"
MODEL="tinyllama-1.1b"

echo ""
echo "=== Testing Enhanced Mode ==="
echo "Command: python -m runner --email_models $MODEL --topic=\"$TOPIC\" --checklist_mode=enhanced"
python -m runner --email_models $MODEL --topic="$TOPIC" --checklist_mode=enhanced
if [ $? -eq 0 ]; then
    echo "✅ Enhanced mode: SUCCESS"
else
    echo "❌ Enhanced mode: FAILED"
fi

echo ""
echo "=== Testing Extract-Only Mode ==="
echo "Command: python -m runner --email_models $MODEL --topic=\"$TOPIC\" --checklist_mode=extract_only"
python -m runner --email_models $MODEL --topic="$TOPIC" --checklist_mode=extract_only
if [ $? -eq 0 ]; then
    echo "✅ Extract-only mode: SUCCESS"
else
    echo "❌ Extract-only mode: FAILED"
fi

echo ""
echo "=== Testing Preprocess Mode ==="
echo "Command: python -m runner --email_models $MODEL --topic=\"$TOPIC\" --checklist_mode=preprocess"
python -m runner --email_models $MODEL --topic="$TOPIC" --checklist_mode=preprocess
if [ $? -eq 0 ]; then
    echo "✅ Preprocess mode: SUCCESS"
else
    echo "❌ Preprocess mode: FAILED"
fi

echo ""
echo "=== Checking File Organization ==="
DATE=$(date +%Y-%m-%d)
TRAINING_DIR="output/training_data/$DATE"

echo "Training data directory: $TRAINING_DIR"
if [ -d "$TRAINING_DIR/enhanced" ]; then
    echo "✅ Enhanced directory exists"
    ls -la "$TRAINING_DIR/enhanced/"
else
    echo "❌ Enhanced directory missing"
fi

if [ -d "$TRAINING_DIR/extract_only" ]; then
    echo "✅ Extract-only directory exists"
    ls -la "$TRAINING_DIR/extract_only/"
else
    echo "❌ Extract-only directory missing"
fi

if [ -d "$TRAINING_DIR/preprocess" ]; then
    echo "✅ Preprocess directory exists"
    ls -la "$TRAINING_DIR/preprocess/"
else
    echo "❌ Preprocess directory missing"
fi

echo ""
echo "=== Job Completed ==="
echo "Finished at: $(date)"
echo "Check output files in: $TRAINING_DIR"