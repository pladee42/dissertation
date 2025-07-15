#!/bin/bash
#SBATCH --job-name=dpo_data_prep
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/logs/data_prep_%j.out
#SBATCH --error=outputs/logs/data_prep_%j.err
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# DPO Data Preparation Script
# 
# Usage:
#   From dpo/ directory: sbatch slurm/prepare_data.sh [input_folder]
#   From slurm/ directory: sbatch prepare_data.sh [input_folder]
#
# Examples:
#   sbatch slurm/prepare_data.sh ../output/multi_topic_results/20250715_035107
#   sbatch slurm/prepare_data.sh ../output/multi_topic_results/20250714_043247
#
# Default: Uses most recent data (20250715_035107)
# Output: Creates timestamped JSONL file in outputs/datasets/

# Load required modules
module load Anaconda3/2024.02-1

# Activate environment
source activate dis-venv3

# Set variables
INPUT_FOLDER=${1:-"../output/multi_topic_results/20250715_035107"}
OUTPUT_FILE="outputs/datasets/dpo_data_$(date +%Y%m%d_%H%M%S).jsonl"

echo "Starting DPO data preparation..."
echo "Input folder: $INPUT_FOLDER"
echo "Output file: $OUTPUT_FILE"

# Run data preparation
python scripts/prepare_data.py \
    --input-folder "$INPUT_FOLDER" \
    --output-file "$OUTPUT_FILE" \
    --config configs/data_config.yaml

echo "Data preparation completed!"