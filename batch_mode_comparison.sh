#!/bin/bash
#SBATCH --job-name=batch_mode_comparison
#SBATCH --output=log/batch_comparison_%j.out
#SBATCH --error=log/batch_comparison_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=1-5

# Batch mode comparison script for topics T0001-T0005 across all 3 checklist modes
# Usage: sbatch batch_mode_comparison.sh

echo "=== Batch Mode Comparison on HPC ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
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

# Topic mapping for array job
declare -a TOPICS=(
    "T0001"  # Call to Support the Polar Bear Cub Survival Act
    "T0002"  # Giving Tuesday Fundraising Campaign for Wildlife
    "T0003"  # Membership Renewal and Urgent Appeal for Endangered Species
    "T0004"  # Endangered Species Day Virtual Events and Advocacy
    "T0005"  # Petition to Protect North Atlantic Right Whales from Ship Strikes
)

# Get current topic
TOPIC_UID=${TOPICS[$SLURM_ARRAY_TASK_ID-1]}
MODEL="tinyllama-1.1b"

echo "Processing topic: $TOPIC_UID"
echo "Model: $MODEL"

# Create results directory for this job
RESULTS_DIR="log/batch_results_$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR

# Function to run mode and capture results
run_mode() {
    local mode=$1
    local output_file="$RESULTS_DIR/${TOPIC_UID}_${mode}_task${SLURM_ARRAY_TASK_ID}.log"
    
    echo "Running $mode mode for $TOPIC_UID..."
    echo "Output file: $output_file"
    
    start_time=$(date +%s)
    python -m multi_topic_runner --topics $TOPIC_UID --email_models $MODEL --checklist_mode=$mode > "$output_file" 2>&1
    exit_code=$?
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $mode mode completed successfully in ${runtime}s"
        echo "$TOPIC_UID,$mode,SUCCESS,$runtime" >> "$RESULTS_DIR/summary_task${SLURM_ARRAY_TASK_ID}.csv"
    else
        echo "❌ $mode mode failed with exit code $exit_code"
        echo "$TOPIC_UID,$mode,FAILED,$runtime" >> "$RESULTS_DIR/summary_task${SLURM_ARRAY_TASK_ID}.csv"
    fi
}

# Initialize summary file
echo "topic_uid,mode,status,runtime_seconds" > "$RESULTS_DIR/summary_task${SLURM_ARRAY_TASK_ID}.csv"

echo ""
echo "=== Running All Modes for $TOPIC_UID ==="

# Run all three modes
run_mode "enhanced"
sleep 10  # Brief pause between modes

run_mode "extract_only"
sleep 10

run_mode "preprocess"

echo ""
echo "=== Task Completed ==="
echo "Finished at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "Summary file: $RESULTS_DIR/summary_task${SLURM_ARRAY_TASK_ID}.csv"

# Show summary
echo ""
echo "=== Summary for $TOPIC_UID ==="
cat "$RESULTS_DIR/summary_task${SLURM_ARRAY_TASK_ID}.csv"