#!/bin/bash
#SBATCH --job-name=multi_topic
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=./log/multi_topic_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate conda environment
source activate dis-venv3

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH

# Set model cache directories
export HF_HOME=./downloaded_models
export HF_HUB_CACHE=./downloaded_models

# vLLM environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Clean up GPU memory and check status
echo "Cleaning up GPU processes..."
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | xargs -r kill -9
echo "GPU status after cleanup:"
nvidia-smi

# Run script
echo "Running multi_topic_runner.py..."
# python -m multi_topic_runner --email_generation=medium --judge_model=gemini-2.5-flash --checklist_mode=enhanced --all_topics &> enhanced.log
# python -m multi_topic_runner --email_generation=medium --judge_model=gemini-2.5-flash --checklist_mode=extract_only --all_topics &> extract_only.log
python -m multi_topic_runner --email_generation=medium --checklist_model=deepseek-r1 --judge_model=o3-mini --checklist_mode=preprocess --all_topics

# For DPO
# python -m multi_topic_runner --email_generation=dpo-hyb --checklist_model=deepseek-r1 --judge_model=o3-mini --checklist_mode=preprocess --all_topics
# python -m multi_topic_runner --email_generation=dpo-syn --checklist_model=deepseek-r1 --judge_model=o3-mini --checklist_mode=preprocess --all_topics

# For debugging & Testing
# python -m multi_topic_runner --email_models=stablelm-2-1.6b --checklist_model=gpt-4.1-nano --judge_model=o3-mini --checklist_mode=preprocess --topics=T0001

# Check execution status
if [ $? -eq 0 ]; then
    echo "Execution completed successfully!"
else
    echo "Execution failed with error code $?"
fi

# Monitor GPU status after running
echo "GPU status after execution:"
nvidia-smi

echo "Job completed"
