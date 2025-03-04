#!/bin/bash
#SBATCH --job-name=deepseek_r1_70b
#SBATCH --time=04:30:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --output=./log/deepseek_r1_70b_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/11.8.0

# Initialize conda for bash shell (critical for non-interactive shells)
source $(conda info --base)/etc/profile.d/conda.sh

# Activate conda environment
conda activate dis-venv

# Ensure required packages are installed
echo "Checking and installing required packages..."
pip install --quiet transformers accelerate bitsandbytes sentencepiece protobuf

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
# Set BitsAndBytes to use CUDA 11.8
# export BNB_CUDA_VERSION=118

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running main.py with native model parallelism..."
python main.py

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
