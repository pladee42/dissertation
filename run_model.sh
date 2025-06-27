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
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate conda environment
source activate dis-venv3

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
# SGLang environment variables
export SGLANG_BACKEND=flashinfer
export SGLANG_DISABLE_DISK_CACHE=false
export SGLANG_CHUNK_PREFILL_BUDGET=512
export SGLANG_MEM_FRACTION_STATIC=0.85

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Run script without distributed launcher (using built-in model parallelism)
echo "Running main.py with native model parallelism..."
python -m multi_model_runner --prompt_mode=all --language_model=deepseek-r1-1.5b --topic='Polar Bears Rescue by University of Sheffield' --checklist_model=llama-3-3b --judge_model=llama-3-8b

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
