#!/bin/bash
#SBATCH --job-name=test_checklist
#SBATCH --time=08:30:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --mem=240G
#SBATCH --output=./log/test_checklist_%j.log
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load required modules
module load Anaconda3/2024.02-1
module load CUDA/12.4.0
module load GCC/12.2.0

# Activate conda environment
source activate dis-venv3
conda env export > dis-venv3.yml

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions
export MKL_THREADING_LAYER=GNU

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Start SGLang server first
echo "Starting SGLang server..."
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000 --host 0.0.0.0 &
SGLANG_PID=$!

# Wait for server to start
echo "Waiting for SGLang server to start..."
sleep 30

# Verify server is running
if curl -s http://localhost:30000/health > /dev/null; then
    echo "SGLang server is running"
else
    echo "Failed to start SGLang server"
    exit 1
fi

# Run script
echo "Running test pipeline..."
python -m test_pipeline

# Clean up SGLang server
echo "Stopping SGLang server..."
kill $SGLANG_PID 2>/dev/null || true

# Check execution status
if [ $? -eq 0 ]; then
    echo "Execution completed successfully!"
else
    echo "Execution failed with error code $?"
fi

echo "Job completed"
