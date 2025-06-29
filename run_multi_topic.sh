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

# Try to start SGLang server
echo "Attempting to start SGLang server..."
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 30000 --host 0.0.0.0 &
SGLANG_PID=$!

# Wait for server to start
echo "Waiting for SGLang server to start..."
sleep 30

# Verify server is running
if curl -s http://localhost:30000/health > /dev/null; then
    echo "SGLang server is running"
    SERVER_AVAILABLE=true
else
    echo "SGLang server failed to start, running with fallback mode"
    SERVER_AVAILABLE=false
    # Kill the failed process
    kill $SGLANG_PID 2>/dev/null || true
fi

# Run script (agents will handle SGLang unavailability gracefully)
echo "Running multi_topic_runner.py..."
python -m multi_topic_runner

# Clean up SGLang server if it was running
if [ "$SERVER_AVAILABLE" = true ]; then
    echo "Stopping SGLang server..."
    kill $SGLANG_PID 2>/dev/null || true
fi

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
