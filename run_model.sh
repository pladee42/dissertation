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
source activate agent-env

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For multi-GPU setups
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions

# Set model cache directories
export HF_HOME=./downloaded_models
export TRANSFORMERS_CACHE=./downloaded_models
export HF_HUB_CACHE=./downloaded_models

# vLLM environment variables
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

# Verify PyTorch installation and print version information
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Monitor GPU status before running
echo "GPU status before execution:"
nvidia-smi

# Try to start vLLM server
echo "Attempting to start vLLM server..."
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --download-dir ./downloaded_models --port 30000 --host 0.0.0.0 &
VLLM_PID=$!

# Wait for server to start
echo "Waiting for vLLM server to start..."
sleep 30

# Verify server is running
if curl -s http://localhost:30000/health > /dev/null; then
    echo "vLLM server is running"
    SERVER_AVAILABLE=true
else
    echo "vLLM server failed to start, running with fallback mode"
    SERVER_AVAILABLE=false
    # Kill the failed process
    kill $VLLM_PID 2>/dev/null || true
fi

# Run script (agents will handle SGLang unavailability gracefully)
echo "Running runner.py..."
python -m runner

# Clean up vLLM server if it was running
if [ "$SERVER_AVAILABLE" = true ]; then
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
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
