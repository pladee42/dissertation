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

# Activate conda environment using the new method
conda activate dis-venv

# Verify PyTorch installation and print version information
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Set up environment variables for distributed training
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(( 10000 + $RANDOM % 20000 ))
export GPUS_PER_NODE=4
export TORCH_EXTENSIONS_DIR=$HOME/.cache/torch_extensions

# Create DeepSpeed config if not exists
if [ ! -f "ds_config.json" ]; then
    cat > ds_config.json << 'EOL'
{
    "fp16": {
        "enabled": true
    },
    "bf16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
EOL
    echo "Created DeepSpeed configuration file: ds_config.json"
fi

# Try using torchrun directly (available in newer PyTorch versions)
echo "Attempting to run with torchrun..."
torchrun --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py --deepspeed ds_config.json

# Fallback: If torchrun fails, try directly with deepspeed
if [ $? -ne 0 ]; then
    echo "torchrun failed, attempting to run with deepspeed directly..."
    # Ensure deepspeed is installed
    pip install --quiet deepspeed
    
    # Run with deepspeed launcher
    deepspeed \
        --num_gpus=$GPUS_PER_NODE \
        main.py --deepspeed ds_config.json
fi
