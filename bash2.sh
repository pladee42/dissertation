#!/bin/bash
#SBATCH --job-name=local_llm_dissertation2 # Replace JOB_NAME with a name you like
#SBATCH --time=04:30:00  # Change this to a longer time if you need more time
#SBATCH --nodes=1  # Specify a number of nodes
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --mem=140G
#SBATCH --output=./log/result7.txt  # This is where your output and errors are logged
#SBATCH --mail-user=wratthapoom1@sheffield.ac.uk  # Request job update email notifications, remove this line if you don't want to be notified
#SBATCH --mail-type=ALL

module load Anaconda3/2024.02-1
module load CUDA/11.8.0

# Activate conda environment
source activate dis-venv

python3 main.py