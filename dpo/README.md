# DPO Fine-Tuning for Email Generation

## Quick Start
1. Prepare data: `python scripts/prepare_data.py --input-folder output/multi_topic_results/20250714_043247`
2. Train model: `python scripts/train_dpo.py --data-file outputs/datasets/dpo_data.jsonl`

## SLURM Usage
```bash
sbatch slurm/prepare_data.sh 20250714_043247
sbatch slurm/train_dpo.sh
```

## Project Structure
- `scripts/`: Python scripts for data prep and training
- `configs/`: YAML configuration files
- `slurm/`: SLURM batch job scripts
- `outputs/`: Generated datasets, models, and logs