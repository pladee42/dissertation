# DPO Training - Simplified Workflow

## Overview

This simplified DPO training system provides **merged models ready for manual upload** to HuggingFace. The training script handles everything locally and gives you the final merged model.

## Quick Start

### 1. Train DPO Models
```bash
# Train single model
sbatch slurm/train_single.sh tinyllama

# Train all models sequentially  
sbatch slurm/train_all_sequential.sh
```

### 2. Training Output
After training completes, you'll get:
```
outputs/models/dpo_tinyllama_20250716_123456/
├── merged_model/          # 🚀 Upload this to HuggingFace
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
├── adapter_config.json    # LoRA adapter (backup)
├── adapter_model.bin      # LoRA weights (backup)
└── training_info.yaml     # Training metadata
```

### 3. Manual Upload to HuggingFace
```bash
# Navigate to merged model directory
cd outputs/models/dpo_tinyllama_20250716_123456/merged_model/

# Upload using HF CLI
huggingface-cli upload your-username/tinyllama-1.1b-dpo . --repo-type model

# OR use web interface at huggingface.co
```

### 4. Update Config
Edit `../../config/config.py` and update the model_id:
```python
'tinyllama-1.1b-dpo': {
    'model_id': 'your-username/tinyllama-1.1b-dpo',  # Your uploaded model
    # ... rest stays the same
}
```

### 5. Use in Pipeline
```bash
# Use DPO model
python -m runner --email_models tinyllama-1.1b-dpo

# Compare base vs DPO
python -m runner --email_models tinyllama-1.1b tinyllama-1.1b-dpo
```

## File Structure

### Essential Files
```
dpo/
├── scripts/
│   ├── train_dpo.py           # ✅ Main training script
│   ├── compare_models.py      # ✅ Compare base vs DPO
│   └── evaluate_comparisons.py # ✅ Generate metrics
├── slurm/                     # ✅ SLURM job scripts
├── configs/                   # ✅ Model configurations
└── outputs/                   # ✅ Training results
```

## Example Training Output

```
✅ Training completed!
📁 LoRA adapter saved to: outputs/models/dpo_tinyllama_20250716_123456
🔗 Merged model saved to: outputs/models/dpo_tinyllama_20250716_123456/merged_model
📝 Training info saved to: outputs/models/dpo_tinyllama_20250716_123456/training_info.yaml
🚀 Ready for manual upload to HuggingFace!

📝 Manual upload instructions:
1. Navigate to each merged_model directory
2. Use 'huggingface-cli upload' or web interface
3. Update config with your HF model paths
```

## Comparison Tools

After uploading models, use the comparison tools:

```bash
# List available comparisons
python dpo/scripts/compare_models.py --list

# Run comparison
python dpo/scripts/compare_models.py --base tinyllama-1.1b --dpo tinyllama-1.1b-dpo

# Evaluate results
python dpo/scripts/evaluate_comparisons.py --all
```