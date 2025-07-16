#!/usr/bin/env python3
"""
Simple script to merge LoRA adapters with base models and upload to Hugging Face Hub
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

def load_model_registry():
    """Load model registry"""
    with open('../configs/model_registry.yaml', 'r') as f:
        return yaml.safe_load(f)

def find_trained_models():
    """Find all trained DPO models"""
    models_dir = Path('../outputs/models')
    trained_models = []
    
    if not models_dir.exists():
        print("No models directory found. Train some models first!")
        return trained_models
    
    for model_dir in models_dir.glob('dpo_*'):
        if model_dir.is_dir():
            # Check if it has the required files
            if (model_dir / 'adapter_config.json').exists():
                model_name = model_dir.name.replace('dpo_', '').split('_')[0]
                trained_models.append({
                    'name': model_name,
                    'path': str(model_dir),
                    'timestamp': model_dir.name.split('_')[-1] if '_' in model_dir.name else 'unknown'
                })
    
    return trained_models

def create_model_card(model_name, base_model_id, training_info):
    """Create a simple model card for the DPO model"""
    card_content = f"""---
tags:
- dpo
- fine-tuned
- email-generation
base_model: {base_model_id}
library_name: transformers
pipeline_tag: text-generation
---

# {model_name.title()} DPO Fine-tuned

This model is a DPO (Direct Preference Optimization) fine-tuned version of [{base_model_id}](https://huggingface.co/{base_model_id}).

## Model Details

- **Base Model**: {base_model_id}
- **Fine-tuning Method**: DPO (Direct Preference Optimization)
- **Task**: Email Generation
- **Training Data**: Preference pairs for email generation task
- **Training Date**: {training_info.get('timestamp', 'Unknown')}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}-dpo")
tokenizer = AutoTokenizer.from_pretrained("{model_name}-dpo")

# Generate email
prompt = "Write a professional email about..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500)
email = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

- **Method**: Direct Preference Optimization (DPO)
- **LoRA Configuration**: Rank 8, Alpha 16
- **Training Steps**: Based on dataset size
- **Beta Parameter**: 0.1

## Intended Use

This model is designed for generating professional emails in charity/fundraising contexts.
It has been fine-tuned to align better with human preferences for email quality and style.

## Limitations

- Specific to email generation tasks
- Trained on charity/fundraising domain
- May not generalize to other text generation tasks

## Citation

If you use this model, please cite:
```
@misc{{{model_name}-dpo,
  title={{{model_name.title()} DPO Fine-tuned for Email Generation}},
  author={{Your Name}},
  year={{2025}},
  howpublished={{\\url{{https://huggingface.co/{model_name}-dpo}}}}
}}
```
"""
    return card_content

def merge_and_upload_model(model_info, hf_username, dry_run=False):
    """Merge LoRA adapter with base model and upload to HF Hub with model card"""
    model_name = model_info['name']
    model_path = model_info['path']
    
    print(f"\\n🔄 Processing {model_name}...")
    
    # Load registry to get base model info
    registry = load_model_registry()
    if model_name not in registry['models']:
        print(f"❌ Model {model_name} not found in registry")
        return False
    
    base_model_id = registry['models'][model_name]['base_model']
    print(f"📦 Base model: {base_model_id}")
    print(f"🎯 LoRA adapter: {model_path}")
    
    # HF Hub model name
    hf_model_name = f"{hf_username}/{model_name}-dpo"
    print(f"🚀 Target HF Hub: {hf_model_name}")
    
    if dry_run:
        print("🔍 DRY RUN - Would merge and upload but not executing")
        print(f"📝 Would create model card for {hf_model_name}")
        return True
    
    try:
        # Import required libraries
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print("📥 Loading base model...")
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("🔧 Loading LoRA adapter...")
        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, model_path)
        merged_model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        print("📝 Creating model card...")
        # Create model card
        model_card = create_model_card(model_name, base_model_id, model_info)
        
        print("📤 Uploading to Hugging Face Hub...")
        # Upload model and tokenizer
        merged_model.push_to_hub(hf_model_name, use_temp_dir=True)
        tokenizer.push_to_hub(hf_model_name, use_temp_dir=True)
        
        # Upload model card
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_id=hf_model_name,
                repo_type="model"
            )
            print("📝 Model card uploaded")
        except Exception as card_error:
            print(f"⚠️  Model uploaded but model card failed: {card_error}")
        
        print(f"✅ Successfully uploaded {hf_model_name}")
        
        # Cleanup
        del merged_model, base_model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to merge/upload {model_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Merge LoRA adapters and upload to HF Hub')
    parser.add_argument('--username', required=True, help='Your Hugging Face username')
    parser.add_argument('--model', help='Specific model to process (e.g., tinyllama)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--all', action='store_true', help='Process all trained models')
    
    args = parser.parse_args()
    
    print("🔍 Scanning for trained DPO models...")
    trained_models = find_trained_models()
    
    if not trained_models:
        print("❌ No trained models found!")
        print("Train some models first using: sbatch dpo/slurm/train_single.sh")
        return
    
    print(f"✅ Found {len(trained_models)} trained models:")
    for model in trained_models:
        print(f"  - {model['name']} (trained: {model['timestamp']})")
    
    # Filter models to process
    models_to_process = []
    if args.model:
        models_to_process = [m for m in trained_models if m['name'] == args.model]
        if not models_to_process:
            print(f"❌ Model '{args.model}' not found in trained models")
            return
    elif args.all:
        models_to_process = trained_models
    else:
        # Default: process first trained model
        models_to_process = [trained_models[0]]
        print(f"🎯 Processing first model: {trained_models[0]['name']}")
        print("Use --all to process all models or --model <name> for specific model")
    
    print(f"\\n🚀 Starting merge and upload process...")
    print(f"HF Username: {args.username}")
    
    success_count = 0
    for model_info in models_to_process:
        if merge_and_upload_model(model_info, args.username, args.dry_run):
            success_count += 1
    
    print(f"\\n📊 Summary: {success_count}/{len(models_to_process)} models successfully processed")
    
    if not args.dry_run and success_count > 0:
        print("\\n🎉 Next steps:")
        print("1. Add model entries to config/config.py")
        print("2. Test with: python -m runner --email_models <model-name-dpo>")

if __name__ == "__main__":
    main()