#!/usr/bin/env python3
"""
DPO Training Script
Fine-tune language models using Direct Preference Optimization
"""

import argparse
import json
import yaml
import torch
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from model_manager import ModelManager

def load_config(config_path: str) -> Dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dpo_dataset(data_file: str, tokenizer) -> Dataset:
    """Load and tokenize DPO dataset"""
    samples = []
    with open(data_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    dataset = Dataset.from_list(samples)
    
    def tokenize_function(examples):
        # Tokenize prompt, chosen, and rejected responses
        model_inputs = {
            'prompt': examples['prompt'],
            'chosen': examples['chosen'], 
            'rejected': examples['rejected']
        }
        return model_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def setup_model_and_tokenizer(config: Dict, cache_dir: str = "../downloaded_models"):
    """Setup model, tokenizer with quantization and LoRA"""
    
    # Quantization config
    if config['model']['use_quantization']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model and tokenizer from cache
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_dpo_trainer(model, tokenizer, train_dataset, val_dataset, config: Dict, output_dir: str):
    """Create and configure DPO trainer"""
    
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"dpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        beta=config['dpo']['beta'],
        max_length=config['dpo']['max_length'],
    )
    
    return trainer

def load_model_registry():
    """Load model registry for multi-model support"""
    with open('configs/model_registry.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_single_model(data_file: str, model_key: str, output_base_dir: str, resume_checkpoint: str = None, cache_dir: str = "../downloaded_models"):
    """Train a single model with its specific configuration"""
    registry = load_model_registry()
    
    if model_key not in registry['models']:
        raise ValueError(f"Model {model_key} not found in registry. Available: {list(registry['models'].keys())}")
    
    model_info = registry['models'][model_key]
    config_file = model_info['config_file']
    
    print(f"Training {model_info['name']} ({model_info['size']})...")
    print(f"Using config: {config_file}")
    
    # Initialize model manager and ensure model availability
    print("Checking model availability...")
    model_manager = ModelManager(cache_dir)
    available_models, failed_models = model_manager.ensure_models_available([model_key])
    
    if model_key in failed_models:
        raise RuntimeError(f"Failed to download model {model_key}")
    
    print(f"✅ Model {model_key} is available")
    
    # Load model-specific configuration
    config = load_config(config_file)
    
    # Create model-specific output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_base_dir, f"dpo_{model_key}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Base model: {config['model']['base_model']}")
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config, cache_dir)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dpo_dataset(data_file, tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("Setting up trainer...")
    trainer = create_dpo_trainer(model, tokenizer, train_dataset, val_dataset, config, output_dir)
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training config and model info
    with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    with open(os.path.join(output_dir, 'model_info.yaml'), 'w') as f:
        yaml.dump(model_info, f)
    
    print(f"Training completed! Model saved to {output_dir}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Train DPO model(s) for email generation')
    parser.add_argument('--data-file', required=True,
                       help='Path to DPO training data JSONL file')
    parser.add_argument('--model', default=None,
                       help='Specific model to train (e.g., tinyllama, vicuna, phi3, llama3, stablelm)')
    parser.add_argument('--models', default=None,
                       help='Comma-separated list of models to train (e.g., tinyllama,phi3)')
    parser.add_argument('--config', default='configs/training_config.yaml',
                       help='Training configuration file (ignored when using --model or --models)')
    parser.add_argument('--output-dir', default='outputs/models',
                       help='Base output directory for model checkpoints')
    parser.add_argument('--resume-from-checkpoint', default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--cache-dir', default='../downloaded_models',
                       help='Directory for cached models (default: ../downloaded_models)')
    
    args = parser.parse_args()
    
    # Multi-model training mode
    if args.model or args.models:
        models_to_train = []
        
        if args.model:
            models_to_train = [args.model]
        elif args.models:
            models_to_train = [m.strip() for m in args.models.split(',')]
        
        print(f"Multi-model training mode: {len(models_to_train)} models")
        print(f"Models: {models_to_train}")
        
        # Check and ensure all models are available before training
        print("Checking model availability for all models...")
        model_manager = ModelManager(args.cache_dir)
        available_models, failed_models = model_manager.ensure_models_available(models_to_train)
        
        if failed_models:
            print(f"❌ Failed to download models: {failed_models}")
            print("Continuing with available models only...")
            models_to_train = available_models
        
        print(f"✅ Ready to train {len(models_to_train)} models: {models_to_train}")
        
        trained_models = []
        for model_key in models_to_train:
            try:
                output_dir = train_single_model(
                    args.data_file, 
                    model_key, 
                    args.output_dir, 
                    args.resume_from_checkpoint,
                    args.cache_dir
                )
                trained_models.append((model_key, output_dir))
                print(f"✅ {model_key} training completed: {output_dir}")
            except Exception as e:
                print(f"❌ {model_key} training failed: {e}")
        
        print(f"\n🎉 Multi-model training summary:")
        for model_key, output_dir in trained_models:
            print(f"  {model_key}: {output_dir}")
        
        return
    
    # Legacy single-model mode (for backward compatibility)
    print("Legacy single-model training mode")
    config = load_config(args.config)
    
    # Set output directory
    if args.output_dir == 'outputs/models':  # Default value
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_output_dir = f"outputs/models/dpo_model_{timestamp}"
    else:
        full_output_dir = args.output_dir
    
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"Starting DPO training...")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {full_output_dir}")
    print(f"Base model: {config['model']['base_model']}")
    
    # Setup model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config, args.cache_dir)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dpo_dataset(args.data_file, tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create trainer
    print("Setting up trainer...")
    trainer = create_dpo_trainer(model, tokenizer, train_dataset, val_dataset, config, full_output_dir)
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(full_output_dir)
    
    # Save training config
    with open(os.path.join(full_output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print(f"Training completed! Model saved to {full_output_dir}")

if __name__ == "__main__":
    main()