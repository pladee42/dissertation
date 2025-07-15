#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluate fine-tuned DPO model performance
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_path: str, adapter_path: str = None):
    """Load base model and optional adapter"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer

def generate_email(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate email using the model"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

def main():
    parser = argparse.ArgumentParser(description='Evaluate DPO fine-tuned model')
    parser.add_argument('--model-path', required=True,
                       help='Path to fine-tuned model directory')
    parser.add_argument('--base-model', default='meta-llama/Llama-3-8b-instruct',
                       help='Base model name')
    parser.add_argument('--prompt', default='Write an email about supporting wildlife conservation',
                       help='Test prompt for generation')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.base_model, args.model_path)
    
    print(f"Generating email for prompt: {args.prompt}")
    generated_email = generate_email(model, tokenizer, args.prompt)
    
    print("\n" + "="*50)
    print("Generated Email:")
    print("="*50)
    print(generated_email)
    print("="*50)

if __name__ == "__main__":
    main()