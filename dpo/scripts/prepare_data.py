#!/usr/bin/env python3
"""
DPO Data Preparation Script
Converts multi_topic_results to DPO training format
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from typing import List, Dict, Tuple

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_prompt_from_email(email_content: str) -> str:
    """Extract clean prompt from email content"""
    # Remove email headers and extract core topic
    lines = email_content.split('\n')
    # Look for subject line or first meaningful content
    for line in lines:
        if line.startswith('Subject:'):
            return line.replace('Subject:', '').strip()
    
    # Fallback: use first paragraph
    return lines[0][:200] + "..." if len(lines[0]) > 200 else lines[0]

def process_complete_results(results_file: str, config: Dict) -> List[Dict]:
    """Process complete_results.json into DPO format"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    dpo_samples = []
    
    for result in data.get('results', []):
        emails = result.get('emails', [])
        
        # Filter emails by length
        filtered_emails = [
            email for email in emails 
            if (config['filtering']['min_email_length'] <= 
                len(email.get('email_content', '')) <= 
                config['filtering']['max_email_length'])
        ]
        
        if len(filtered_emails) < 2:
            continue  # Need at least 2 emails for comparison
        
        # Sort by weighted_score (highest first)
        sorted_emails = sorted(
            filtered_emails, 
            key=lambda x: x.get('evaluation', {}).get('weighted_score', 0),
            reverse=True
        )
        
        best_email = sorted_emails[0]
        worst_email = sorted_emails[-1]
        
        # Check minimum score difference
        best_score = best_email.get('evaluation', {}).get('weighted_score', 0)
        worst_score = worst_email.get('evaluation', {}).get('weighted_score', 0)
        
        if best_score - worst_score < config['dataset']['min_score_difference']:
            continue
        
        # Extract prompt from best email (they should be similar)
        prompt = extract_prompt_from_email(best_email.get('email_content', ''))
        
        dpo_sample = {
            'prompt': prompt,
            'chosen': best_email.get('email_content', ''),
            'rejected': worst_email.get('email_content', ''),
            'chosen_score': best_score,
            'rejected_score': worst_score,
            'chosen_model': best_email.get('model_name', ''),
            'rejected_model': worst_email.get('model_name', ''),
            'topic_info': result.get('topic', {})
        }
        
        dpo_samples.append(dpo_sample)
    
    return dpo_samples

def save_dpo_dataset(samples: List[Dict], output_file: str, config: Dict):
    """Save DPO samples to JSONL format"""
    # Limit samples per topic if specified
    max_samples = config['dataset'].get('max_samples_per_topic')
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for sample in samples:
            # Clean format for DPO training
            clean_sample = {
                'prompt': sample['prompt'],
                'chosen': sample['chosen'],
                'rejected': sample['rejected']
            }
            f.write(json.dumps(clean_sample) + '\n')
    
    # Save metadata separately
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        metadata = {
            'total_samples': len(samples),
            'avg_score_difference': sum(s['chosen_score'] - s['rejected_score'] for s in samples) / len(samples),
            'samples_with_metadata': samples
        }
        json.dump(metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Prepare DPO training data from multi_topic_results')
    parser.add_argument('--input-folder', required=True, 
                       help='Path to multi_topic_results folder (e.g., output/multi_topic_results/20250714_043247)')
    parser.add_argument('--output-file', default='outputs/datasets/dpo_data.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--config', default='configs/data_config.yaml',
                       help='Data configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Find complete_results.json in the input folder
    results_file = os.path.join(args.input_folder, 'complete_results.json')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"complete_results.json not found in {args.input_folder}")
    
    print(f"Processing {results_file}...")
    
    # Process data
    dpo_samples = process_complete_results(results_file, config)
    
    print(f"Generated {len(dpo_samples)} DPO training samples")
    
    # Save dataset
    save_dpo_dataset(dpo_samples, args.output_file, config)
    
    print(f"DPO dataset saved to {args.output_file}")
    print(f"Metadata saved to {args.output_file.replace('.jsonl', '_metadata.json')}")

if __name__ == "__main__":
    main()