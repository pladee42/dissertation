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
    total_topics = len(data.get('results', []))
    
    print(f"Processing {total_topics} topics from {results_file}")
    
    for i, result in enumerate(data.get('results', [])):
        emails = result.get('emails', [])
        print(f"Topic {i+1}/{total_topics}: Found {len(emails)} emails")
        
        # Debug: Check email structure
        if emails:
            sample_email = emails[0]
            print(f"  Sample email keys: {list(sample_email.keys())}")
            if 'evaluation' in sample_email:
                eval_keys = list(sample_email['evaluation'].keys())
                print(f"  Evaluation keys: {eval_keys}")
        
        # Filter emails by length
        filtered_emails = [
            email for email in emails 
            if (config['filtering']['min_email_length'] <= 
                len(email.get('email_content', '')) <= 
                config['filtering']['max_email_length'])
        ]
        
        print(f"  After length filtering: {len(filtered_emails)} emails")
        
        if len(filtered_emails) < 2:
            print(f"  Skipping: Need at least 2 emails for comparison")
            continue  # Need at least 2 emails for comparison
        
        # Sort by weighted_score (highest first) - with fallback scoring
        sorted_emails = sorted(
            filtered_emails, 
            key=lambda x: get_email_score(x),
            reverse=True
        )
        
        best_email = sorted_emails[0]
        worst_email = sorted_emails[-1]
        
        # Check minimum score difference
        best_score = get_email_score(best_email)
        worst_score = get_email_score(worst_email)
        
        print(f"  Best score: {best_score}, Worst score: {worst_score}")
        
        if best_score - worst_score < config['dataset']['min_score_difference']:
            print(f"  Skipping: Score difference {best_score - worst_score} < {config['dataset']['min_score_difference']}")
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
        print(f"  ✅ Created DPO sample for topic {i+1}")
    
    print(f"\nTotal DPO samples created: {len(dpo_samples)}")
    return dpo_samples

def get_email_score(email: Dict) -> float:
    """Extract score from email with fallbacks"""
    evaluation = email.get('evaluation', {})
    
    # Try weighted_score first
    if 'weighted_score' in evaluation:
        return evaluation['weighted_score']
    
    # Try final_score
    if 'final_score' in evaluation:
        return evaluation['final_score']
    
    # Calculate from checklist scores
    if 'checklist_scores' in evaluation:
        checklist_scores = evaluation['checklist_scores']
        if checklist_scores:
            # Calculate average confidence of 'yes' answers
            yes_scores = [item['confidence'] for item in checklist_scores if item.get('result') == 'yes']
            return sum(yes_scores) / len(checklist_scores) if checklist_scores else 0.0
    
    # Default fallback
    return 0.0

def save_dpo_dataset(samples: List[Dict], output_file: str, config: Dict):
    """Save DPO samples to JSONL format"""
    # Limit samples per topic if specified
    max_samples = config['dataset'].get('max_samples_per_topic')
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Handle empty samples case
    if not samples:
        print("WARNING: No valid DPO samples were generated!")
        print("This could be due to:")
        print("1. All emails being too short/long (check filtering criteria)")
        print("2. Insufficient score differences between emails")
        print("3. Data structure mismatch")
        
        # Create empty file
        with open(output_file, 'w') as f:
            pass
        
        # Save empty metadata
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            metadata = {
                'total_samples': 0,
                'avg_score_difference': 0.0,
                'warning': 'No valid samples generated',
                'samples_with_metadata': []
            }
            json.dump(metadata, f, indent=2)
        return
    
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