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
    """Process complete_results.json into DPO format using existing ranks"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    dpo_samples = []
    total_topics = len(data.get('results', []))
    
    print(f"Processing {total_topics} topics from {results_file}")
    
    for i, result in enumerate(data.get('results', [])):
        emails = result.get('emails', [])
        print(f"Topic {i+1}/{total_topics}: Found {len(emails)} emails")
        
        # Debug: Check available ranks
        if emails:
            ranks = [email.get('rank', 'N/A') for email in emails]
            print(f"  Available ranks: {ranks}")
        
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
        
        # Sort by rank (rank 1 = best, higher rank = worse)
        sorted_emails = sorted(
            filtered_emails, 
            key=lambda x: x.get('rank', float('inf'))  # Put emails without rank at the end
        )
        
        best_email = sorted_emails[0]  # Rank 1 (best)
        worst_email = sorted_emails[-1]  # Highest rank (worst)
        
        best_rank = best_email.get('rank', 'N/A')
        worst_rank = worst_email.get('rank', 'N/A')
        
        print(f"  Best rank: {best_rank}, Worst rank: {worst_rank}")
        
        # Skip if ranks are too close (optional - can be removed if not needed)
        if isinstance(best_rank, int) and isinstance(worst_rank, int):
            rank_difference = worst_rank - best_rank
            if rank_difference < config['dataset'].get('min_rank_difference', 1):
                print(f"  Skipping: Rank difference {rank_difference} too small")
                continue
        
        # Extract topic information for prompt
        topic_info = result.get('topic', 'Unknown Topic')
        
        # topic_info is a string in this data format
        if isinstance(topic_info, str):
            topic_name = topic_info
        elif isinstance(topic_info, dict):
            topic_name = topic_info.get('name', 'Unknown Topic')
        else:
            topic_name = 'Unknown Topic'
        
        # Create a proper prompt based on the topic
        prompt = f"Generate a fundraising email for: {topic_name}"
        
        dpo_sample = {
            'prompt': prompt,
            'chosen': best_email.get('email_content', ''),
            'rejected': worst_email.get('email_content', ''),
            'chosen_rank': best_rank,
            'rejected_rank': worst_rank,
            'chosen_model': best_email.get('model_name', ''),
            'rejected_model': worst_email.get('model_name', ''),
            'topic_info': topic_info
        }
        
        dpo_samples.append(dpo_sample)
        print(f"  ✅ Created DPO sample: rank {best_rank} vs {worst_rank}")
    
    print(f"\nTotal DPO samples created: {len(dpo_samples)}")
    return dpo_samples

# Removed get_email_score function - now using ranks directly

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
        print("2. Insufficient rank differences between emails")
        print("3. Less than 2 emails per topic after filtering")
        print("4. Missing rank information in the data")
        
        # Create empty file
        with open(output_file, 'w') as f:
            pass
        
        # Save empty metadata
        metadata_file = output_file.replace('.jsonl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            metadata = {
                'total_samples': 0,
                'avg_rank_difference': 0.0,
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
        # Calculate average rank difference (higher = better separation)
        rank_differences = []
        for s in samples:
            if isinstance(s.get('chosen_rank'), int) and isinstance(s.get('rejected_rank'), int):
                rank_differences.append(s['rejected_rank'] - s['chosen_rank'])
        
        avg_rank_diff = sum(rank_differences) / len(rank_differences) if rank_differences else 0
        
        metadata = {
            'total_samples': len(samples),
            'avg_rank_difference': avg_rank_diff,
            'rank_differences': rank_differences,
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