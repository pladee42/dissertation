#!/usr/bin/env python3
"""
Basic Evaluation Script for DPO Comparisons
Simple metrics: output length, response time, basic quality checks
"""

import os
import sys
import json
import csv
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.config import get_model_pairs, get_model_config

def calculate_basic_metrics(text: str) -> Dict:
    """Calculate basic text metrics"""
    if not text:
        return {
            'length_chars': 0,
            'length_words': 0,
            'length_lines': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'paragraph_count': 0
        }
    
    # Basic counts
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    line_count = len(text.split('\\n'))
    
    # Average word length
    avg_word_length = sum(len(word.strip('.,!?;:"()')) for word in words) / word_count if word_count > 0 else 0
    
    # Sentence count (simple approximation)
    sentence_count = len(re.findall(r'[.!?]+', text))
    
    # Paragraph count
    paragraph_count = len([p for p in text.split('\\n\\n') if p.strip()])
    
    return {
        'length_chars': char_count,
        'length_words': word_count,
        'length_lines': line_count,
        'avg_word_length': round(avg_word_length, 2),
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count
    }

def calculate_quality_metrics(text: str) -> Dict:
    """Calculate basic quality indicators"""
    if not text:
        return {
            'has_greeting': False,
            'has_closing': False,
            'has_call_to_action': False,
            'professional_tone': False,
            'completeness_score': 0
        }
    
    text_lower = text.lower()
    
    # Basic email structure checks
    greeting_words = ['dear', 'hello', 'hi', 'greetings']
    has_greeting = any(word in text_lower for word in greeting_words)
    
    closing_words = ['sincerely', 'regards', 'best', 'thank you', 'thanks']
    has_closing = any(word in text_lower for word in closing_words)
    
    # Call to action indicators
    action_phrases = ['donate', 'support', 'help', 'contribute', 'visit', 'contact', 'learn more']
    has_call_to_action = any(phrase in text_lower for phrase in action_phrases)
    
    # Professional tone indicators
    professional_words = ['organization', 'foundation', 'program', 'initiative', 'community']
    professional_tone = any(word in text_lower for word in professional_words)
    
    # Completeness score (0-100)
    completeness_score = 0
    if has_greeting: completeness_score += 25
    if has_closing: completeness_score += 25
    if has_call_to_action: completeness_score += 25
    if len(text.split()) >= 50: completeness_score += 25  # Reasonable length
    
    return {
        'has_greeting': has_greeting,
        'has_closing': has_closing,
        'has_call_to_action': has_call_to_action,
        'professional_tone': professional_tone,
        'completeness_score': completeness_score
    }

def extract_email_content(output_file: Path) -> str:
    """Extract email content from model output file"""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Find email content between markers
        if "GENERATED EMAIL:" in content:
            start_marker = "GENERATED EMAIL:"
            end_marker = "BASIC METRICS:"
            
            start_idx = content.find(start_marker)
            if start_idx != -1:
                start_idx = content.find('-'*30, start_idx) + 30
                end_idx = content.find(end_marker)
                if end_idx == -1:
                    end_idx = len(content)
                
                email_content = content[start_idx:end_idx].strip()
                return email_content
        
        return ""
    except Exception as e:
        print(f"⚠️ Error reading {output_file}: {e}")
        return ""

def evaluate_comparison_directory(comparison_dir: Path) -> Dict:
    """Evaluate a single comparison directory"""
    print(f"📊 Evaluating: {comparison_dir.name}")
    
    # Load summary
    summary_file = comparison_dir / "comparison_summary.json"
    if not summary_file.exists():
        print(f"❌ No summary file found in {comparison_dir}")
        return None
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    base_model = summary['base_model']
    dpo_model = summary['dpo_model']
    
    # Find output files
    base_output_file = comparison_dir / summary['files']['base_output']
    dpo_output_file = comparison_dir / summary['files']['dpo_output']
    
    # Extract email content
    base_email = extract_email_content(base_output_file)
    dpo_email = extract_email_content(dpo_output_file)
    
    # Calculate metrics
    base_basic = calculate_basic_metrics(base_email)
    dpo_basic = calculate_basic_metrics(dpo_email)
    
    base_quality = calculate_quality_metrics(base_email)
    dpo_quality = calculate_quality_metrics(dpo_email)
    
    # Comparison metrics
    length_diff = dpo_basic['length_words'] - base_basic['length_words']
    quality_diff = dpo_quality['completeness_score'] - base_quality['completeness_score']
    
    evaluation = {
        'comparison_id': comparison_dir.name,
        'timestamp': summary['timestamp'],
        'base_model': base_model,
        'dpo_model': dpo_model,
        'topic': summary.get('topic', 'Unknown'),
        'base_metrics': {**base_basic, **base_quality},
        'dpo_metrics': {**dpo_basic, **dpo_quality},
        'comparison': {
            'length_difference': length_diff,
            'quality_difference': quality_diff,
            'dpo_longer': length_diff > 0,
            'dpo_better_quality': quality_diff > 0,
            'both_successful': summary['success']['base'] and summary['success']['dpo']
        }
    }
    
    print(f"  ✅ Base: {base_basic['length_words']} words, Quality: {base_quality['completeness_score']}")
    print(f"  ✅ DPO:  {dpo_basic['length_words']} words, Quality: {dpo_quality['completeness_score']}")
    print(f"  📈 Difference: {length_diff:+d} words, {quality_diff:+d} quality points")
    
    return evaluation

def generate_comparison_report(evaluations: List[Dict], output_file: Path):
    """Generate CSV report of all evaluations"""
    if not evaluations:
        print("❌ No evaluations to report")
        return
    
    fieldnames = [
        'comparison_id', 'timestamp', 'base_model', 'dpo_model', 'topic',
        'base_length_words', 'base_completeness_score',
        'dpo_length_words', 'dpo_completeness_score',
        'length_difference', 'quality_difference',
        'dpo_longer', 'dpo_better_quality', 'both_successful'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for eval_data in evaluations:
            row = {
                'comparison_id': eval_data['comparison_id'],
                'timestamp': eval_data['timestamp'],
                'base_model': eval_data['base_model'],
                'dpo_model': eval_data['dpo_model'],
                'topic': eval_data['topic'],
                'base_length_words': eval_data['base_metrics']['length_words'],
                'base_completeness_score': eval_data['base_metrics']['completeness_score'],
                'dpo_length_words': eval_data['dpo_metrics']['length_words'],
                'dpo_completeness_score': eval_data['dpo_metrics']['completeness_score'],
                'length_difference': eval_data['comparison']['length_difference'],
                'quality_difference': eval_data['comparison']['quality_difference'],
                'dpo_longer': eval_data['comparison']['dpo_longer'],
                'dpo_better_quality': eval_data['comparison']['dpo_better_quality'],
                'both_successful': eval_data['comparison']['both_successful']
            }
            writer.writerow(row)
    
    print(f"📊 Report saved: {output_file}")

def generate_summary_stats(evaluations: List[Dict]):
    """Generate summary statistics"""
    if not evaluations:
        return
    
    total_comparisons = len(evaluations)
    successful_comparisons = sum(1 for e in evaluations if e['comparison']['both_successful'])
    
    # DPO improvements
    dpo_longer_count = sum(1 for e in evaluations if e['comparison']['dpo_longer'])
    dpo_better_quality_count = sum(1 for e in evaluations if e['comparison']['dpo_better_quality'])
    
    # Average differences
    avg_length_diff = sum(e['comparison']['length_difference'] for e in evaluations) / total_comparisons
    avg_quality_diff = sum(e['comparison']['quality_difference'] for e in evaluations) / total_comparisons
    
    print(f"\\n📈 SUMMARY STATISTICS")
    print(f"{'='*50}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Successful comparisons: {successful_comparisons}")
    print(f"DPO models longer: {dpo_longer_count}/{total_comparisons} ({dpo_longer_count/total_comparisons*100:.1f}%)")
    print(f"DPO models better quality: {dpo_better_quality_count}/{total_comparisons} ({dpo_better_quality_count/total_comparisons*100:.1f}%)")
    print(f"Average length difference: {avg_length_diff:+.1f} words")
    print(f"Average quality difference: {avg_quality_diff:+.1f} points")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DPO model comparisons')
    parser.add_argument('--comparison-dir', help='Specific comparison directory to evaluate')
    parser.add_argument('--all', action='store_true', help='Evaluate all comparison directories')
    parser.add_argument('--output', default='comparison_evaluation.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    comparisons_root = Path("../outputs/comparisons")
    
    if not comparisons_root.exists():
        print(f"❌ Comparisons directory not found: {comparisons_root}")
        print("💡 Run some comparisons first using: python compare_models.py")
        return
    
    evaluations = []
    
    if args.comparison_dir:
        # Evaluate single directory
        comparison_path = comparisons_root / args.comparison_dir
        if comparison_path.exists():
            eval_result = evaluate_comparison_directory(comparison_path)
            if eval_result:
                evaluations.append(eval_result)
        else:
            print(f"❌ Comparison directory not found: {comparison_path}")
            return
    
    elif args.all:
        # Evaluate all comparison directories
        comparison_dirs = [d for d in comparisons_root.iterdir() if d.is_dir() and d.name.startswith('comparison_')]
        
        if not comparison_dirs:
            print(f"❌ No comparison directories found in {comparisons_root}")
            return
        
        print(f"🔍 Found {len(comparison_dirs)} comparison directories")
        
        for comparison_dir in sorted(comparison_dirs):
            eval_result = evaluate_comparison_directory(comparison_dir)
            if eval_result:
                evaluations.append(eval_result)
    
    else:
        print("❌ Please specify --comparison-dir <name> or --all")
        return
    
    if evaluations:
        # Generate report
        output_file = Path(args.output)
        generate_comparison_report(evaluations, output_file)
        generate_summary_stats(evaluations)
        
        print(f"\\n🎉 Evaluation complete!")
        print(f"📊 {len(evaluations)} comparisons evaluated")
        print(f"📈 Report saved: {output_file}")
    else:
        print("❌ No evaluations completed")

if __name__ == "__main__":
    main()