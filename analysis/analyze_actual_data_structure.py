#!/usr/bin/env python3
"""
Analyze the actual data structure to understand correct sample sizes.
"""

import json
from pathlib import Path
import numpy as np

def analyze_data_file(filepath, label):
    """Analyze a single complete_results.json file."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label}: {filepath}")
    print('='*60)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Total topics: {data.get('total_topics', 'N/A')}")
    print(f"Successful topics: {len(data.get('successful_results', []))}")
    
    # Count emails by model
    model_counts = {}
    all_scores = []
    model_scores = {}
    
    for result in data.get('successful_results', []):
        if 'emails' in result:
            for email in result['emails']:
                model_id = email.get('model_id', email.get('model_name', 'unknown'))
                
                # Simplify model names
                if 'TinyLlama' in model_id:
                    model_key = 'M0001_TinyLlama'
                elif 'vicuna' in model_id:
                    model_key = 'M0002_Vicuna'
                elif 'Phi-3' in model_id:
                    model_key = 'M0003_Phi3'
                elif 'llama-3-8b' in model_id:
                    model_key = 'M0004_Llama3'
                elif 'stablelm' in model_id:
                    model_key = 'M0005_StableLM'
                else:
                    model_key = model_id
                
                model_counts[model_key] = model_counts.get(model_key, 0) + 1
                
                # Collect scores
                score = email.get('overall_score', None)
                if score is not None:
                    all_scores.append(score)
                    if model_key not in model_scores:
                        model_scores[model_key] = []
                    model_scores[model_key].append(score)
    
    print("\nEmail counts by model:")
    total_emails = 0
    for model, count in sorted(model_counts.items()):
        print(f"  {model}: {count}")
        total_emails += count
    
    print(f"\nTotal emails: {total_emails}")
    
    if all_scores:
        print(f"\nOverall statistics:")
        print(f"  N = {len(all_scores)}")
        print(f"  Mean = {np.mean(all_scores):.4f}")
        print(f"  SD = {np.std(all_scores, ddof=1):.4f}")
        print(f"  Min = {np.min(all_scores):.4f}")
        print(f"  Max = {np.max(all_scores):.4f}")
        
        print(f"\nPer-model statistics:")
        for model, scores in sorted(model_scores.items()):
            if scores:
                print(f"  {model}:")
                print(f"    N = {len(scores)}")
                print(f"    Mean = {np.mean(scores):.4f}")
                print(f"    SD = {np.std(scores, ddof=1):.4f}")
    
    return {
        'total_topics': data.get('total_topics', 0),
        'successful_topics': len(data.get('successful_results', [])),
        'total_emails': total_emails,
        'model_counts': model_counts,
        'all_scores': all_scores,
        'model_scores': model_scores
    }

def main():
    """Analyze all three data files."""
    print("\n" + "="*70)
    print("ACTUAL DATA STRUCTURE ANALYSIS")
    print("="*70)
    
    # Data files
    files = {
        'Baseline': 'output/multi_topic_results/20250722_061212/complete_results.json',
        'DPO-Synthetic': 'output/multi_topic_results/20250722_123509/complete_results.json',
        'DPO-Hybrid': 'output/multi_topic_results/20250731_164142/complete_results.json'
    }
    
    all_results = {}
    
    for label, filepath in files.items():
        if Path(filepath).exists():
            all_results[label] = analyze_data_file(filepath, label)
        else:
            print(f"\nWARNING: {label} file not found at {filepath}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print("\nSample sizes across conditions:")
    print("-" * 40)
    for condition in ['Baseline', 'DPO-Synthetic', 'DPO-Hybrid']:
        if condition in all_results:
            result = all_results[condition]
            print(f"\n{condition}:")
            print(f"  Topics: {result['successful_topics']}")
            print(f"  Total evaluations: {result['total_emails']}")
            print(f"  Expected (50 topics × 5 models): 250")
            
            if result['all_scores']:
                print(f"  Actual N with scores: {len(result['all_scores'])}")
    
    # Check for consistency
    print("\n" + "="*70)
    print("DATA CONSISTENCY CHECK")
    print("="*70)
    
    if all(cond in all_results for cond in ['Baseline', 'DPO-Synthetic', 'DPO-Hybrid']):
        baseline_n = len(all_results['Baseline']['all_scores'])
        synthetic_n = len(all_results['DPO-Synthetic']['all_scores'])
        hybrid_n = len(all_results['DPO-Hybrid']['all_scores'])
        
        print(f"\nActual sample sizes:")
        print(f"  Baseline: N = {baseline_n}")
        print(f"  DPO-Synthetic: N = {synthetic_n}")
        print(f"  DPO-Hybrid: N = {hybrid_n}")
        
        if baseline_n == synthetic_n == hybrid_n:
            print(f"\n✓ Sample sizes are EQUAL across conditions (N = {baseline_n})")
        else:
            print(f"\n✗ Sample sizes are UNEQUAL across conditions")
            print(f"  This violates balanced design assumption!")
        
        # Calculate correct ANOVA degrees of freedom
        total_n = baseline_n + synthetic_n + hybrid_n
        df_between = 2  # 3 groups - 1
        df_within = total_n - 3  # Total N - number of groups
        
        print(f"\nCorrect ANOVA degrees of freedom:")
        print(f"  df(between) = {df_between}")
        print(f"  df(within) = {df_within}")
        print(f"  F({df_between}, {df_within})")
        
        print(f"\nThe paper currently reports:")
        print(f"  N = 145 (INCORRECT)")
        print(f"  F(2, 432) (INCORRECT)")
        
        print(f"\nShould be:")
        print(f"  N = {baseline_n} per condition")
        print(f"  F({df_between}, {df_within})")

if __name__ == "__main__":
    main()