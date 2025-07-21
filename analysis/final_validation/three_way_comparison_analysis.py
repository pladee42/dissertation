#!/usr/bin/env python3
"""
Three-Way Comparison Analysis for Final Validation Protocol
Implements paired t-tests, ANOVA, and effect size analysis
"""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
from typing import Dict, List, Tuple
from effect_size_calculator import interpret_effect_sizes, eta_squared

def load_results_data(file_path: str) -> List[float]:
    """
    Load scores from complete_results.json file
    
    Args:
        file_path: Path to complete_results.json
    
    Returns:
        List of overall scores from the results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        scores = []
        if 'results' in data:
            for topic_result in data['results']:
                for email in topic_result.get('emails', []):
                    if 'evaluation' in email and 'overall_score' in email['evaluation']:
                        scores.append(email['evaluation']['overall_score'])
        
        return scores
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def validate_data_quality(baseline: List[float], synthetic: List[float], hybrid: List[float]) -> Dict:
    """
    Validate data quality and sample sizes
    
    Args:
        baseline: Baseline model scores
        synthetic: DPO-Synthetic scores  
        hybrid: DPO-Hybrid scores
    
    Returns:
        Dict with validation results
    """
    validation = {
        'baseline_n': len(baseline),
        'synthetic_n': len(synthetic), 
        'hybrid_n': len(hybrid),
        'equal_sizes': len(baseline) == len(synthetic) == len(hybrid),
        'min_sample_size': min(len(baseline), len(synthetic), len(hybrid)),
        'adequate_power': min(len(baseline), len(synthetic), len(hybrid)) >= 30
    }
    
    # Check for missing data
    validation['baseline_complete'] = len(baseline) > 0 and not any(np.isnan(baseline))
    validation['synthetic_complete'] = len(synthetic) > 0 and not any(np.isnan(synthetic))
    validation['hybrid_complete'] = len(hybrid) > 0 and not any(np.isnan(hybrid))
    
    return validation

def perform_paired_t_tests(baseline: List[float], synthetic: List[float], hybrid: List[float]) -> Dict:
    """
    Perform paired t-tests for all comparisons
    
    Args:
        baseline: Baseline model scores
        synthetic: DPO-Synthetic scores
        hybrid: DPO-Hybrid scores
    
    Returns:
        Dict with t-test results
    """
    results = {}
    
    # Baseline vs DPO-Synthetic
    t1, p1 = stats.ttest_rel(baseline, synthetic)
    results['baseline_vs_synthetic'] = {
        't_statistic': t1,
        'p_value': p1,
        'significant': p1 < 0.05,
        'direction': 'synthetic > baseline' if np.mean(synthetic) > np.mean(baseline) else 'baseline > synthetic'
    }
    
    # Baseline vs DPO-Hybrid
    t2, p2 = stats.ttest_rel(baseline, hybrid)
    results['baseline_vs_hybrid'] = {
        't_statistic': t2,
        'p_value': p2,
        'significant': p2 < 0.05,
        'direction': 'hybrid > baseline' if np.mean(hybrid) > np.mean(baseline) else 'baseline > hybrid'
    }
    
    # DPO-Synthetic vs DPO-Hybrid
    t3, p3 = stats.ttest_rel(synthetic, hybrid)
    results['synthetic_vs_hybrid'] = {
        't_statistic': t3,
        'p_value': p3,
        'significant': p3 < 0.05,
        'direction': 'hybrid > synthetic' if np.mean(hybrid) > np.mean(synthetic) else 'synthetic > hybrid'
    }
    
    # Bonferroni correction for multiple comparisons
    corrected_alpha = 0.05 / 3
    for comparison in results:
        results[comparison]['significant_bonferroni'] = results[comparison]['p_value'] < corrected_alpha
    
    return results

def perform_anova(baseline: List[float], synthetic: List[float], hybrid: List[float]) -> Dict:
    """
    Perform one-way ANOVA for three-group comparison
    
    Args:
        baseline: Baseline model scores
        synthetic: DPO-Synthetic scores
        hybrid: DPO-Hybrid scores
    
    Returns:
        Dict with ANOVA results
    """
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(baseline, synthetic, hybrid)
    
    # Calculate effect size (eta squared)
    df_between = 2  # 3 groups - 1
    df_within = len(baseline) + len(synthetic) + len(hybrid) - 3
    
    eta_sq_result = eta_squared(f_stat, df_between, df_within)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'df_between': df_between,
        'df_within': df_within,
        'eta_squared': eta_sq_result['eta_squared'],
        'eta_squared_interpretation': eta_sq_result['interpretation'],
        'meets_methodology_threshold': eta_sq_result['meets_threshold']
    }

def generate_summary_statistics(baseline: List[float], synthetic: List[float], hybrid: List[float]) -> Dict:
    """
    Generate descriptive statistics for all groups
    
    Args:
        baseline: Baseline model scores
        synthetic: DPO-Synthetic scores
        hybrid: DPO-Hybrid scores
    
    Returns:
        Dict with summary statistics
    """
    def group_stats(scores: List[float], name: str) -> Dict:
        return {
            'name': name,
            'n': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        }
    
    return {
        'baseline': group_stats(baseline, 'Baseline'),
        'dpo_synthetic': group_stats(synthetic, 'DPO-Synthetic'),
        'dpo_hybrid': group_stats(hybrid, 'DPO-Hybrid')
    }

def run_complete_analysis(baseline_file: str, synthetic_file: str, hybrid_file: str) -> Dict:
    """
    Run complete three-way comparison analysis
    
    Args:
        baseline_file: Path to baseline results
        synthetic_file: Path to DPO-Synthetic results
        hybrid_file: Path to DPO-Hybrid results
    
    Returns:
        Complete analysis results
    """
    print("Loading data...")
    baseline_scores = load_results_data(baseline_file)
    synthetic_scores = load_results_data(synthetic_file)
    hybrid_scores = load_results_data(hybrid_file)
    
    if not all([baseline_scores, synthetic_scores, hybrid_scores]):
        print("Error: Could not load all data files")
        return {}
    
    print("Validating data quality...")
    validation = validate_data_quality(baseline_scores, synthetic_scores, hybrid_scores)
    
    print("Calculating summary statistics...")
    summary_stats = generate_summary_statistics(baseline_scores, synthetic_scores, hybrid_scores)
    
    print("Performing paired t-tests...")
    t_test_results = perform_paired_t_tests(baseline_scores, synthetic_scores, hybrid_scores)
    
    print("Performing ANOVA...")
    anova_results = perform_anova(baseline_scores, synthetic_scores, hybrid_scores)
    
    print("Calculating effect sizes...")
    effect_sizes = interpret_effect_sizes(baseline_scores, synthetic_scores, hybrid_scores)
    
    return {
        'data_validation': validation,
        'summary_statistics': summary_stats,
        't_test_results': t_test_results,
        'anova_results': anova_results,
        'effect_sizes': effect_sizes,
        'methodology_validation_summary': {
            'effect_sizes_within_predicted_ranges': all(effect_sizes['methodology_validation'].values()),
            'eta_squared_meets_threshold': anova_results['meets_methodology_threshold'],
            'overall_validation_status': 'PASS' if all(effect_sizes['methodology_validation'].values()) and anova_results['meets_methodology_threshold'] else 'PARTIAL'
        }
    }

def print_analysis_results(results: Dict):
    """Print formatted analysis results"""
    print("\n" + "="*60)
    print("THREE-WAY COMPARISON ANALYSIS RESULTS")
    print("="*60)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    for group_name, stats in results['summary_statistics'].items():
        print(f"\n{stats['name']}:")
        print(f"  N = {stats['n']}, Mean = {stats['mean']:.3f}, SD = {stats['std']:.3f}")
    
    # T-test results
    print("\nPAIRED T-TEST RESULTS:")
    for comparison, data in results['t_test_results'].items():
        sig_marker = "***" if data['significant'] else ""
        print(f"\n{comparison.replace('_', ' ').title()}:")
        print(f"  t = {data['t_statistic']:.3f}, p = {data['p_value']:.4f} {sig_marker}")
        print(f"  Direction: {data['direction']}")
    
    # ANOVA results
    print(f"\nANOVA RESULTS:")
    anova = results['anova_results']
    sig_marker = "***" if anova['significant'] else ""
    print(f"  F({anova['df_between']},{anova['df_within']}) = {anova['f_statistic']:.3f}, p = {anova['p_value']:.4f} {sig_marker}")
    print(f"  η² = {anova['eta_squared']:.3f} ({anova['eta_squared_interpretation']})")
    print(f"  Meets η² > 0.06 threshold: {'✓' if anova['meets_methodology_threshold'] else '✗'}")
    
    # Effect sizes
    print("\nEFFECT SIZES (Cohen's d):")
    for comparison, data in results['effect_sizes'].items():
        if comparison != 'methodology_validation':
            print(f"\n{comparison.replace('_', ' ').title()}:")
            print(f"  d = {data['cohens_d']:.3f} ({data['interpretation']})")
            print(f"  95% CI: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]")
    
    # Methodology validation
    print(f"\nMETHODOLOGY VALIDATION:")
    validation = results['methodology_validation_summary']
    print(f"  Effect sizes within predicted ranges: {'✓' if validation['effect_sizes_within_predicted_ranges'] else '✗'}")
    print(f"  η² meets threshold: {'✓' if validation['eta_squared_meets_threshold'] else '✗'}")
    print(f"  Overall status: {validation['overall_validation_status']}")

def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Three-way comparison analysis')
    parser.add_argument('--baseline', help='Baseline results file', default=None)
    parser.add_argument('--synthetic', help='DPO-Synthetic results file', default=None)
    parser.add_argument('--hybrid', help='DPO-Hybrid results file', default=None)
    parser.add_argument('--example', action='store_true', help='Run with example data')
    
    args = parser.parse_args()
    
    if args.example or not all([args.baseline, args.synthetic, args.hybrid]):
        print("Running with example data...")
        
        # Generate example data
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50).tolist()
        synthetic = np.random.normal(0.65, 0.1, 50).tolist()  # Medium effect
        hybrid = np.random.normal(0.8, 0.1, 50).tolist()      # Large effect
        
        # Create temporary files
        example_data = {
            'results': []
        }
        
        for i, (b, s, h) in enumerate(zip(baseline, synthetic, hybrid)):
            topic_result = {
                'topic_id': f'T{i+1:04d}',
                'emails': [
                    {'evaluation': {'overall_score': b}},
                    {'evaluation': {'overall_score': s}},
                    {'evaluation': {'overall_score': h}}
                ]
            }
            example_data['results'].append(topic_result)
        
        # Save temporary files
        temp_dir = Path('temp_analysis')
        temp_dir.mkdir(exist_ok=True)
        
        for name, scores in [('baseline', baseline), ('synthetic', synthetic), ('hybrid', hybrid)]:
            data_copy = example_data.copy()
            data_copy['results'] = []
            for i, score in enumerate(scores):
                topic_result = {
                    'topic_id': f'T{i+1:04d}',
                    'emails': [{'evaluation': {'overall_score': score}}]
                }
                data_copy['results'].append(topic_result)
            
            with open(temp_dir / f'{name}_results.json', 'w') as f:
                json.dump(data_copy, f)
        
        # Run analysis
        results = run_complete_analysis(
            str(temp_dir / 'baseline_results.json'),
            str(temp_dir / 'synthetic_results.json'),
            str(temp_dir / 'hybrid_results.json')
        )
        
    else:
        # Run with provided files
        results = run_complete_analysis(args.baseline, args.synthetic, args.hybrid)
    
    if results:
        print_analysis_results(results)

if __name__ == "__main__":
    main()