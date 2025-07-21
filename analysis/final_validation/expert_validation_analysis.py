#!/usr/bin/env python3
"""
Expert Validation Analysis for Final Validation Protocol
Implements correlation analysis between automated and expert assessment
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json

def calculate_correlation(automated_scores: List[float], expert_scores: List[float]) -> Dict:
    """
    Calculate correlation between automated and expert scores
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
    
    Returns:
        Dict with correlation results and validation
    """
    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(automated_scores, expert_scores)
    
    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = stats.spearmanr(automated_scores, expert_scores)
    
    # Check methodology threshold (r > 0.80)
    meets_threshold_pearson = abs(r_pearson) > 0.80
    meets_threshold_spearman = abs(r_spearman) > 0.80
    
    return {
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'pearson_significant': p_pearson < 0.05,
        'spearman_r': r_spearman,
        'spearman_p': p_spearman,
        'spearman_significant': p_spearman < 0.05,
        'meets_methodology_threshold': meets_threshold_pearson,
        'robust_correlation_adequate': meets_threshold_spearman,
        'sample_size': len(automated_scores)
    }

def calculate_agreement_metrics(automated_scores: List[float], expert_scores: List[float]) -> Dict:
    """
    Calculate additional agreement metrics
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
    
    Returns:
        Dict with agreement metrics
    """
    # Mean absolute error
    mae = np.mean(np.abs(np.array(automated_scores) - np.array(expert_scores)))
    
    # Root mean square error
    rmse = np.sqrt(np.mean((np.array(automated_scores) - np.array(expert_scores))**2))
    
    # Mean bias (systematic over/under-estimation)
    bias = np.mean(np.array(automated_scores) - np.array(expert_scores))
    
    # Intraclass correlation coefficient (ICC) approximation
    # Using two-way random effects model
    automated = np.array(automated_scores)
    expert = np.array(expert_scores)
    
    # Create data matrix for ICC calculation
    data = np.column_stack([automated, expert])
    n_subjects = len(automated)
    
    # Calculate ICC(2,1) - two-way random, single measures
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)
    
    # Mean squares
    ms_between = 2 * np.sum((row_means - grand_mean)**2) / (n_subjects - 1)
    ms_within = np.sum((data - row_means.reshape(-1, 1))**2) / n_subjects
    
    # ICC calculation
    icc = (ms_between - ms_within) / (ms_between + ms_within)
    
    return {
        'mean_absolute_error': mae,
        'root_mean_square_error': rmse,
        'mean_bias': bias,
        'icc_estimate': icc,
        'agreement_interpretation': 'excellent' if abs(icc) > 0.75 else 'good' if abs(icc) > 0.60 else 'moderate'
    }

def bland_altman_analysis(automated_scores: List[float], expert_scores: List[float]) -> Dict:
    """
    Perform Bland-Altman analysis for method comparison
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
    
    Returns:
        Dict with Bland-Altman results
    """
    automated = np.array(automated_scores)
    expert = np.array(expert_scores)
    
    # Calculate differences and means
    differences = automated - expert
    means = (automated + expert) / 2
    
    # Summary statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    # Limits of agreement (95%)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    # Check for proportional bias (correlation between differences and means)
    r_bias, p_bias = stats.pearsonr(means, differences)
    proportional_bias = abs(r_bias) > 0.3 and p_bias < 0.05
    
    return {
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'upper_limit_agreement': upper_loa,
        'lower_limit_agreement': lower_loa,
        'proportional_bias_r': r_bias,
        'proportional_bias_p': p_bias,
        'has_proportional_bias': proportional_bias,
        'within_acceptable_limits': abs(mean_diff) < 0.1 and (upper_loa - lower_loa) < 0.4
    }

def inter_rater_reliability(scores_list: List[List[float]]) -> Dict:
    """
    Calculate inter-rater reliability when multiple experts available
    
    Args:
        scores_list: List of score lists from different raters
    
    Returns:
        Dict with reliability metrics
    """
    if len(scores_list) < 2:
        return {'error': 'Need at least 2 raters for reliability analysis'}
    
    # Convert to numpy array
    scores_array = np.array(scores_list).T  # Transpose for subjects x raters
    
    # Calculate Cronbach's alpha
    n_items = scores_array.shape[1]
    item_variances = np.var(scores_array, axis=0, ddof=1)
    total_variance = np.var(np.sum(scores_array, axis=1), ddof=1)
    
    cronbach_alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    
    # Pairwise correlations
    pairwise_correlations = []
    for i in range(len(scores_list)):
        for j in range(i + 1, len(scores_list)):
            r, _ = stats.pearsonr(scores_list[i], scores_list[j])
            pairwise_correlations.append(r)
    
    mean_pairwise_r = np.mean(pairwise_correlations)
    
    return {
        'cronbach_alpha': cronbach_alpha,
        'mean_pairwise_correlation': mean_pairwise_r,
        'n_raters': len(scores_list),
        'n_subjects': len(scores_list[0]),
        'reliability_interpretation': 'excellent' if cronbach_alpha > 0.9 else 'good' if cronbach_alpha > 0.8 else 'acceptable' if cronbach_alpha > 0.7 else 'questionable'
    }

def generate_expert_validation_plot(automated_scores: List[float], expert_scores: List[float], 
                                   output_file: Optional[str] = None) -> None:
    """
    Generate correlation and Bland-Altman plots
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
        output_file: Optional file path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Correlation plot
    ax1.scatter(automated_scores, expert_scores, alpha=0.6)
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(automated_scores, expert_scores)
    line_x = np.array([min(automated_scores), max(automated_scores)])
    line_y = slope * line_x + intercept
    ax1.plot(line_x, line_y, 'r-', alpha=0.8)
    
    # Perfect correlation line
    min_val = min(min(automated_scores), min(expert_scores))
    max_val = max(max(automated_scores), max(expert_scores))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
    
    ax1.set_xlabel('Automated Scores')
    ax1.set_ylabel('Expert Scores')
    ax1.set_title(f'Expert vs Automated Correlation\nr = {r_value:.3f}, p = {p_value:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bland-Altman plot
    automated = np.array(automated_scores)
    expert = np.array(expert_scores)
    differences = automated - expert
    means = (automated + expert) / 2
    
    ax2.scatter(means, differences, alpha=0.6)
    
    # Add mean difference line
    mean_diff = np.mean(differences)
    ax2.axhline(mean_diff, color='red', linestyle='-', label=f'Mean diff: {mean_diff:.3f}')
    
    # Add limits of agreement
    std_diff = np.std(differences, ddof=1)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    ax2.axhline(upper_loa, color='red', linestyle='--', alpha=0.8, label='95% limits')
    ax2.axhline(lower_loa, color='red', linestyle='--', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Mean of Automated and Expert Scores')
    ax2.set_ylabel('Difference (Automated - Expert)')
    ax2.set_title('Bland-Altman Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def run_expert_validation_analysis(automated_scores: List[float], 
                                  expert_scores: List[float],
                                  multiple_expert_scores: Optional[List[List[float]]] = None) -> Dict:
    """
    Run complete expert validation analysis
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
        multiple_expert_scores: Optional list of expert score lists for reliability
    
    Returns:
        Complete expert validation results
    """
    print("Calculating correlation metrics...")
    correlation_results = calculate_correlation(automated_scores, expert_scores)
    
    print("Calculating agreement metrics...")
    agreement_results = calculate_agreement_metrics(automated_scores, expert_scores)
    
    print("Performing Bland-Altman analysis...")
    bland_altman_results = bland_altman_analysis(automated_scores, expert_scores)
    
    results = {
        'correlation_analysis': correlation_results,
        'agreement_metrics': agreement_results,
        'bland_altman_analysis': bland_altman_results
    }
    
    # Inter-rater reliability if multiple experts
    if multiple_expert_scores:
        print("Calculating inter-rater reliability...")
        reliability_results = inter_rater_reliability(multiple_expert_scores)
        results['inter_rater_reliability'] = reliability_results
    
    # Overall validation summary
    results['validation_summary'] = {
        'meets_correlation_threshold': correlation_results['meets_methodology_threshold'],
        'acceptable_agreement': agreement_results['icc_estimate'] > 0.6,
        'no_systematic_bias': abs(bland_altman_results['mean_difference']) < 0.1,
        'overall_validation_status': 'PASS' if (
            correlation_results['meets_methodology_threshold'] and 
            agreement_results['icc_estimate'] > 0.6 and 
            abs(bland_altman_results['mean_difference']) < 0.1
        ) else 'PARTIAL'
    }
    
    return results

def print_expert_validation_results(results: Dict):
    """Print formatted expert validation results"""
    print("\n" + "="*60)
    print("EXPERT VALIDATION ANALYSIS RESULTS")
    print("="*60)
    
    # Correlation analysis
    corr = results['correlation_analysis']
    print(f"\nCORRELATION ANALYSIS:")
    print(f"  Pearson r = {corr['pearson_r']:.3f}, p = {corr['pearson_p']:.4f}")
    print(f"  Spearman r = {corr['spearman_r']:.3f}, p = {corr['spearman_p']:.4f}")
    print(f"  Meets r > 0.80 threshold: {'✓' if corr['meets_methodology_threshold'] else '✗'}")
    print(f"  Sample size: {corr['sample_size']}")
    
    # Agreement metrics
    agree = results['agreement_metrics']
    print(f"\nAGREEMENT METRICS:")
    print(f"  Mean Absolute Error: {agree['mean_absolute_error']:.4f}")
    print(f"  Root Mean Square Error: {agree['root_mean_square_error']:.4f}")
    print(f"  Mean Bias: {agree['mean_bias']:.4f}")
    print(f"  ICC Estimate: {agree['icc_estimate']:.3f} ({agree['agreement_interpretation']})")
    
    # Bland-Altman analysis
    ba = results['bland_altman_analysis']
    print(f"\nBLAND-ALTMAN ANALYSIS:")
    print(f"  Mean Difference: {ba['mean_difference']:.4f}")
    print(f"  Limits of Agreement: [{ba['lower_limit_agreement']:.3f}, {ba['upper_limit_agreement']:.3f}]")
    print(f"  Proportional Bias: {'Yes' if ba['has_proportional_bias'] else 'No'}")
    print(f"  Within Acceptable Limits: {'✓' if ba['within_acceptable_limits'] else '✗'}")
    
    # Inter-rater reliability (if available)
    if 'inter_rater_reliability' in results:
        irr = results['inter_rater_reliability']
        print(f"\nINTER-RATER RELIABILITY:")
        print(f"  Cronbach's Alpha: {irr['cronbach_alpha']:.3f} ({irr['reliability_interpretation']})")
        print(f"  Mean Pairwise Correlation: {irr['mean_pairwise_correlation']:.3f}")
        print(f"  Number of Raters: {irr['n_raters']}")
    
    # Overall validation
    validation = results['validation_summary']
    print(f"\nVALIDATION SUMMARY:")
    print(f"  Correlation threshold met: {'✓' if validation['meets_correlation_threshold'] else '✗'}")
    print(f"  Acceptable agreement: {'✓' if validation['acceptable_agreement'] else '✗'}")
    print(f"  No systematic bias: {'✓' if validation['no_systematic_bias'] else '✗'}")
    print(f"  Overall status: {validation['overall_validation_status']}")

def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Expert validation analysis')
    parser.add_argument('--automated', help='Automated scores JSON file', default=None)
    parser.add_argument('--expert', help='Expert scores JSON file', default=None)
    parser.add_argument('--example', action='store_true', help='Run with example data')
    parser.add_argument('--plot', help='Output plot file', default=None)
    
    args = parser.parse_args()
    
    if args.example or not all([args.automated, args.expert]):
        print("Running with example data...")
        
        # Generate correlated example data
        np.random.seed(42)
        true_scores = np.random.uniform(0.3, 0.9, 50)
        
        # Automated scores with some noise
        automated = true_scores + np.random.normal(0, 0.05, 50)
        automated = np.clip(automated, 0, 1)
        
        # Expert scores with different noise pattern (high correlation)
        expert = true_scores + np.random.normal(0, 0.03, 50)
        expert = np.clip(expert, 0, 1)
        
        # Run analysis
        results = run_expert_validation_analysis(automated.tolist(), expert.tolist())
        print_expert_validation_results(results)
        
        # Generate plot
        generate_expert_validation_plot(automated.tolist(), expert.tolist(), args.plot)
        
    else:
        # Load real data (implement as needed)
        print("Loading real data files...")
        # Implementation would load from JSON files
        pass

if __name__ == "__main__":
    main()