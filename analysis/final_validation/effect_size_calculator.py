#!/usr/bin/env python3
"""
Effect Size Calculator for Final Validation Protocol
Calculates Cohen's d and η² with confidence intervals
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict

def cohens_d(group1: List[float], group2: List[float]) -> Dict:
    """
    Calculate Cohen's d effect size between two groups
    
    Args:
        group1: First group scores
        group2: Second group scores
    
    Returns:
        Dict with effect size, interpretation, and confidence interval
    """
    n1, n2 = len(group1), len(group2)
    
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    
    # Confidence interval (95%)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se
    
    # Interpretation
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'cohens_d': d,
        'interpretation': interpretation,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'n1': n1,
        'n2': n2
    }

def eta_squared(f_statistic: float, df_between: int, df_within: int) -> Dict:
    """
    Calculate η² (eta squared) effect size for ANOVA
    
    Args:
        f_statistic: F-statistic from ANOVA
        df_between: Degrees of freedom between groups
        df_within: Degrees of freedom within groups
    
    Returns:
        Dict with eta squared and interpretation
    """
    eta_sq = (df_between * f_statistic) / (df_between * f_statistic + df_within)
    
    # Interpretation guidelines
    if eta_sq < 0.01:
        interpretation = "negligible"
    elif eta_sq < 0.06:
        interpretation = "small"
    elif eta_sq < 0.14:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        'eta_squared': eta_sq,
        'interpretation': interpretation,
        'meets_threshold': eta_sq > 0.06  # Methodology threshold
    }

def interpret_effect_sizes(baseline_scores: List[float], 
                          dpo_synthetic_scores: List[float], 
                          dpo_hybrid_scores: List[float]) -> Dict:
    """
    Calculate all effect sizes for three-way comparison
    
    Args:
        baseline_scores: Baseline model scores
        dpo_synthetic_scores: DPO-Synthetic model scores
        dpo_hybrid_scores: DPO-Hybrid model scores
    
    Returns:
        Dict with all pairwise effect sizes and validation status
    """
    results = {}
    
    # Pairwise Cohen's d calculations
    results['baseline_vs_synthetic'] = cohens_d(baseline_scores, dpo_synthetic_scores)
    results['baseline_vs_hybrid'] = cohens_d(baseline_scores, dpo_hybrid_scores)
    results['synthetic_vs_hybrid'] = cohens_d(dpo_synthetic_scores, dpo_hybrid_scores)
    
    # Validation against methodology predictions
    validations = {}
    
    # Baseline vs DPO-Synthetic: expected d = 0.5-0.7
    d1 = abs(results['baseline_vs_synthetic']['cohens_d'])
    validations['baseline_vs_synthetic'] = 0.5 <= d1 <= 0.7
    
    # Baseline vs DPO-Hybrid: expected d = 0.7-1.0
    d2 = abs(results['baseline_vs_hybrid']['cohens_d'])
    validations['baseline_vs_hybrid'] = 0.7 <= d2 <= 1.0
    
    # DPO-Synthetic vs DPO-Hybrid: expected d = 0.3-0.5
    d3 = abs(results['synthetic_vs_hybrid']['cohens_d'])
    validations['synthetic_vs_hybrid'] = 0.3 <= d3 <= 0.5
    
    results['methodology_validation'] = validations
    
    return results

def main():
    """Test the effect size calculator with example data"""
    print("Effect Size Calculator - Test Mode")
    print("=" * 50)
    
    # Example data
    baseline = [0.4, 0.5, 0.6, 0.45, 0.55, 0.5, 0.48, 0.52, 0.47, 0.53]
    synthetic = [0.6, 0.65, 0.7, 0.62, 0.68, 0.64, 0.66, 0.69, 0.63, 0.67]
    hybrid = [0.75, 0.8, 0.85, 0.78, 0.82, 0.79, 0.81, 0.84, 0.77, 0.83]
    
    # Calculate effect sizes
    results = interpret_effect_sizes(baseline, synthetic, hybrid)
    
    # Print results
    for comparison, data in results.items():
        if comparison != 'methodology_validation':
            print(f"\n{comparison.replace('_', ' ').title()}:")
            print(f"  Cohen's d: {data['cohens_d']:.3f}")
            print(f"  Interpretation: {data['interpretation']}")
            print(f"  95% CI: [{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]")
    
    print("\nMethodology Validation:")
    for comparison, valid in results['methodology_validation'].items():
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"  {comparison}: {status}")

if __name__ == "__main__":
    main()