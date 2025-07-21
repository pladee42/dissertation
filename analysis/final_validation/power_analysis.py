#!/usr/bin/env python3
"""
Power Analysis Framework for Final Validation Protocol
Implements post-hoc power analysis and sample size calculations
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

def calculate_post_hoc_power(effect_size: float, sample_size: int, alpha: float = 0.05, 
                            test_type: str = 'two_sample') -> float:
    """
    Calculate post-hoc statistical power
    
    Args:
        effect_size: Cohen's d or other effect size measure
        sample_size: Sample size (per group for two-sample tests)
        alpha: Type I error rate
        test_type: Type of test ('two_sample', 'paired', 'anova')
    
    Returns:
        Statistical power (1 - β)
    """
    if test_type == 'two_sample':
        # Two independent samples t-test
        df = 2 * sample_size - 2
        ncp = effect_size * np.sqrt(sample_size / 2)  # Non-centrality parameter
        t_critical = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
    elif test_type == 'paired':
        # Paired t-test
        df = sample_size - 1
        ncp = effect_size * np.sqrt(sample_size)
        t_critical = stats.t.ppf(1 - alpha/2, df)
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        
    elif test_type == 'anova':
        # One-way ANOVA (simplified for equal groups)
        # Convert eta-squared to f
        f = np.sqrt(effect_size / (1 - effect_size))
        df_between = 2  # 3 groups - 1
        df_within = 3 * sample_size - 3
        ncp = f * np.sqrt(3 * sample_size)
        f_critical = stats.f.ppf(1 - alpha, df_between, df_within)
        power = 1 - stats.ncf.cdf(f_critical, df_between, df_within, ncp**2)
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return power

def sample_size_for_power(effect_size: float, power: float = 0.80, alpha: float = 0.05,
                         test_type: str = 'two_sample') -> int:
    """
    Calculate required sample size for desired power
    
    Args:
        effect_size: Expected effect size
        power: Desired statistical power
        alpha: Type I error rate
        test_type: Type of test
    
    Returns:
        Required sample size per group
    """
    # Use iterative search to find sample size
    for n in range(5, 1000):
        calculated_power = calculate_post_hoc_power(effect_size, n, alpha, test_type)
        if calculated_power >= power:
            return n
    
    return 1000  # If not found within reasonable range

def effect_size_detection_analysis(sample_size: int, power: float = 0.80, alpha: float = 0.05,
                                  test_type: str = 'two_sample') -> Dict:
    """
    Determine minimum detectable effect size
    
    Args:
        sample_size: Available sample size
        power: Desired power
        alpha: Type I error rate
        test_type: Type of test
    
    Returns:
        Dict with minimum detectable effect sizes
    """
    # Search for minimum detectable effect size
    effect_sizes = np.arange(0.1, 3.0, 0.01)
    
    for es in effect_sizes:
        calculated_power = calculate_post_hoc_power(es, sample_size, alpha, test_type)
        if calculated_power >= power:
            min_detectable_es = es
            break
    else:
        min_detectable_es = 3.0  # Maximum searched
    
    # Categorize by Cohen's conventions
    small_effect_power = calculate_post_hoc_power(0.2, sample_size, alpha, test_type)
    medium_effect_power = calculate_post_hoc_power(0.5, sample_size, alpha, test_type)
    large_effect_power = calculate_post_hoc_power(0.8, sample_size, alpha, test_type)
    
    return {
        'min_detectable_effect_size': min_detectable_es,
        'power_for_small_effect': small_effect_power,
        'power_for_medium_effect': medium_effect_power,
        'power_for_large_effect': large_effect_power,
        'adequate_for_small_effects': small_effect_power >= 0.80,
        'adequate_for_medium_effects': medium_effect_power >= 0.80,
        'adequate_for_large_effects': large_effect_power >= 0.80
    }

def sensitivity_analysis(baseline_sample_size: int, effect_size_range: Tuple[float, float],
                        test_type: str = 'paired') -> Dict:
    """
    Perform sensitivity analysis across effect size and sample size ranges
    
    Args:
        baseline_sample_size: Current sample size
        effect_size_range: Range of effect sizes to test
        test_type: Type of statistical test
    
    Returns:
        Dict with sensitivity analysis results
    """
    effect_sizes = np.linspace(effect_size_range[0], effect_size_range[1], 20)
    sample_sizes = np.arange(10, baseline_sample_size + 50, 5)
    
    # Power across effect sizes (fixed sample size)
    power_by_effect = []
    for es in effect_sizes:
        power = calculate_post_hoc_power(es, baseline_sample_size, test_type=test_type)
        power_by_effect.append(power)
    
    # Power across sample sizes (medium effect size)
    medium_effect = 0.5
    power_by_sample = []
    for n in sample_sizes:
        power = calculate_post_hoc_power(medium_effect, n, test_type=test_type)
        power_by_sample.append(power)
    
    return {
        'effect_sizes': effect_sizes.tolist(),
        'power_by_effect_size': power_by_effect,
        'sample_sizes': sample_sizes.tolist(),
        'power_by_sample_size': power_by_sample,
        'baseline_sample_size': baseline_sample_size,
        'effect_size_range': effect_size_range
    }

def validate_methodology_power_requirements(current_results: Dict) -> Dict:
    """
    Validate if current sample sizes meet methodology requirements
    
    Args:
        current_results: Results from three-way comparison analysis
    
    Returns:
        Dict with power validation results
    """
    # Extract sample size (assuming equal across groups)
    sample_size = current_results.get('data_validation', {}).get('min_sample_size', 50)
    
    # Expected effect sizes from methodology
    expected_effects = {
        'baseline_vs_synthetic': (0.5, 0.7),    # Medium effect
        'baseline_vs_hybrid': (0.7, 1.0),       # Large effect  
        'synthetic_vs_hybrid': (0.3, 0.5)       # Small-medium effect
    }
    
    power_validation = {}
    
    for comparison, (min_es, max_es) in expected_effects.items():
        # Calculate power for expected range
        min_power = calculate_post_hoc_power(min_es, sample_size, test_type='paired')
        max_power = calculate_post_hoc_power(max_es, sample_size, test_type='paired')
        
        # Required sample sizes for 80% power
        min_n_required = sample_size_for_power(min_es, power=0.80, test_type='paired')
        max_n_required = sample_size_for_power(max_es, power=0.80, test_type='paired')
        
        power_validation[comparison] = {
            'expected_effect_range': (min_es, max_es),
            'power_range': (min_power, max_power),
            'adequate_power': min_power >= 0.80,
            'current_sample_size': sample_size,
            'required_sample_size_range': (min_n_required, max_n_required),
            'sample_size_adequate': sample_size >= max_n_required
        }
    
    # ANOVA power for eta-squared > 0.06
    anova_power = calculate_post_hoc_power(0.06, sample_size, test_type='anova')
    anova_n_required = sample_size_for_power(0.06, power=0.80, test_type='anova')
    
    power_validation['anova_three_way'] = {
        'expected_eta_squared': 0.06,
        'current_power': anova_power,
        'adequate_power': anova_power >= 0.80,
        'current_sample_size': sample_size,
        'required_sample_size': anova_n_required,
        'sample_size_adequate': sample_size >= anova_n_required
    }
    
    # Overall assessment
    all_adequate = all(
        comparison_data.get('adequate_power', False) 
        for comparison_data in power_validation.values()
    )
    
    power_validation['overall_assessment'] = {
        'all_comparisons_adequate_power': all_adequate,
        'current_sample_size': sample_size,
        'methodology_requirements_met': all_adequate,
        'recommendations': generate_power_recommendations(power_validation)
    }
    
    return power_validation

def generate_power_recommendations(power_validation: Dict) -> List[str]:
    """Generate recommendations based on power analysis"""
    recommendations = []
    
    inadequate_comparisons = [
        comp for comp, data in power_validation.items() 
        if isinstance(data, dict) and not data.get('adequate_power', True)
    ]
    
    if inadequate_comparisons:
        recommendations.append(f"Increase sample size for adequate power in: {', '.join(inadequate_comparisons)}")
    
    overall = power_validation.get('overall_assessment', {})
    if not overall.get('methodology_requirements_met', True):
        max_required = max([
            data.get('required_sample_size', 0) if isinstance(data.get('required_sample_size'), int) 
            else max(data.get('required_sample_size_range', [0])) if isinstance(data.get('required_sample_size_range'), (list, tuple))
            else 0
            for data in power_validation.values() if isinstance(data, dict)
        ])
        recommendations.append(f"Consider increasing sample size to at least {max_required} per group")
    
    if not recommendations:
        recommendations.append("Current sample size appears adequate for detecting expected effect sizes")
    
    return recommendations

def generate_power_curves(sample_size: int, output_file: str = None):
    """Generate power curves visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Power vs Effect Size
    effect_sizes = np.linspace(0.1, 1.5, 50)
    powers = [calculate_post_hoc_power(es, sample_size, test_type='paired') for es in effect_sizes]
    
    ax1.plot(effect_sizes, powers, 'b-', linewidth=2)
    ax1.axhline(y=0.80, color='r', linestyle='--', alpha=0.7, label='80% Power')
    ax1.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Small Effect')
    ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='Medium Effect')
    ax1.axvline(x=0.8, color='gray', linestyle=':', alpha=0.9, label='Large Effect')
    
    ax1.set_xlabel('Effect Size (Cohen\'s d)')
    ax1.set_ylabel('Statistical Power')
    ax1.set_title(f'Power vs Effect Size (n={sample_size})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Power vs Sample Size
    sample_sizes = np.arange(10, 100, 2)
    medium_effect_powers = [calculate_post_hoc_power(0.5, n, test_type='paired') for n in sample_sizes]
    small_effect_powers = [calculate_post_hoc_power(0.2, n, test_type='paired') for n in sample_sizes]
    
    ax2.plot(sample_sizes, medium_effect_powers, 'b-', linewidth=2, label='Medium Effect (d=0.5)')
    ax2.plot(sample_sizes, small_effect_powers, 'g-', linewidth=2, label='Small Effect (d=0.2)')
    ax2.axhline(y=0.80, color='r', linestyle='--', alpha=0.7, label='80% Power')
    ax2.axvline(x=sample_size, color='orange', linestyle='-', alpha=0.8, label=f'Current n={sample_size}')
    
    ax2.set_xlabel('Sample Size per Group')
    ax2.set_ylabel('Statistical Power')
    ax2.set_title('Power vs Sample Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Power curves saved to {output_file}")
    else:
        plt.show()

def print_power_analysis_results(results: Dict):
    """Print formatted power analysis results"""
    print("\n" + "="*60)
    print("POWER ANALYSIS RESULTS")
    print("="*60)
    
    for comparison, data in results.items():
        if comparison == 'overall_assessment':
            continue
            
        print(f"\n{comparison.replace('_', ' ').upper()}:")
        if 'expected_effect_range' in data:
            print(f"  Expected Effect Range: {data['expected_effect_range']}")
            print(f"  Power Range: {data['power_range'][0]:.3f} - {data['power_range'][1]:.3f}")
            print(f"  Adequate Power: {'✓' if data['adequate_power'] else '✗'}")
            print(f"  Required Sample Size: {data['required_sample_size_range']}")
        else:
            print(f"  Expected η²: {data.get('expected_eta_squared', 'N/A')}")
            print(f"  Current Power: {data.get('current_power', 0):.3f}")
            print(f"  Adequate Power: {'✓' if data.get('adequate_power', False) else '✗'}")
    
    # Overall assessment
    overall = results.get('overall_assessment', {})
    print(f"\nOVERALL ASSESSMENT:")
    print(f"  Current Sample Size: {overall.get('current_sample_size', 'N/A')}")
    print(f"  Methodology Requirements Met: {'✓' if overall.get('methodology_requirements_met', False) else '✗'}")
    
    print(f"\nRECOMMENDATIONS:")
    for rec in overall.get('recommendations', []):
        print(f"  • {rec}")

def main():
    """Main function with example usage"""
    print("Power Analysis Framework - Example Mode")
    print("=" * 50)
    
    # Example: Current study with n=50
    current_sample_size = 50
    
    # Validate methodology power requirements
    example_results = {
        'data_validation': {
            'min_sample_size': current_sample_size
        }
    }
    
    print(f"Analyzing power for sample size n={current_sample_size}...")
    power_results = validate_methodology_power_requirements(example_results)
    print_power_analysis_results(power_results)
    
    # Generate power curves
    print("\nGenerating power curves...")
    generate_power_curves(current_sample_size)

if __name__ == "__main__":
    main()