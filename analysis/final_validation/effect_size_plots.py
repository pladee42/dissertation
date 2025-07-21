#!/usr/bin/env python3
"""
Effect Size Visualization for Final Validation Protocol
Creates forest plots and effect size visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional

def create_forest_plot(effect_sizes: Dict, output_file: Optional[str] = None):
    """
    Create forest plot for effect sizes with confidence intervals
    
    Args:
        effect_sizes: Effect size results from analysis
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data for plotting
    comparisons = []
    d_values = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    comparison_names = {
        'baseline_vs_synthetic': 'Baseline vs DPO-Synthetic',
        'baseline_vs_hybrid': 'Baseline vs DPO-Hybrid', 
        'synthetic_vs_hybrid': 'DPO-Synthetic vs DPO-Hybrid'
    }
    
    for key, name in comparison_names.items():
        if key in effect_sizes:
            data = effect_sizes[key]
            comparisons.append(name)
            d_values.append(data['cohens_d'])
            ci_lowers.append(data['ci_lower'])
            ci_uppers.append(data['ci_upper'])
            
            # Color by effect size
            d_abs = abs(data['cohens_d'])
            if d_abs >= 0.8:
                colors.append('red')  # Large effect
            elif d_abs >= 0.5:
                colors.append('orange')  # Medium effect
            elif d_abs >= 0.2:
                colors.append('blue')  # Small effect
            else:
                colors.append('gray')  # Negligible
    
    # Create horizontal positions
    y_pos = np.arange(len(comparisons))
    
    # Plot confidence intervals
    for i, (d, lower, upper, color) in enumerate(zip(d_values, ci_lowers, ci_uppers, colors)):
        ax.plot([lower, upper], [i, i], color=color, linewidth=2, alpha=0.7)
        ax.plot([lower, lower], [i-0.1, i+0.1], color=color, linewidth=2)
        ax.plot([upper, upper], [i-0.1, i+0.1], color=color, linewidth=2)
        ax.scatter(d, i, color=color, s=100, zorder=5)
    
    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect')
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.8, color='red', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparisons)
    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title("Effect Sizes with 95% Confidence Intervals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add effect size values as text
    for i, d in enumerate(d_values):
        ax.text(d + 0.05, i, f'd = {d:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Forest plot saved to {output_file}")
    else:
        plt.show()

def create_effect_size_comparison(effect_sizes: Dict, output_file: Optional[str] = None):
    """
    Create bar plot comparing effect sizes
    
    Args:
        effect_sizes: Effect size results from analysis
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    comparison_names = {
        'baseline_vs_synthetic': 'Baseline vs\nDPO-Synthetic',
        'baseline_vs_hybrid': 'Baseline vs\nDPO-Hybrid',
        'synthetic_vs_hybrid': 'DPO-Synthetic vs\nDPO-Hybrid'
    }
    
    comparisons = []
    d_values = []
    colors = []
    
    for key, name in comparison_names.items():
        if key in effect_sizes:
            data = effect_sizes[key]
            comparisons.append(name)
            d_abs = abs(data['cohens_d'])
            d_values.append(d_abs)
            
            # Color by effect size magnitude
            if d_abs >= 0.8:
                colors.append('#d62728')  # Red for large
            elif d_abs >= 0.5:
                colors.append('#ff7f0e')  # Orange for medium
            elif d_abs >= 0.2:
                colors.append('#1f77b4')  # Blue for small
            else:
                colors.append('#7f7f7f')  # Gray for negligible
    
    bars = ax.bar(comparisons, d_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add effect size threshold lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect (0.2)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect (0.5)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large effect (0.8)')
    
    # Add value labels on bars
    for bar, value in zip(bars, d_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel("Effect Size (|Cohen's d|)")
    ax.set_title("Effect Size Magnitudes by Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(d_values) * 1.2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Effect size comparison saved to {output_file}")
    else:
        plt.show()

def main():
    """Example usage of effect size plots"""
    # Example effect sizes data
    example_effect_sizes = {
        'baseline_vs_synthetic': {
            'cohens_d': 0.612,
            'ci_lower': 0.312,
            'ci_upper': 0.912,
            'interpretation': 'medium'
        },
        'baseline_vs_hybrid': {
            'cohens_d': 1.421,
            'ci_lower': 1.098,
            'ci_upper': 1.744,
            'interpretation': 'large'
        },
        'synthetic_vs_hybrid': {
            'cohens_d': 0.809,
            'ci_lower': 0.502,
            'ci_upper': 1.116,
            'interpretation': 'large'
        }
    }
    
    print("Creating effect size visualizations...")
    
    # Create forest plot
    create_forest_plot(example_effect_sizes, "effect_size_forest_plot.png")
    
    # Create comparison plot
    create_effect_size_comparison(example_effect_sizes, "effect_size_comparison.png")

if __name__ == "__main__":
    main()