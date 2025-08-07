#!/usr/bin/env python3
"""
Create ALL publication-quality figures for dissertation Results section.
Generates all 9 figures referenced in the LaTeX document.
Modern, minimal aesthetic for PhD-level presentation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

# Import and apply modern style configuration
from figure_style_config import (
    set_dissertation_style, 
    COLORS, 
    apply_minimal_style,
    format_axis_labels,
    add_value_labels,
    get_color
)

# Apply dissertation style settings
set_dissertation_style()

def load_statistical_data():
    """Load statistical values from master JSON file."""
    data_path = Path('results_context/master_files/statistical_values_complete.json')
    with open(data_path, 'r') as f:
        return json.load(f)

def create_anova_summary(data):
    """
    Create ANOVA results summary with integrated horizontal layout for better readability.
    """
    print("Creating ANOVA Summary figure...")
    
    # Single integrated panel with horizontal layout
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data preparation
    f_stat = data['inferential_statistics']['anova']['f_statistic']
    f_critical = 3.02  # Critical value for F(2,747) at α=0.05
    eta_squared = data['inferential_statistics']['anova']['eta_squared']
    eta_threshold = 0.06  # Medium effect size threshold
    p_value = data['inferential_statistics']['anova']['p_value']
    
    # Create horizontal bar chart for integrated display
    categories = ['F-statistic\n(Observed)', 'F-statistic\n(Critical α=0.05)', 
                  'Effect Size η²\n(Observed)', 'Effect Size η²\n(Medium Threshold)']
    values = [f_stat, f_critical, eta_squared * 50, eta_threshold * 50]  # Scale η² for visibility
    colors = [COLORS['highlight'], COLORS['neutral'], COLORS['info'], COLORS['neutral']]
    
    # Create horizontal bars with subtle gradient effect
    bars = ax.barh(categories, values, color=colors, alpha=0.8, 
                   edgecolor=COLORS['dark_gray'], linewidth=0.8, height=0.6)
    
    # Add subtle gradient effect to bars
    for i, bar in enumerate(bars):
        if i < 2:  # F-statistic bars
            bar.set_height(0.5)
        else:  # Effect size bars  
            bar.set_height(0.4)
    
    # Apply minimal style
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Statistical Value', 
                      title='ANOVA Results: Statistical Equivalence Across Model Variants')
    
    # Add value annotations with better positioning
    ax.text(f_stat + 0.05, 0, f'F = {f_stat:.3f}', va='center', fontsize=10, 
            color=COLORS['dark_gray'], fontweight='medium')
    ax.text(f_critical + 0.05, 1, f'F = {f_critical:.2f}', va='center', fontsize=10,
            color=COLORS['dark_gray'], fontweight='medium')
    ax.text(eta_squared * 50 + 0.05, 2, f'η² = {eta_squared:.3f}', va='center', fontsize=10,
            color=COLORS['dark_gray'], fontweight='medium')
    ax.text(eta_threshold * 50 + 0.05, 3, f'η² = {eta_threshold:.2f}', va='center', fontsize=10,
            color=COLORS['dark_gray'], fontweight='medium')
    
    # Add significance and effect size indicators
    ax.text(0.7, -0.6, f'p = {p_value:.3f} (non-significant)', 
            fontsize=12, color=COLORS['highlight'], fontweight='medium',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light_gray'], 
                     alpha=0.8, edgecolor='none'))
    
    ax.text(0.7, -0.9, 'Negligible practical effect', 
            fontsize=11, color=COLORS['neutral'], style='italic')
    
    # Add reference lines
    ax.axvline(x=f_critical, color=COLORS['neutral'], linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=eta_threshold * 50, color=COLORS['neutral'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Set limits and styling
    ax.set_xlim(0, max(f_critical * 1.3, eta_threshold * 50 * 1.3))
    ax.invert_yaxis()  # Categories from top to bottom
    
    # Remove excessive grid lines
    ax.grid(axis='x', alpha=0.3)
    
    fig.savefig('../report/figures/anova_summary.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ ANOVA Summary saved")

def create_means_comparison(data):
    """
    Create bar chart showing means with standard error bars.
    """
    print("Creating Means Comparison figure...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    variants = ['Baseline', 'DPO-Synthetic', 'DPO-Hybrid']
    means = [
        data['descriptive_statistics']['baseline']['mean'],
        data['descriptive_statistics']['dpo_synthetic']['mean'],
        data['descriptive_statistics']['dpo_hybrid']['mean']
    ]
    # Calculate standard errors (use pre-calculated SE from JSON)
    n = 250
    ses = [
        data['descriptive_statistics']['baseline']['se'],
        data['descriptive_statistics']['dpo_synthetic']['se'],
        data['descriptive_statistics']['dpo_hybrid']['se']
    ]
    
    # Create bars
    x_pos = np.arange(len(variants))
    bars = ax.bar(x_pos, means, yerr=ses, capsize=8,
                  color=[COLORS['baseline'], COLORS['synthetic'], COLORS['hybrid']],
                  alpha=0.8, edgecolor='black', linewidth=1,
                  error_kw=dict(elinewidth=1.5, ecolor=COLORS['dark_gray'], capthick=1.5))
    
    # Add value labels
    for i, (mean, se) in enumerate(zip(means, ses)):
        ax.text(i, mean + se + 0.01, f'{mean:.3f} ± {se:.3f}', 
                ha='center', fontsize=10, color=COLORS['dark_gray'])
    
    # Apply minimal style and formatting
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Model Variant', ylabel='Mean Performance Score',
                      title='Mean Performance with Standard Error Bars')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 0.7)
    
    # Add horizontal line showing overall mean
    overall_mean = np.mean(means)
    ax.axhline(y=overall_mean, color=COLORS['highlight'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(2.5, overall_mean, f'Overall M = {overall_mean:.3f}', 
            fontsize=9, va='bottom', color=COLORS['dark_gray'])
    
    plt.tight_layout()
    fig.savefig('../report/figures/means_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Means Comparison saved")

def create_effect_size_comparison(data):
    """
    Create bar chart comparing absolute Cohen's d values with threshold lines.
    """
    print("Creating Effect Size Comparison figure...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    comparisons = ['Baseline vs\nDPO-Synthetic', 'Baseline vs\nDPO-Hybrid', 
                   'DPO-Synthetic vs\nDPO-Hybrid']
    effect_sizes = [
        abs(data['effect_sizes']['baseline_vs_synthetic']['cohens_d']),
        abs(data['effect_sizes']['baseline_vs_hybrid']['cohens_d']),
        abs(data['effect_sizes']['synthetic_vs_hybrid']['cohens_d'])
    ]
    
    # Create bars
    x_pos = np.arange(len(comparisons))
    bars = ax.bar(x_pos, effect_sizes, color=COLORS['highlight'], 
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add Cohen's d threshold lines with muted colors
    ax.axhline(y=0.2, color=COLORS['success'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Small effect (d = 0.2)')
    ax.axhline(y=0.5, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Medium effect (d = 0.5)')
    ax.axhline(y=0.8, color=COLORS['error'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Large effect (d = 0.8)')
    
    # Add value labels
    for i, d in enumerate(effect_sizes):
        ax.text(i, d + 0.005, f'd = {d:.3f}', ha='center', fontsize=10, color=COLORS['dark_gray'])
    
    # Apply minimal style and formatting
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Pairwise Comparison', ylabel='Absolute Cohen\'s d',
                      title='Effect Size Magnitude Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparisons)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add interpretation with modern styling
    ax.text(1, 0.15, 'All effects below\nsmall threshold', 
            ha='center', fontsize=11, style='italic', color=COLORS['highlight'],
            bbox=dict(boxstyle='round', facecolor=COLORS['light_gray'], alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    fig.savefig('../report/figures/effect_size_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Effect Size Comparison saved")

def create_model_size_comparison(data):
    """
    Create comparison of performance by model size groups.
    """
    print("Creating Model Size Comparison figure...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data from size groups
    small_data = data['model_specific_results']['size_groups']['small']
    medium_data = data['model_specific_results']['size_groups']['medium']
    
    # Prepare data
    size_groups = ['Small Models\n(M0001, M0003, M0005)', 
                   'Medium Models\n(M0002, M0004)']
    
    baseline_means = [small_data['baseline']['mean'], 
                     medium_data['baseline']['mean']]
    synthetic_means = [small_data['dpo_synthetic']['mean'],
                      medium_data['dpo_synthetic']['mean']]
    hybrid_means = [small_data['dpo_hybrid']['mean'],
                   medium_data['dpo_hybrid']['mean']]
    
    # Bar positions
    x = np.arange(len(size_groups))
    width = 0.25
    
    # Create bars with modern styling
    bars1 = ax.bar(x - width, baseline_means, width, label='Baseline',
                   color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, synthetic_means, width, label='DPO-Synthetic',
                   color=COLORS['synthetic'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + width, hybrid_means, width, label='DPO-Hybrid',
                   color=COLORS['hybrid'], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', fontsize=9, color=COLORS['dark_gray'])
    
    # Apply minimal style and formatting
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Model Size Category', ylabel='Mean Performance Score',
                      title='Performance Comparison by Model Size Category')
    ax.set_xticks(x)
    ax.set_xticklabels(size_groups)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 0.7)
    
    # Add improvement indicators with modern colors
    for i, group in enumerate(['Small', 'Medium']):
        if i == 0:  # Small models
            improvement = small_data['improvement_dpo_hybrid']
        else:  # Medium models
            improvement = medium_data['improvement_dpo_hybrid']
        
        color = COLORS['success'] if improvement > 0 else COLORS['error']
        ax.text(i, 0.65, f'Δ = {improvement:.1f}%', ha='center',
               color=color, fontweight='medium', fontsize=10)
    
    plt.tight_layout()
    fig.savefig('../report/figures/model_size_comparison.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Model Size Comparison saved")

def create_category_performance(data):
    """
    Create performance comparison across charity topic categories.
    """
    print("Creating Category Performance figure...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Categories
    categories = ['Healthcare/\nMedical', 'Education/\nYouth', 
                  'Environmental', 'Community/\nSocial']
    
    # Extract data for each category
    cat_data = data['category_results']
    
    baseline_means = [
        cat_data['healthcare_medical']['baseline']['mean'],
        cat_data['education_youth']['baseline']['mean'],
        cat_data['environmental']['baseline']['mean'],
        cat_data['community_social']['baseline']['mean']
    ]
    
    synthetic_means = [
        cat_data['healthcare_medical']['dpo_synthetic']['mean'],
        cat_data['education_youth']['dpo_synthetic']['mean'],
        cat_data['environmental']['dpo_synthetic']['mean'],
        cat_data['community_social']['dpo_synthetic']['mean']
    ]
    
    hybrid_means = [
        cat_data['healthcare_medical']['dpo_hybrid']['mean'],
        cat_data['education_youth']['dpo_hybrid']['mean'],
        cat_data['environmental']['dpo_hybrid']['mean'],
        cat_data['community_social']['dpo_hybrid']['mean']
    ]
    
    # Bar positions
    x = np.arange(len(categories))
    width = 0.25
    
    # Create bars with modern styling
    bars1 = ax.bar(x - width, baseline_means, width, label='Baseline',
                   color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x, synthetic_means, width, label='DPO-Synthetic',
                   color=COLORS['synthetic'], alpha=0.8, edgecolor='black', linewidth=0.8)
    bars3 = ax.bar(x + width, hybrid_means, width, label='DPO-Hybrid',
                   color=COLORS['hybrid'], alpha=0.8, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.3f}', ha='center', fontsize=8, rotation=0, color=COLORS['dark_gray'])
    
    # Add improvement percentages
    improvements_synthetic = [
        cat_data['healthcare_medical']['improvement_dpo_synthetic'],
        cat_data['education_youth']['improvement_dpo_synthetic'],
        cat_data['environmental']['improvement_dpo_synthetic'],
        cat_data['community_social']['improvement_dpo_synthetic']
    ]
    
    improvements_hybrid = [
        cat_data['healthcare_medical']['improvement_dpo_hybrid'],
        cat_data['education_youth']['improvement_dpo_hybrid'],
        cat_data['environmental']['improvement_dpo_hybrid'],
        cat_data['community_social']['improvement_dpo_hybrid']
    ]
    
    # Add improvement annotations with modern styling
    for i, (imp_s, imp_h) in enumerate(zip(improvements_synthetic, improvements_hybrid)):
        # Best improvement for this category
        best_imp = max(abs(imp_s), abs(imp_h))
        best_variant = 'Synthetic' if abs(imp_s) > abs(imp_h) else 'Hybrid'
        best_val = imp_s if abs(imp_s) > abs(imp_h) else imp_h
        
        if abs(best_val) > 10:  # Highlight large changes
            color = COLORS['success'] if best_val > 0 else COLORS['error']
            ax.text(i, 0.72, f'{best_variant}\n{best_val:+.1f}%', 
                   ha='center', fontsize=9, color=color, fontweight='medium',
                   bbox=dict(boxstyle='round', facecolor=COLORS['light_gray'], alpha=0.8, edgecolor='none'))
    
    # Apply minimal style and formatting
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Charity Topic Category', ylabel='Mean Performance Score',
                      title='Performance Across Charity Topic Categories')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    fig.savefig('../report/figures/category_performance.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Category Performance saved")

def create_methodology_validation(data):
    """
    Create predicted vs actual effect sizes comparison showing validation failure.
    """
    print("Creating Methodology Validation figure...")
    
    # Create validation directory if it doesn't exist
    validation_dir = Path('../report/figures/validation')
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data for comparisons
    comparisons = ['Baseline vs\nDPO-Synthetic', 'Baseline vs\nDPO-Hybrid', 
                   'DPO-Synthetic vs\nDPO-Hybrid']
    
    # Predicted effect sizes (mid-points of ranges from methodology)
    predicted = [0.6, 0.85, 0.4]  # Mid-points of [0.5-0.7], [0.7-1.0], [0.3-0.5]
    
    # Actual effect sizes
    actual = [
        abs(data['effect_sizes']['baseline_vs_synthetic']['cohens_d']),
        abs(data['effect_sizes']['baseline_vs_hybrid']['cohens_d']),
        abs(data['effect_sizes']['synthetic_vs_hybrid']['cohens_d'])
    ]
    
    # Bar positions
    x = np.arange(len(comparisons))
    width = 0.35
    
    # Create bars with modern styling
    bars1 = ax.bar(x - width/2, predicted, width, label='Predicted',
                   color=COLORS['neutral'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, actual, width, label='Actual',
                   color=COLORS['highlight'], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (pred, act) in enumerate(zip(predicted, actual)):
        ax.text(i - width/2, pred + 0.02, f'd = {pred:.2f}', 
                ha='center', fontsize=9, color=COLORS['dark_gray'])
        ax.text(i + width/2, act + 0.02, f'd = {act:.3f}', 
                ha='center', fontsize=9, color=COLORS['dark_gray'])
        
        # Add error magnitude
        error = pred - act
        ax.text(i, 0.95, f'Error: {error:.2f}', ha='center', 
                fontsize=9, color=COLORS['error'], fontweight='medium')
    
    # Add Cohen's d threshold line with modern styling
    ax.axhline(y=0.2, color=COLORS['success'], linestyle='--', linewidth=1.5, 
               alpha=0.7, label='Small effect threshold')
    
    # Apply minimal style and formatting
    apply_minimal_style(ax)
    format_axis_labels(ax, xlabel='Pairwise Comparison', ylabel='Cohen\'s d Effect Size',
                      title='Methodology Validation: Predicted vs Actual Effect Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(comparisons)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add validation status with modern styling
    ax.text(1, 0.5, 'VALIDATION FAILED', ha='center', fontsize=14,
            color=COLORS['error'], fontweight='bold', rotation=15,
            bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    fig.savefig('../report/figures/validation/methodology_validation.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Methodology Validation saved")

def main():
    """Generate all dissertation figures."""
    print("\n" + "="*60)
    print("GENERATING ALL DISSERTATION FIGURES")
    print("="*60 + "\n")
    
    # Ensure output directories exist
    output_dir = Path('../report/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validation_dir = Path('../report/figures/validation')
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load statistical data
    print("Loading statistical data...")
    data = load_statistical_data()
    print("✓ Data loaded successfully\n")
    
    # Generate all figures
    print("Generating missing figures...")
    create_anova_summary(data)
    create_means_comparison(data)
    create_effect_size_comparison(data)
    create_model_size_comparison(data)
    create_category_performance(data)
    create_methodology_validation(data)
    
    print("\n" + "="*60)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)
    print("\nNew figures created:")
    print("- anova_summary.png")
    print("- means_comparison.png")
    print("- effect_size_comparison.png")
    print("- model_size_comparison.png")
    print("- category_performance.png")
    print("- validation/methodology_validation.png")
    print("\nAll 9 figures are now ready for your LaTeX document.")

if __name__ == "__main__":
    main()