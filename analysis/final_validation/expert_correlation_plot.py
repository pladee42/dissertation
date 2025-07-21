#!/usr/bin/env python3
"""
Expert Correlation Visualization for Final Validation Protocol
Creates scatter plots and correlation analysis visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import List, Optional

def create_correlation_scatter_plot(automated_scores: List[float], 
                                   expert_scores: List[float],
                                   output_file: Optional[str] = None):
    """
    Create scatter plot showing correlation between automated and expert scores
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate correlation
    r_pearson, p_pearson = stats.pearsonr(automated_scores, expert_scores)
    r_spearman, p_spearman = stats.spearmanr(automated_scores, expert_scores)
    
    # Create scatter plot
    ax.scatter(automated_scores, expert_scores, alpha=0.6, s=50, color='steelblue')
    
    # Add regression line
    slope, intercept, _, _, _ = stats.linregress(automated_scores, expert_scores)
    line_x = np.array([min(automated_scores), max(automated_scores)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', alpha=0.8, linewidth=2, label=f'Regression line')
    
    # Add perfect correlation line
    min_val = min(min(automated_scores), min(expert_scores))
    max_val = max(max(automated_scores), max(expert_scores))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
           label='Perfect agreement (y=x)')
    
    # Formatting
    ax.set_xlabel('Automated Scores', fontsize=12)
    ax.set_ylabel('Expert Scores', fontsize=12)
    ax.set_title('Automated vs Expert Score Correlation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add correlation statistics
    stats_text = f'Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})\n'
    stats_text += f'Spearman ρ = {r_spearman:.3f} (p = {p_spearman:.4f})\n'
    stats_text += f'Threshold (r > 0.80): {"✓ MET" if abs(r_pearson) > 0.80 else "✗ NOT MET"}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation scatter plot saved to {output_file}")
    else:
        plt.show()

def create_bland_altman_plot(automated_scores: List[float], 
                            expert_scores: List[float],
                            output_file: Optional[str] = None):
    """
    Create Bland-Altman plot for method comparison
    
    Args:
        automated_scores: Automated evaluation scores
        expert_scores: Expert evaluation scores
        output_file: Optional output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    
    # Create scatter plot
    ax.scatter(means, differences, alpha=0.6, s=50, color='steelblue')
    
    # Add mean difference line
    ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2,
              label=f'Mean difference: {mean_diff:.3f}')
    
    # Add limits of agreement
    ax.axhline(upper_loa, color='red', linestyle='--', alpha=0.8, linewidth=2,
              label=f'Upper LoA: {upper_loa:.3f}')
    ax.axhline(lower_loa, color='red', linestyle='--', alpha=0.8, linewidth=2,
              label=f'Lower LoA: {lower_loa:.3f}')
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Check for proportional bias
    r_bias, p_bias = stats.pearsonr(means, differences)
    if abs(r_bias) > 0.3 and p_bias < 0.05:
        # Add trend line if proportional bias exists
        slope, intercept, _, _, _ = stats.linregress(means, differences)
        trend_x = np.array([min(means), max(means)])
        trend_y = slope * trend_x + intercept
        ax.plot(trend_x, trend_y, 'orange', linewidth=2, alpha=0.7,
               label=f'Trend (r = {r_bias:.3f})')
    
    # Formatting
    ax.set_xlabel('Mean of Automated and Expert Scores', fontsize=12)
    ax.set_ylabel('Difference (Automated - Expert)', fontsize=12)
    ax.set_title('Bland-Altman Plot: Method Agreement Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add agreement statistics
    agreement_text = f'Mean difference: {mean_diff:.4f}\n'
    agreement_text += f'95% LoA: [{lower_loa:.3f}, {upper_loa:.3f}]\n'
    agreement_text += f'SD of differences: {std_diff:.4f}\n'
    agreement_text += f'Proportional bias: {"Yes" if abs(r_bias) > 0.3 and p_bias < 0.05 else "No"}'
    
    ax.text(0.02, 0.98, agreement_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Bland-Altman plot saved to {output_file}")
    else:
        plt.show()

def create_correlation_matrix_plot(multiple_expert_scores: List[List[float]],
                                  output_file: Optional[str] = None):
    """
    Create correlation matrix for multiple expert raters
    
    Args:
        multiple_expert_scores: List of expert score lists
        output_file: Optional output file path
    """
    if len(multiple_expert_scores) < 2:
        print("Need at least 2 expert raters for correlation matrix")
        return
    
    # Calculate correlation matrix
    n_experts = len(multiple_expert_scores)
    correlation_matrix = np.zeros((n_experts, n_experts))
    
    for i in range(n_experts):
        for j in range(n_experts):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                r, _ = stats.pearsonr(multiple_expert_scores[i], multiple_expert_scores[j])
                correlation_matrix[i, j] = r
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pearson Correlation', rotation=270, labelpad=20)
    
    # Set ticks and labels
    expert_labels = [f'Expert {i+1}' for i in range(n_experts)]
    ax.set_xticks(range(n_experts))
    ax.set_yticks(range(n_experts))
    ax.set_xticklabels(expert_labels)
    ax.set_yticklabels(expert_labels)
    
    # Add correlation values as text
    for i in range(n_experts):
        for j in range(n_experts):
            text = ax.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Inter-Expert Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_file}")
    else:
        plt.show()

def create_reliability_summary_plot(reliability_metrics: dict, output_file: Optional[str] = None):
    """
    Create summary plot of reliability metrics
    
    Args:
        reliability_metrics: Dictionary with reliability results
        output_file: Optional output file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Cronbach's Alpha visualization
    alpha = reliability_metrics.get('cronbach_alpha', 0)
    alpha_categories = ['Poor\n(<0.7)', 'Acceptable\n(0.7-0.8)', 'Good\n(0.8-0.9)', 'Excellent\n(>0.9)', 'Observed']
    alpha_thresholds = [0.7, 0.8, 0.9, 1.0, alpha]
    colors = ['red', 'orange', 'lightgreen', 'darkgreen', 'blue']
    
    bars1 = ax1.bar(alpha_categories, alpha_thresholds, color=colors, alpha=0.7)
    
    # Highlight observed value
    bars1[-1].set_color('red')
    bars1[-1].set_alpha(0.9)
    bars1[-1].set_height(alpha)
    
    ax1.set_ylabel("Cronbach's Alpha")
    ax1.set_title("Internal Consistency Reliability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value label
    ax1.text(len(alpha_categories)-1, alpha + 0.05, f'{alpha:.3f}', 
            ha='center', va='bottom', fontweight='bold')
    
    # Mean pairwise correlation
    mean_r = reliability_metrics.get('mean_pairwise_correlation', 0)
    n_raters = reliability_metrics.get('n_raters', 2)
    
    # Create bar for mean correlation
    ax2.bar(['Mean Pairwise\nCorrelation'], [mean_r], color='steelblue', alpha=0.7)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, 
               label='Acceptable threshold (0.8)')
    
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title(f'Inter-Rater Agreement (n={n_raters} raters)')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Add value label
    ax2.text(0, mean_r + 0.05, f'{mean_r:.3f}', 
            ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Reliability summary saved to {output_file}")
    else:
        plt.show()

def main():
    """Example usage of expert correlation plots"""
    print("Creating expert correlation visualizations with example data...")
    
    # Generate correlated example data
    np.random.seed(42)
    true_scores = np.random.uniform(0.3, 0.9, 50)
    
    # Automated scores with some noise (high correlation)
    automated = true_scores + np.random.normal(0, 0.05, 50)
    automated = np.clip(automated, 0, 1)
    
    # Expert scores with different noise pattern
    expert = true_scores + np.random.normal(0, 0.03, 50)
    expert = np.clip(expert, 0, 1)
    
    # Multiple experts for reliability analysis
    expert2 = true_scores + np.random.normal(0, 0.04, 50)
    expert2 = np.clip(expert2, 0, 1)
    expert3 = true_scores + np.random.normal(0, 0.06, 50)
    expert3 = np.clip(expert3, 0, 1)
    
    # Create all visualizations
    create_correlation_scatter_plot(automated.tolist(), expert.tolist(),
                                   "expert_correlation_scatter.png")
    
    create_bland_altman_plot(automated.tolist(), expert.tolist(),
                           "expert_bland_altman.png")
    
    create_correlation_matrix_plot([expert.tolist(), expert2.tolist(), expert3.tolist()],
                                  "expert_correlation_matrix.png")
    
    # Example reliability metrics
    reliability_metrics = {
        'cronbach_alpha': 0.89,
        'mean_pairwise_correlation': 0.83,
        'n_raters': 3
    }
    
    create_reliability_summary_plot(reliability_metrics, "expert_reliability_summary.png")

if __name__ == "__main__":
    main()