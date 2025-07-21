#!/usr/bin/env python3
"""
Create Final Validation Protocol Figure
Generates flowchart showing three-way comparison framework and validation procedures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_final_validation_protocol_figure(output_file="final_validation_protocol.pdf"):
    """
    Create comprehensive Final Validation Protocol flowchart
    
    Args:
        output_file: Output file path for the figure
    """
    # Create figure with specific size for paper
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    color_input = '#e8f4fd'      # Light blue
    color_models = ['#ffebee', '#e8f5e8', '#fff3e0']  # Light red, green, orange
    color_analysis = '#f3e5f5'   # Light purple
    color_expert = '#e0f2f1'     # Light teal
    color_output = '#f9fbe7'     # Light lime
    
    # Title
    ax.text(7, 9.5, 'Final Validation Protocol', 
           fontsize=18, fontweight='bold', ha='center')
    ax.text(7, 9.1, 'Three-Way Model Comparison Framework', 
           fontsize=14, ha='center', style='italic')
    
    # 1. Input Section - 50 Unseen Topics
    input_box = FancyBboxPatch((1, 8), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color_input, 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2.5, 8.4, '50 Unseen Validation Topics', 
           fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 8.1, '(No training data exposure)', 
           fontsize=10, ha='center', style='italic')
    
    # 2. Three Model Paths
    model_names = ['Baseline Model', 'DPO-Synthetic', 'DPO-Hybrid']
    model_x_positions = [1.5, 5.5, 9.5]
    
    for i, (name, x_pos, color) in enumerate(zip(model_names, model_x_positions, color_models)):
        # Model box
        model_box = FancyBboxPatch((x_pos-1, 6.5), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=color, 
                                  edgecolor='black', linewidth=1.5)
        ax.add_patch(model_box)
        ax.text(x_pos+0.5, 7.1, name, 
               fontsize=11, fontweight='bold', ha='center')
        ax.text(x_pos+0.5, 6.8, 'Email Generation', 
               fontsize=10, ha='center')
        
        # Arrow from input to model
        arrow = ConnectionPatch((2.5, 8), (x_pos+0.5, 7.5), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=20, fc="black")
        ax.add_artist(arrow)
    
    # 3. Statistical Analysis Section
    stat_box = FancyBboxPatch((1, 4.5), 5.5, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color_analysis, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(stat_box)
    ax.text(3.75, 5.7, 'Statistical Analysis Framework', 
           fontsize=12, fontweight='bold', ha='center')
    
    # Statistical components
    stat_components = [
        'Paired t-tests (all pairs)',
        'ANOVA (three-way comparison)',
        'Effect sizes: d = 0.3-1.0',
        'Thresholds: η² > 0.06'
    ]
    
    for i, component in enumerate(stat_components):
        ax.text(1.3, 5.4 - i*0.2, f'• {component}', 
               fontsize=10, ha='left')
    
    # 4. Expert Validation Section
    expert_box = FancyBboxPatch((7.5, 4.5), 5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_expert, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(expert_box)
    ax.text(10, 5.7, 'Expert Validation', 
           fontsize=12, fontweight='bold', ha='center')
    
    # Expert components
    expert_components = [
        'Blind evaluation protocol',
        'Human professional assessment',
        'Correlation analysis: r > 0.80',
        'Automated-expert agreement'
    ]
    
    for i, component in enumerate(expert_components):
        ax.text(7.8, 5.4 - i*0.2, f'• {component}', 
               fontsize=10, ha='left')
    
    # Arrows from models to analysis sections
    for x_pos in model_x_positions:
        # To statistical analysis
        arrow1 = ConnectionPatch((x_pos+0.5, 6.5), (3.75, 6), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="gray", alpha=0.7)
        ax.add_artist(arrow1)
        
        # To expert validation
        arrow2 = ConnectionPatch((x_pos+0.5, 6.5), (10, 6), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5, 
                               mutation_scale=20, fc="gray", alpha=0.7)
        ax.add_artist(arrow2)
    
    # 5. Expected Effect Sizes Box
    effect_box = FancyBboxPatch((1, 2.5), 5.5, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#fff9c4', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(effect_box)
    ax.text(3.75, 3.7, 'Expected Effect Sizes', 
           fontsize=11, fontweight='bold', ha='center')
    
    effect_predictions = [
        'Baseline vs DPO-Synthetic: d = 0.5-0.7',
        'Baseline vs DPO-Hybrid: d = 0.7-1.0',
        'DPO-Synthetic vs DPO-Hybrid: d = 0.3-0.5'
    ]
    
    for i, prediction in enumerate(effect_predictions):
        ax.text(1.3, 3.4 - i*0.2, f'• {prediction}', 
               fontsize=10, ha='left')
    
    # 6. Validation Criteria Box
    criteria_box = FancyBboxPatch((7.5, 2.5), 5, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#ffecdf', 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(criteria_box)
    ax.text(10, 3.7, 'Validation Criteria', 
           fontsize=11, fontweight='bold', ha='center')
    
    criteria_list = [
        'Effect sizes within predicted ranges',
        'Statistical significance (p < 0.05)',
        'Practical significance (η² > 0.06)',
        'Expert agreement (r > 0.80)'
    ]
    
    for i, criteria in enumerate(criteria_list):
        ax.text(7.8, 3.4 - i*0.2, f'• {criteria}', 
               fontsize=10, ha='left')
    
    # 7. Final Output Section
    output_box = FancyBboxPatch((4, 0.5), 6, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_output, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 1.3, 'Validation Results', 
           fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 1, 'PASS: All criteria met | PARTIAL: Some criteria met | FAIL: Criteria not met', 
           fontsize=10, ha='center', style='italic')
    ax.text(7, 0.7, 'Publication-ready statistical evidence for optimization effectiveness', 
           fontsize=10, ha='center')
    
    # Arrows to final output
    arrow3 = ConnectionPatch((3.75, 2.5), (6, 1.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="black")
    ax.add_artist(arrow3)
    
    arrow4 = ConnectionPatch((10, 2.5), (8, 1.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="black")
    ax.add_artist(arrow4)
    
    # Add methodology phases labels
    ax.text(0.2, 8.4, 'INPUT', fontsize=10, fontweight='bold', 
           rotation=90, va='center', ha='center')
    ax.text(0.2, 7, 'PROCESSING', fontsize=10, fontweight='bold', 
           rotation=90, va='center', ha='center')
    ax.text(0.2, 5.2, 'ANALYSIS', fontsize=10, fontweight='bold', 
           rotation=90, va='center', ha='center')
    ax.text(0.2, 3.2, 'VALIDATION', fontsize=10, fontweight='bold', 
           rotation=90, va='center', ha='center')
    ax.text(0.2, 1.1, 'OUTPUT', fontsize=10, fontweight='bold', 
           rotation=90, va='center', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Also save as PNG for easy viewing
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Final Validation Protocol figure saved as:")
    print(f"  PDF: {output_file}")
    print(f"  PNG: {png_file}")
    
    plt.show()

def main():
    """Create the Final Validation Protocol figure"""
    print("Creating Final Validation Protocol figure...")
    create_final_validation_protocol_figure()

if __name__ == "__main__":
    main()