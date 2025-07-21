#!/usr/bin/env python3
"""
Complete Validation Pipeline Integration
Runs all components and generates comprehensive analysis
"""

import json
import argparse
from pathlib import Path
from typing import Optional, Dict

# Import analysis modules
from three_way_comparison_analysis import run_complete_analysis
from expert_validation_analysis import run_expert_validation_analysis
from power_analysis import validate_methodology_power_requirements
from generate_validation_report import create_validation_report

# Import visualization modules
from effect_size_plots import create_forest_plot, create_effect_size_comparison
from anova_results_plot import (create_model_comparison_boxplot, 
                               create_anova_summary_plot, 
                               create_means_comparison_plot)
from expert_correlation_plot import (create_correlation_scatter_plot,
                                    create_bland_altman_plot)

def load_score_data(file_path: str):
    """Load scores from complete_results.json file"""
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

def run_complete_validation_pipeline(baseline_file: str,
                                    synthetic_file: str, 
                                    hybrid_file: str,
                                    expert_automated_scores: Optional[list] = None,
                                    expert_human_scores: Optional[list] = None,
                                    output_dir: str = "validation_results"):
    """
    Run complete validation pipeline with all analyses and visualizations
    
    Args:
        baseline_file: Path to baseline results
        synthetic_file: Path to DPO-Synthetic results  
        hybrid_file: Path to DPO-Hybrid results
        expert_automated_scores: Optional automated scores for expert validation
        expert_human_scores: Optional expert scores for validation
        output_dir: Output directory for results
    
    Returns:
        Dict with all analysis results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("FINAL VALIDATION PROTOCOL - COMPLETE ANALYSIS")
    print("="*60)
    
    # Stage 1: Core Statistical Analysis
    print("\n[STAGE 1] Running three-way comparison analysis...")
    analysis_results = run_complete_analysis(baseline_file, synthetic_file, hybrid_file)
    
    if not analysis_results:
        print("ERROR: Core analysis failed")
        return {}
    
    # Save analysis results
    analysis_output = output_path / "statistical_analysis_results.json"
    with open(analysis_output, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Statistical analysis saved to {analysis_output}")
    
    # Stage 2: Expert Validation (if data available)
    expert_results = None
    if expert_automated_scores and expert_human_scores:
        print("\n[STAGE 2] Running expert validation analysis...")
        expert_results = run_expert_validation_analysis(expert_automated_scores, expert_human_scores)
        
        expert_output = output_path / "expert_validation_results.json"
        with open(expert_output, 'w') as f:
            json.dump(expert_results, f, indent=2)
        print(f"Expert validation saved to {expert_output}")
    else:
        print("\n[STAGE 2] Expert validation data not provided - skipping")
    
    # Stage 2: Power Analysis
    print("\n[STAGE 2] Running power analysis...")
    power_results = validate_methodology_power_requirements(analysis_results)
    
    power_output = output_path / "power_analysis_results.json"
    with open(power_output, 'w') as f:
        json.dump(power_results, f, indent=2)
    print(f"Power analysis saved to {power_output}")
    
    # Stage 3: Comprehensive Reporting
    print("\n[STAGE 3] Generating comprehensive validation report...")
    report_file = output_path / "comprehensive_validation_report.md"
    create_validation_report(analysis_results, expert_results, power_results, str(report_file))
    
    # Stage 3: Visualizations
    print("\n[STAGE 3] Creating visualizations...")
    
    # Load score data for visualizations
    baseline_scores = load_score_data(baseline_file)
    synthetic_scores = load_score_data(synthetic_file)
    hybrid_scores = load_score_data(hybrid_file)
    
    # Effect size plots
    if 'effect_sizes' in analysis_results:
        print("  Creating effect size plots...")
        create_forest_plot(analysis_results['effect_sizes'], 
                          str(output_path / "effect_size_forest_plot.png"))
        create_effect_size_comparison(analysis_results['effect_sizes'],
                                    str(output_path / "effect_size_comparison.png"))
    
    # ANOVA plots
    if all([baseline_scores, synthetic_scores, hybrid_scores]):
        print("  Creating ANOVA visualizations...")
        create_model_comparison_boxplot(baseline_scores, synthetic_scores, hybrid_scores,
                                       str(output_path / "model_comparison_boxplot.png"))
        
        if 'anova_results' in analysis_results:
            create_anova_summary_plot(analysis_results['anova_results'],
                                     str(output_path / "anova_summary.png"))
        
        if 'summary_statistics' in analysis_results:
            create_means_comparison_plot(analysis_results['summary_statistics'],
                                        str(output_path / "means_comparison.png"))
    
    # Expert correlation plots
    if expert_results and expert_automated_scores and expert_human_scores:
        print("  Creating expert correlation plots...")
        create_correlation_scatter_plot(expert_automated_scores, expert_human_scores,
                                       str(output_path / "expert_correlation_scatter.png"))
        create_bland_altman_plot(expert_automated_scores, expert_human_scores,
                                str(output_path / "expert_bland_altman.png"))
    
    print(f"\n[COMPLETE] All results saved to: {output_path}")
    
    # Summary of validation status
    validation_summary = analysis_results.get('methodology_validation_summary', {})
    overall_status = validation_summary.get('overall_validation_status', 'UNKNOWN')
    
    print(f"\nVALIDATION SUMMARY:")
    print(f"Overall Status: {overall_status}")
    print(f"Effect Sizes Validated: {'✓' if validation_summary.get('effect_sizes_within_predicted_ranges', False) else '✗'}")
    print(f"ANOVA Threshold Met: {'✓' if validation_summary.get('eta_squared_meets_threshold', False) else '✗'}")
    
    if expert_results:
        expert_validation = expert_results.get('validation_summary', {})
        print(f"Expert Correlation Met: {'✓' if expert_validation.get('meets_correlation_threshold', False) else '✗'}")
    
    return {
        'statistical_analysis': analysis_results,
        'expert_validation': expert_results,
        'power_analysis': power_results,
        'output_directory': str(output_path)
    }

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run complete validation pipeline')
    parser.add_argument('--baseline', required=True, help='Baseline results file')
    parser.add_argument('--synthetic', required=True, help='DPO-Synthetic results file')
    parser.add_argument('--hybrid', required=True, help='DPO-Hybrid results file')
    parser.add_argument('--expert-automated', help='Expert validation automated scores JSON file')
    parser.add_argument('--expert-human', help='Expert validation human scores JSON file')
    parser.add_argument('--output', default='validation_results', help='Output directory')
    parser.add_argument('--example', action='store_true', help='Run with example data')
    
    args = parser.parse_args()
    
    if args.example:
        print("Running complete validation pipeline with example data...")
        
        # Create example data files
        import numpy as np
        np.random.seed(42)
        
        temp_dir = Path('temp_validation_example')
        temp_dir.mkdir(exist_ok=True)
        
        # Generate example score data
        baseline_scores = np.random.normal(0.5, 0.1, 50).tolist()
        synthetic_scores = np.random.normal(0.65, 0.1, 50).tolist()
        hybrid_scores = np.random.normal(0.8, 0.1, 50).tolist()
        
        # Create example files
        for name, scores in [('baseline', baseline_scores), 
                           ('synthetic', synthetic_scores), 
                           ('hybrid', hybrid_scores)]:
            data = {
                'results': []
            }
            for i, score in enumerate(scores):
                topic_result = {
                    'topic_id': f'T{i+1:04d}',
                    'emails': [{'evaluation': {'overall_score': score}}]
                }
                data['results'].append(topic_result)
            
            with open(temp_dir / f'{name}_results.json', 'w') as f:
                json.dump(data, f)
        
        # Example expert data
        true_scores = np.random.uniform(0.3, 0.9, 30)
        expert_automated = (true_scores + np.random.normal(0, 0.05, 30)).tolist()
        expert_human = (true_scores + np.random.normal(0, 0.03, 30)).tolist()
        
        # Run pipeline
        results = run_complete_validation_pipeline(
            str(temp_dir / 'baseline_results.json'),
            str(temp_dir / 'synthetic_results.json'),
            str(temp_dir / 'hybrid_results.json'),
            expert_automated,
            expert_human,
            args.output
        )
        
    else:
        # Load expert data if provided
        expert_automated = None
        expert_human = None
        
        if args.expert_automated and args.expert_human:
            with open(args.expert_automated, 'r') as f:
                expert_automated = json.load(f)
            with open(args.expert_human, 'r') as f:
                expert_human = json.load(f)
        
        # Run pipeline with real data
        results = run_complete_validation_pipeline(
            args.baseline,
            args.synthetic, 
            args.hybrid,
            expert_automated,
            expert_human,
            args.output
        )

if __name__ == "__main__":
    main()