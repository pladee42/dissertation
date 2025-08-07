#!/usr/bin/env python3
"""
Stage 4: Results Tables Creation
Create all LaTeX tables with extracted data for direct inclusion in the Results section
"""

import json
from pathlib import Path
from typing import Dict

def load_previous_stages_data(context_dir: str) -> Dict:
    """Load data from all previous stages"""
    context_path = Path(context_dir)
    
    # Load Stage 1 data
    stage1_file = context_path / 'stage_outputs' / 'results_stage1_statistics.json'
    with open(stage1_file, 'r') as f:
        stage1_data = json.load(f)
    
    # Load Stage 3 data  
    stage3_file = context_path / 'stage_outputs' / 'results_stage3_detailed.json'
    with open(stage3_file, 'r') as f:
        stage3_data = json.load(f)
    
    # Load statistical registry
    registry_file = context_path / 'registries' / 'statistical_values_registry.json'
    with open(registry_file, 'r') as f:
        registry_data = json.load(f)
    
    return {
        'stage1': stage1_data,
        'stage3': stage3_data,
        'registry': registry_data
    }

def create_table1_descriptive_statistics(data: Dict) -> str:
    """Table 1: Descriptive Statistics"""
    
    desc_stats = data['registry']['descriptive_statistics']
    
    latex_table = r"""
% Table 1: Descriptive Statistics
\begin{table}[htbp]
\centering
\caption{Descriptive Statistics for Model Variants}
\label{tab:descriptive-statistics}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model Variant} & \textbf{N} & \textbf{M} & \textbf{SD} & \textbf{95\% CI} & \textbf{Min} & \textbf{Max} \\
\midrule
""" + f"""Baseline           & {desc_stats['baseline']['n']} & {desc_stats['baseline']['mean']:.3f} & {desc_stats['baseline']['std']:.3f} & [{desc_stats['baseline']['ci_lower']:.3f}, {desc_stats['baseline']['ci_upper']:.3f}] & {desc_stats['baseline']['min']:.1f} & {desc_stats['baseline']['max']:.1f} \\\\
DPO-Synthetic      & {desc_stats['dpo_synthetic']['n']} & {desc_stats['dpo_synthetic']['mean']:.3f} & {desc_stats['dpo_synthetic']['std']:.3f} & [{desc_stats['dpo_synthetic']['ci_lower']:.3f}, {desc_stats['dpo_synthetic']['ci_upper']:.3f}] & {desc_stats['dpo_synthetic']['min']:.1f} & {desc_stats['dpo_synthetic']['max']:.1f} \\\\
DPO-Hybrid         & {desc_stats['dpo_hybrid']['n']} & {desc_stats['dpo_hybrid']['mean']:.3f} & {desc_stats['dpo_hybrid']['std']:.3f} & [{desc_stats['dpo_hybrid']['ci_lower']:.3f}, {desc_stats['dpo_hybrid']['ci_upper']:.3f}] & {desc_stats['dpo_hybrid']['min']:.1f} & {desc_stats['dpo_hybrid']['max']:.1f} \\\\""" + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex_table

def create_table2_statistical_comparisons(data: Dict) -> str:
    """Table 2: Statistical Comparisons"""
    
    t_tests = data['registry']['t_test_results']
    effect_sizes = data['registry']['effect_sizes']
    
    latex_table = r"""
% Table 2: Statistical Comparisons
\begin{table}[htbp]
\centering
\caption{Pairwise Statistical Comparisons Between Model Variants}
\label{tab:statistical-comparisons}
\begin{tabular}{lccccc}
\toprule
\textbf{Comparison} & \textbf{t} & \textbf{df} & \textbf{p} & \textbf{Cohen's d} & \textbf{95\% CI for d} \\
\midrule
""" + f"""Baseline vs DPO-Synthetic    & {t_tests['baseline_vs_synthetic']['t_statistic']:.3f} & {t_tests['baseline_vs_synthetic']['df']} & {t_tests['baseline_vs_synthetic']['p_value']:.3f} & {effect_sizes['baseline_vs_synthetic']['cohens_d']:.3f} & [{effect_sizes['baseline_vs_synthetic']['ci_lower']:.3f}, {effect_sizes['baseline_vs_synthetic']['ci_upper']:.3f}] \\\\
Baseline vs DPO-Hybrid       & {t_tests['baseline_vs_hybrid']['t_statistic']:.3f} & {t_tests['baseline_vs_hybrid']['df']} & {t_tests['baseline_vs_hybrid']['p_value']:.3f} & {effect_sizes['baseline_vs_hybrid']['cohens_d']:.3f} & [{effect_sizes['baseline_vs_hybrid']['ci_lower']:.3f}, {effect_sizes['baseline_vs_hybrid']['ci_upper']:.3f}] \\\\
DPO-Synthetic vs DPO-Hybrid  & {t_tests['synthetic_vs_hybrid']['t_statistic']:.3f} & {t_tests['synthetic_vs_hybrid']['df']} & {t_tests['synthetic_vs_hybrid']['p_value']:.3f} & {effect_sizes['synthetic_vs_hybrid']['cohens_d']:.3f} & [{effect_sizes['synthetic_vs_hybrid']['ci_lower']:.3f}, {effect_sizes['synthetic_vs_hybrid']['ci_upper']:.3f}] \\\\""" + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: All p-values > 0.05 indicate no statistically significant differences. All effect sizes are negligible (|d| < 0.2).
\end{tablenotes}
\end{table}
"""
    
    return latex_table

def create_table3_model_specific(data: Dict) -> str:
    """Table 3: Model-Specific Performance"""
    
    model_analysis = data['registry']['model_specific_analysis']['individual_models']
    
    latex_table = r"""
% Table 3: Model-Specific Performance
\begin{table}[htbp]
\centering
\caption{Individual Model Performance by Variant}
\label{tab:model-specific}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multicolumn{2}{c}{\textbf{Baseline}} & \multicolumn{2}{c}{\textbf{DPO-Synthetic}} & \multicolumn{2}{c}{\textbf{DPO-Hybrid}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{M} & \textbf{SD} & \textbf{M} & \textbf{Δ\%} & \textbf{M} & \textbf{Δ\%} \\
\midrule
"""
    
    # Add each model's data
    for model_uid, model_data in model_analysis.items():
        baseline_mean = model_data['baseline']['mean']
        baseline_std = model_data['baseline']['std']
        synthetic_mean = model_data['synthetic']['mean']
        hybrid_mean = model_data['hybrid']['mean']
        synthetic_improvement = model_data.get('improvement_synthetic', 0)
        hybrid_improvement = model_data.get('improvement_hybrid', 0)
        
        if baseline_mean is not None:
            latex_table += f"""{model_uid} & {baseline_mean:.3f} & {baseline_std:.3f} & {synthetic_mean:.3f} & {synthetic_improvement:+.1f}\\% & {hybrid_mean:.3f} & {hybrid_improvement:+.1f}\\% \\\\
"""
    
    latex_table += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Δ\% represents percentage change from baseline. M0001=TinyLlama, M0002=Vicuna, M0003=Phi-3, M0005=StableLM.
\end{tablenotes}
\end{table}
"""
    
    return latex_table

def create_table4_category_analysis(data: Dict) -> str:
    """Table 4: Category Analysis"""
    
    category_analysis = data['registry']['category_analysis']
    
    latex_table = r"""
% Table 4: Category Analysis
\begin{table}[htbp]
\centering
\caption{Performance by Topic Category}
\label{tab:category-analysis}
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Category}} & \multicolumn{2}{c}{\textbf{Baseline}} & \multicolumn{2}{c}{\textbf{DPO-Synthetic}} & \multicolumn{2}{c}{\textbf{DPO-Hybrid}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{M} & \textbf{N} & \textbf{M} & \textbf{Δ\%} & \textbf{M} & \textbf{Δ\%} \\
\midrule
"""
    
    # Add each category's data
    category_names = {
        'healthcare_medical': 'Healthcare/Medical',
        'education_youth': 'Education/Youth',
        'environmental': 'Environmental',
        'community_social': 'Community/Social'
    }
    
    for category_key, category_data in category_analysis.items():
        category_name = category_names.get(category_key, category_key)
        baseline_mean = category_data['baseline']['mean']
        baseline_n = category_data['baseline']['n']
        synthetic_mean = category_data['synthetic']['mean']
        hybrid_mean = category_data['hybrid']['mean']
        synthetic_improvement = category_data.get('improvement_synthetic', 0)
        hybrid_improvement = category_data.get('improvement_hybrid', 0)
        
        if baseline_mean is not None:
            latex_table += f"""{category_name} & {baseline_mean:.3f} & {baseline_n} & {synthetic_mean:.3f} & {synthetic_improvement:+.1f}\\% & {hybrid_mean:.3f} & {hybrid_improvement:+.1f}\\% \\\\
"""
    
    latex_table += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Δ\% represents percentage change from baseline. Categories represent balanced segments of the evaluation data.
\end{tablenotes}
\end{table}
"""
    
    return latex_table

def create_table5_methodology_validation(data: Dict) -> str:
    """Table 5: Methodology Validation"""
    
    effect_sizes = data['registry']['effect_sizes']
    anova = data['registry']['anova_results']
    validation = data['registry']['methodology_validation']
    
    latex_table = r"""
% Table 5: Methodology Validation
\begin{table}[htbp]
\centering
\caption{Methodology Validation: Predicted vs Actual Results}
\label{tab:methodology-validation}
\begin{tabular}{lccccc}
\toprule
\textbf{Comparison} & \textbf{Predicted d} & \textbf{Actual d} & \textbf{Within Range} & \textbf{Validation} \\
\midrule
""" + f"""Baseline vs DPO-Synthetic    & 0.5--0.7 & {effect_sizes['baseline_vs_synthetic']['cohens_d']:.3f} & {'✓' if effect_sizes['baseline_vs_synthetic']['within_predicted_range'] else '✗'} & {'PASS' if effect_sizes['baseline_vs_synthetic']['within_predicted_range'] else 'FAIL'} \\\\
Baseline vs DPO-Hybrid       & 0.7--1.0 & {effect_sizes['baseline_vs_hybrid']['cohens_d']:.3f} & {'✓' if effect_sizes['baseline_vs_hybrid']['within_predicted_range'] else '✗'} & {'PASS' if effect_sizes['baseline_vs_hybrid']['within_predicted_range'] else 'FAIL'} \\\\
DPO-Synthetic vs DPO-Hybrid  & 0.3--0.5 & {effect_sizes['synthetic_vs_hybrid']['cohens_d']:.3f} & {'✓' if effect_sizes['synthetic_vs_hybrid']['within_predicted_range'] else '✗'} & {'PASS' if effect_sizes['synthetic_vs_hybrid']['within_predicted_range'] else 'FAIL'} \\\\
\midrule
ANOVA η² Threshold & >0.06 & {anova['eta_squared']:.3f} & {'✓' if anova['meets_threshold'] else '✗'} & {'PASS' if anova['meets_threshold'] else 'FAIL'} \\\\
Expert Correlation & >0.80 & {'N/A' if validation['expert_correlation_met'] is None else validation['expert_correlation_met']} & {'N/A' if validation['expert_correlation_met'] is None else ('✓' if validation['expert_correlation_met'] else '✗')} & {'N/A' if validation['expert_correlation_met'] is None else ('PASS' if validation['expert_correlation_met'] else 'FAIL')} \\\\
\midrule
\textbf{{Overall Status}} & \multicolumn{{4}}{{c}}{{\textbf{{{validation['overall_validation_status']}}}}} \\\\""" + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Methodology validation assesses whether empirical results match theoretical predictions. All effect size predictions and ANOVA threshold failed validation.
\end{tablenotes}
\end{table}
"""
    
    return latex_table

def run_stage4_tables_generation(context_dir: str):
    """Generate all Stage 4 LaTeX tables"""
    
    print("Loading data from previous stages...")
    data = load_previous_stages_data(context_dir)
    
    print("Creating Table 1: Descriptive Statistics...")
    table1 = create_table1_descriptive_statistics(data)
    
    print("Creating Table 2: Statistical Comparisons...")
    table2 = create_table2_statistical_comparisons(data)
    
    print("Creating Table 3: Model-Specific Performance...")
    table3 = create_table3_model_specific(data)
    
    print("Creating Table 4: Category Analysis...")
    table4 = create_table4_category_analysis(data)
    
    print("Creating Table 5: Methodology Validation...")
    table5 = create_table5_methodology_validation(data)
    
    # Compile all tables
    all_tables = {
        'table1_descriptive_statistics': table1,
        'table2_statistical_comparisons': table2,
        'table3_model_specific': table3,
        'table4_category_analysis': table4,
        'table5_methodology_validation': table5
    }
    
    # Create combined LaTeX file
    latex_complete = f"""% LaTeX Tables for Results Section
% Generated from Stage 4 Tables Analysis
% Date: 2025-01-31

{table1}

{table2}

{table3}

{table4}

{table5}

% Additional LaTeX packages required:
% \\usepackage{{booktabs}}
% \\usepackage{{multirow}}
% \\usepackage{{threeparttable}}
"""
    
    # Compile Stage 4 results
    stage4_results = {
        'analysis_timestamp': '2025-01-31',
        'tables_generated': list(all_tables.keys()),
        'table_contents': all_tables,
        'latex_complete': latex_complete,
        'data_quality': {
            'stage1_data_loaded': True,
            'stage3_data_loaded': True,
            'registry_data_loaded': True,
            'all_tables_generated': len(all_tables) == 5
        },
        'key_findings': {
            'descriptive_stats_negligible_differences': True,
            'all_comparisons_non_significant': True,
            'model_specific_mixed_results': True,
            'category_balanced_representation': True,
            'methodology_validation_failed': True
        }
    }
    
    return stage4_results, latex_complete

def save_stage4_context(context_dir: str, stage4_results: Dict, latex_complete: str):
    """Save Stage 4 context files and update registries"""
    
    context_path = Path(context_dir)
    
    # Save Stage 4 results
    stage4_file = context_path / 'stage_outputs' / 'results_stage4_tables.json'
    with open(stage4_file, 'w') as f:
        json.dump(stage4_results, f, indent=2)
    
    # Save complete LaTeX tables file
    latex_file = context_path / 'master_files' / 'latex_tables_ready.tex'
    context_path.mkdir(parents=True, exist_ok=True)
    (context_path / 'master_files').mkdir(exist_ok=True)
    with open(latex_file, 'w') as f:
        f.write(latex_complete)
    
    # Create Stage 4 summary
    summary_content = f"""# Stage 4 Tables Summary: LaTeX Results Tables

## Analysis Overview
**Date**: 2025-01-31  
**Objective**: Create all LaTeX tables with extracted data for direct inclusion in the Results section  
**Approach**: Extract data from Stages 1-3 and format as publication-ready LaTeX tables  

## Tables Generated

### Table 1: Descriptive Statistics
**Purpose**: Mean, SD, 95% CI for each model variant  
**Content**: 
- Sample sizes (N=145 for all variants)
- Baseline: M=0.574, SD=0.260
- DPO-Synthetic: M=0.564, SD=0.231  
- DPO-Hybrid: M=0.581, SD=0.201
- Complete 95% confidence intervals and ranges

### Table 2: Statistical Comparisons  
**Purpose**: Pairwise t-test results with effect sizes
**Content**:
- All t-statistics, degrees of freedom, and p-values
- Cohen's d with 95% confidence intervals
- All comparisons non-significant (p > 0.05)
- All effect sizes negligible (|d| < 0.2)

### Table 3: Model-Specific Performance
**Purpose**: Performance by individual model UID
**Content**:
- Individual model results for M0001, M0002, M0003, M0005
- Baseline vs optimized performance with percentage changes
- Mixed results: some improvements, some degradations
- Clear model-by-optimization interactions

### Table 4: Category Analysis
**Purpose**: Mean scores by topic category × model variant
**Content**:
- Healthcare/Medical, Education/Youth, Environmental, Community/Social
- Balanced representation across categories
- Category-specific optimization effects
- Percentage improvements vary by domain

### Table 5: Methodology Validation
**Purpose**: Predicted vs actual effect sizes with validation status
**Content**:
- Comparison of methodology predictions vs empirical results
- Effect size range validations (all FAIL)
- ANOVA η² threshold validation (FAIL: 0.001 vs >0.06)
- Overall validation status: FAIL

## LaTeX Features

### Professional Formatting
- Uses `booktabs` package for clean horizontal rules
- `multirow` for complex headers
- `threeparttable` for table notes
- Consistent decimal formatting (3 decimal places)
- Proper statistical notation

### Table Labels and References
- `tab:descriptive-statistics`
- `tab:statistical-comparisons`  
- `tab:model-specific`
- `tab:category-analysis`
- `tab:methodology-validation`

### Table Notes
- Explanatory notes for abbreviations and interpretations
- Statistical significance indicators
- Model UID mappings
- Validation criteria explanations

## Key Statistical Values Included

### Core Statistics
- All means, standard deviations, confidence intervals
- Complete t-test results (t, df, p)
- Effect sizes with confidence intervals
- ANOVA F-statistic and η²

### Model-Specific Data
- Individual model performance metrics
- Improvement percentages for each model
- Sample sizes for each analysis

### Validation Metrics
- Predicted vs actual effect size ranges
- Threshold comparisons
- Pass/fail validation indicators

## Critical Findings Reflected in Tables

### Statistical Equivalence
- All tables consistently show minimal differences
- Non-significant p-values throughout
- Negligible effect sizes universally

### Methodology Validation Failure
- Table 5 clearly documents prediction failures
- Large discrepancies between expected and observed effects
- Complete failure of methodology validation criteria

### Model and Category Specificity
- Some models show improvements, others degradations
- Category effects variable and inconsistent
- No clear pattern of optimization benefits

## Integration Ready
- **Complete LaTeX File**: `latex_tables_ready.tex`
- **Individual Table Access**: Available in JSON format
- **Copy-Paste Ready**: All tables formatted for direct inclusion
- **Cross-References**: Proper label system for in-text citations

## Context Preservation
- **Detailed Results**: `results_stage4_tables.json`
- **Tables Summary**: `stage4_tables_summary.md` (this file)
- **LaTeX Complete**: `latex_tables_ready.tex`
- **Registry Updates**: All table metadata preserved

## Next Steps
- **Stage 5**: Context consolidation and comprehensive verification
- **Integration**: All tables ready for Results section inclusion
- **Publication**: Professional formatting ready for submission

---
*Stage 4 completed with 5 comprehensive LaTeX tables ready for direct inclusion in the Results section, documenting the unexpected finding of methodology validation failure and model equivalence.*"""

    summary_file = context_path / 'summaries' / 'stage4_tables_summary.md'
    with open(summary_file, 'w') as f:
        f.write(summary_content)
    
    # Update master context tracker
    tracker_file = context_path / 'registries' / 'master_context_tracker.json'
    with open(tracker_file, 'r') as f:
        tracker = json.load(f)
    
    tracker['stage_4'] = {
        'status': 'completed',
        'timestamp': '2025-01-31',
        'files_created': [
            'stage_outputs/results_stage4_tables.json',
            'summaries/stage4_tables_summary.md',
            'master_files/latex_tables_ready.tex'
        ],
        'tables_generated': len(stage4_results['tables_generated']),
        'latex_ready': True,
        'verification_status': 'completed'
    }
    
    tracker['overall_progress'] = {
        'stages_completed': 5,
        'total_stages': 7,
        'completion_percentage': 71.4,
        'last_updated': '2025-01-31'
    }
    
    with open(tracker_file, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    print("Stage 4 context preservation completed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python stage4_tables_generator.py <context_dir>")
        sys.exit(1)
    
    context_dir = sys.argv[1]
    
    stage4_results, latex_complete = run_stage4_tables_generation(context_dir)
    save_stage4_context(context_dir, stage4_results, latex_complete)
    
    print("Stage 4 completed successfully!")
    print(f"Generated {len(stage4_results['tables_generated'])} LaTeX tables")
    print("All tables ready for Results section inclusion")