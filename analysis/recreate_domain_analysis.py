#!/usr/bin/env python3
"""
Recreate domain-specific performance analysis with correct topic categorization.
Uses validation topics T0101-T0150 from config/topic_val.json.
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def categorize_validation_topics(topics_file: str) -> Dict[str, str]:
    """
    Categorize validation topics T0101-T0150 based on keywords.
    Returns mapping of topic_uid to category.
    """
    # Category definitions adapted from analyze_topics.py
    categories = {
        "Environmental/Wildlife Conservation": [
            "logging", "congo basin", "vaquita porpoise", "ocean", "plastic",
            "reforestation", "fire", "climate change", "extreme weather",
            "marine protected", "river cleanup", "orangutans", "earth day",
            "anti-poaching", "planet"
        ],
        "Human Rights/Social Justice": [
            "safe schools", "equality", "lgbtq+", "workplace", "canvass",
            "crisis hotline", "bisexual", "discriminatory bills",
            "marriage equality", "senior healthcare", "activists",
            "war crimes", "journalists", "displaced families",
            "facial recognition", "dissent", "clean water", "human right",
            "women's rights", "child marriage", "the hague", "iran",
            "yemen", "myanmar", "sudan"
        ],
        "Community/Charity Support": [
            "foster parents", "school supplies", "champions for children",
            "mobile library", "read to children", "family in crisis",
            "musical instruments", "arts program", "cyberbullying",
            "playground", "boys & girls club", "community center"
        ],
        "Media/Broadcasting/News": [
            "fund drive", "documentary", "science friday", "journalists",
            "pitches", "public radio", "tote bag", "misinformation",
            "election", "listener", "public media"
        ]
    }
    
    # Load validation topics
    with open(topics_file, 'r') as f:
        topics = json.load(f)
    
    topic_categories = {}
    uncategorized = []
    
    for topic in topics:
        topic_uid = topic['uid']
        topic_name = topic['topic_name'].lower()
        assigned = False
        
        # Try to categorize based on keywords
        for category, keywords in categories.items():
            if any(keyword in topic_name for keyword in keywords):
                topic_categories[topic_uid] = category
                assigned = True
                break
        
        if not assigned:
            # Manual assignment for uncategorized based on topic UID ranges
            if topic_uid in ['T0101', 'T0102', 'T0103', 'T0104', 'T0105', 
                           'T0106', 'T0107', 'T0108', 'T0109', 'T0110']:
                topic_categories[topic_uid] = "Environmental/Wildlife Conservation"
            elif topic_uid in ['T0111', 'T0112', 'T0113', 'T0114', 'T0115',
                              'T0116', 'T0117', 'T0118', 'T0119', 'T0120',
                              'T0131', 'T0132', 'T0133', 'T0134', 'T0135',
                              'T0136', 'T0137', 'T0138', 'T0139', 'T0140']:
                topic_categories[topic_uid] = "Human Rights/Social Justice"
            elif topic_uid in ['T0121', 'T0122', 'T0123', 'T0124', 'T0125',
                              'T0126', 'T0127', 'T0128', 'T0129', 'T0130']:
                topic_categories[topic_uid] = "Community/Charity Support"
            elif topic_uid in ['T0141', 'T0142', 'T0143', 'T0144', 'T0145',
                              'T0146', 'T0147', 'T0148', 'T0149', 'T0150']:
                topic_categories[topic_uid] = "Media/Broadcasting/News"
            else:
                uncategorized.append((topic_uid, topic['topic_name']))
    
    if uncategorized:
        print(f"Warning: {len(uncategorized)} topics could not be categorized")
        for uid, name in uncategorized:
            print(f"  - {uid}: {name}")
    
    return topic_categories

def load_results_data(filepath: str, variant_name: str) -> Dict:
    """Load and parse complete_results.json file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\nLoading {variant_name} data from: {filepath}")
    
    # Check data structure
    if 'successful_results' in data:
        results_key = 'successful_results'
    elif 'results' in data:
        results_key = 'results'
    else:
        raise ValueError(f"Cannot find results in {filepath}")
    
    print(f"  Found {len(data[results_key])} results under key '{results_key}'")
    
    # Extract scores by topic and model
    topic_scores = {}
    
    for result in data[results_key]:
        # Get topic UID - it's directly in the result
        topic_uid = result.get('topic_uid')
        if not topic_uid:
            # Try other possible locations
            if 'topic' in result and isinstance(result['topic'], dict):
                topic_uid = result['topic'].get('uid')
            
        if not topic_uid:
            continue
            
        if 'emails' in result:
            for email in result['emails']:
                # Get model ID
                model_id = email.get('model_id', email.get('model_name', 'unknown'))
                
                # Get score - check different possible locations
                score = None
                if 'overall_score' in email:
                    score = email['overall_score']
                elif 'evaluation' in email and 'overall_score' in email['evaluation']:
                    score = email['evaluation']['overall_score']
                elif 'evaluation' in email and 'weighted_score' in email['evaluation']:
                    score = email['evaluation']['weighted_score']
                
                if score is not None:
                    if topic_uid not in topic_scores:
                        topic_scores[topic_uid] = []
                    topic_scores[topic_uid].append({
                        'model': model_id,
                        'score': score
                    })
    
    print(f"  Extracted scores for {len(topic_scores)} topics")
    
    return topic_scores

def calculate_category_statistics(topic_scores: Dict, topic_categories: Dict) -> Dict:
    """Calculate statistics by category."""
    # Group scores by category
    category_scores = {}
    
    for topic_uid, scores in topic_scores.items():
        if topic_uid in topic_categories:
            category = topic_categories[topic_uid]
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].extend([s['score'] for s in scores])
    
    # Calculate statistics
    category_stats = {}
    for category, scores in category_scores.items():
        if scores:
            category_stats[category] = {
                'n': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores, ddof=1) if len(scores) > 1 else 0,
                'min': np.min(scores),
                'max': np.max(scores)
            }
        else:
            category_stats[category] = {
                'n': 0,
                'mean': None,
                'std': None,
                'min': None,
                'max': None
            }
    
    return category_stats

def main():
    """Main analysis function."""
    print("="*70)
    print("DOMAIN-SPECIFIC PERFORMANCE ANALYSIS - CORRECTED")
    print("="*70)
    
    # Step 1: Categorize validation topics
    print("\nStep 1: Categorizing validation topics...")
    topic_categories = categorize_validation_topics('config/topic_val.json')
    
    # Count topics per category
    category_counts = {}
    for category in set(topic_categories.values()):
        category_counts[category] = sum(1 for c in topic_categories.values() if c == category)
    
    print("\nTopic distribution across categories:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} topics")
    
    # Step 2: Load experimental data
    print("\nStep 2: Loading experimental data...")
    
    data_files = {
        'Baseline': 'output/multi_topic_results/20250722_061212/complete_results.json',
        'DPO-Synthetic': 'output/multi_topic_results/20250722_123509/complete_results.json',
        'DPO-Hybrid': 'output/multi_topic_results/20250731_164142/complete_results.json'
    }
    
    variant_data = {}
    for variant, filepath in data_files.items():
        try:
            variant_data[variant] = load_results_data(filepath, variant)
        except Exception as e:
            print(f"  Error loading {variant}: {e}")
            variant_data[variant] = {}
    
    # Step 3: Calculate statistics by category
    print("\nStep 3: Calculating category statistics...")
    
    category_analysis = {}
    for variant, topic_scores in variant_data.items():
        print(f"\nAnalyzing {variant}...")
        stats = calculate_category_statistics(topic_scores, topic_categories)
        
        for category, cat_stats in stats.items():
            if category not in category_analysis:
                category_analysis[category] = {}
            category_analysis[category][variant] = cat_stats
            
            if cat_stats['mean'] is not None:
                print(f"  {category}: N={cat_stats['n']}, M={cat_stats['mean']:.3f}, SD={cat_stats['std']:.3f}")
    
    # Step 4: Calculate percentage changes from baseline
    print("\nStep 4: Calculating percentage changes from baseline...")
    
    for category in category_analysis:
        if 'Baseline' in category_analysis[category]:
            baseline_mean = category_analysis[category]['Baseline']['mean']
            
            if baseline_mean and baseline_mean > 0:
                for variant in ['DPO-Synthetic', 'DPO-Hybrid']:
                    if variant in category_analysis[category]:
                        variant_mean = category_analysis[category][variant]['mean']
                        if variant_mean is not None:
                            pct_change = ((variant_mean - baseline_mean) / baseline_mean) * 100
                            category_analysis[category][f'{variant}_change'] = pct_change
    
    # Step 5: Generate output
    print("\nStep 5: Generating output files...")
    
    # Save raw analysis
    with open('analysis/domain_analysis_corrected.json', 'w') as f:
        json.dump({
            'topic_categories': topic_categories,
            'category_counts': category_counts,
            'category_analysis': category_analysis
        }, f, indent=2)
    
    # Generate summary table
    print("\n" + "="*70)
    print("CORRECTED DOMAIN-SPECIFIC PERFORMANCE TABLE")
    print("="*70)
    
    print(f"{'Category':<35} {'Baseline':<20} {'DPO-Synthetic':<25} {'DPO-Hybrid':<25}")
    print(f"{'':35} {'N':>5} {'M':>7} {'SD':>7} {'M':>7} {'Δ%':>8} {'M':>7} {'Δ%':>8}")
    print("-"*110)
    
    for category in sorted(category_analysis.keys()):
        cat_data = category_analysis[category]
        
        # Baseline
        base = cat_data.get('Baseline', {})
        base_n = base.get('n', 0)
        base_m = base.get('mean', 0)
        base_sd = base.get('std', 0)
        
        # DPO-Synthetic
        synth = cat_data.get('DPO-Synthetic', {})
        synth_m = synth.get('mean', 0)
        synth_change = cat_data.get('DPO-Synthetic_change', 0)
        
        # DPO-Hybrid
        hybrid = cat_data.get('DPO-Hybrid', {})
        hybrid_m = hybrid.get('mean', 0)
        hybrid_change = cat_data.get('DPO-Hybrid_change', 0)
        
        # Truncate category name if needed
        cat_name = category[:33] + '..' if len(category) > 35 else category
        
        print(f"{cat_name:<35} {base_n:>5} {base_m:>7.3f} {base_sd:>7.3f} "
              f"{synth_m:>7.3f} {synth_change:>+8.1f} {hybrid_m:>7.3f} {hybrid_change:>+8.1f}")
    
    print("\nOutput files generated:")
    print("  - analysis/domain_analysis_corrected.json")
    
    # Generate LaTeX table snippet
    latex_table = generate_latex_table(category_analysis)
    with open('analysis/domain_table_corrected.tex', 'w') as f:
        f.write(latex_table)
    print("  - analysis/domain_table_corrected.tex")

def generate_latex_table(category_analysis: Dict) -> str:
    """Generate LaTeX table for the corrected domain analysis."""
    
    latex = r"""\begin{table}[H]
\centering
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{\textbf{Category}} & \multicolumn{2}{c}{\textbf{Baseline}} & \multicolumn{2}{c}{\textbf{DPO-Synthetic}} & \multicolumn{2}{c}{\textbf{DPO-Hybrid}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& \textbf{M} & \textbf{N} & \textbf{M} & \textbf{Δ\%} & \textbf{M} & \textbf{Δ\%} \\
\midrule
"""
    
    # Sort categories for consistent ordering
    category_order = [
        "Environmental/Wildlife Conservation",
        "Human Rights/Social Justice",
        "Community/Charity Support",
        "Media/Broadcasting/News"
    ]
    
    for category in category_order:
        if category in category_analysis:
            cat_data = category_analysis[category]
            
            # Get values with defaults
            base = cat_data.get('Baseline', {})
            base_m = base.get('mean', 0) or 0
            base_n = base.get('n', 0)
            
            synth = cat_data.get('DPO-Synthetic', {})
            synth_m = synth.get('mean', 0) or 0
            synth_change = cat_data.get('DPO-Synthetic_change', 0) or 0
            
            hybrid = cat_data.get('DPO-Hybrid', {})
            hybrid_m = hybrid.get('mean', 0) or 0
            hybrid_change = cat_data.get('DPO-Hybrid_change', 0) or 0
            
            # Shorten category name for table
            if "Environmental" in category:
                cat_name = "Environmental/Wildlife"
            elif "Human Rights" in category:
                cat_name = "Human Rights/Social"
            elif "Community" in category:
                cat_name = "Community/Charity"
            elif "Media" in category:
                cat_name = "Media/Broadcasting"
            else:
                cat_name = category
            
            latex += f"{cat_name} & {base_m:.3f} & {base_n} & {synth_m:.3f} & {synth_change:+.1f}\\% & {hybrid_m:.3f} & {hybrid_change:+.1f}\\% \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\caption[Performance by Topic Category]{Performance by Topic Category. Domain-specific analysis presenting mean performance scores (M), sample sizes (N), and percentage changes from baseline (Δ\%) across four topic categories. Findings reveal differential optimization effectiveness depending on content domain.}
\label{tab:category-analysis}
\end{table}
"""
    
    return latex

if __name__ == "__main__":
    main()