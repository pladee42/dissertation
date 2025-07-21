#!/usr/bin/env python3
"""
Analyze Existing Data to Find Traditional vs Reasoning Patterns
This script helps identify which existing runs might contain traditional vs reasoning model data
"""

import json
from pathlib import Path
from datetime import datetime
import re

def analyze_result_file(file_path):
    """Analyze a single complete_results.json file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        analysis = {
            'file': str(file_path),
            'date': file_path.parent.name,
            'total_topics': data.get('total_topics', 0),
            'successful_topics': data.get('successful_topics', 0),
            'models_used': set(),
            'has_placeholder_issues': False,
            'placeholder_examples': [],
            'average_scores': [],
            'checklist_quality': []
        }
        
        # Analyze results
        if 'results' in data:
            for topic_result in data['results']:
                for email in topic_result.get('emails', []):
                    # Track models
                    model_name = email.get('model_name', 'unknown')
                    analysis['models_used'].add(model_name)
                    
                    # Check for placeholder content
                    content = email.get('email_content', '')
                    if any(marker in content.upper() for marker in ['[COPY WRITER', 'PLACEHOLDER', 'YOUR EMAIL HERE']):
                        analysis['has_placeholder_issues'] = True
                        score = email.get('evaluation', {}).get('overall_score', 0)
                        analysis['placeholder_examples'].append({
                            'model': model_name,
                            'score': score,
                            'content_preview': content[:100]
                        })
                    
                    # Collect scores
                    if 'evaluation' in email and 'overall_score' in email['evaluation']:
                        analysis['average_scores'].append(email['evaluation']['overall_score'])
                    
                    # Analyze checklist quality
                    if 'evaluation' in email and 'checklist_scores' in email['evaluation']:
                        checklist = email['evaluation']['checklist_scores']
                        analysis['checklist_quality'].append({
                            'num_items': len(checklist),
                            'avg_desc_length': sum(len(item.get('description', '')) for item in checklist) / len(checklist) if checklist else 0
                        })
        
        # Calculate averages
        if analysis['average_scores']:
            analysis['mean_score'] = sum(analysis['average_scores']) / len(analysis['average_scores'])
        else:
            analysis['mean_score'] = 0
            
        if analysis['checklist_quality']:
            analysis['avg_checklist_items'] = sum(q['num_items'] for q in analysis['checklist_quality']) / len(analysis['checklist_quality'])
            analysis['avg_checklist_desc_length'] = sum(q['avg_desc_length'] for q in analysis['checklist_quality']) / len(analysis['checklist_quality'])
        else:
            analysis['avg_checklist_items'] = 0
            analysis['avg_checklist_desc_length'] = 0
        
        return analysis
        
    except Exception as e:
        return {'file': str(file_path), 'error': str(e)}

def main():
    """Analyze all available complete_results.json files"""
    output_dir = Path("../../output/multi_topic_results")
    
    print("🔍 Analyzing existing complete_results.json files...")
    print("=" * 80)
    
    all_results = []
    potential_traditional = []
    potential_reasoning = []
    
    # Find all complete_results.json files
    for result_file in output_dir.glob("*/complete_results.json"):
        print(f"\n📄 Analyzing: {result_file.parent.name}")
        analysis = analyze_result_file(result_file)
        all_results.append(analysis)
        
        # Try to identify if this used traditional or reasoning models
        if analysis.get('has_placeholder_issues'):
            # Check if high scores were given to placeholder content
            high_score_placeholders = [ex for ex in analysis.get('placeholder_examples', []) 
                                     if ex['score'] > 0.7]
            if high_score_placeholders:
                print("  ⚠️  Found high scores for placeholder content - likely TRADITIONAL models")
                potential_traditional.append(analysis)
            else:
                print("  ✅ Low scores for placeholder content - likely REASONING models")
                potential_reasoning.append(analysis)
        
        # Print summary
        print(f"  - Topics: {analysis.get('total_topics', 0)}")
        print(f"  - Mean score: {analysis.get('mean_score', 0):.2%}")
        print(f"  - Avg checklist items: {analysis.get('avg_checklist_items', 0):.1f}")
        print(f"  - Models used: {', '.join(analysis.get('models_used', []))}")
    
    # Generate report
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total files analyzed: {len(all_results)}")
    print(f"Potential traditional model runs: {len(potential_traditional)}")
    print(f"Potential reasoning model runs: {len(potential_reasoning)}")
    
    # Save detailed analysis
    analysis_file = Path("existing_data_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'potential_traditional': potential_traditional,
            'potential_reasoning': potential_reasoning,
            'analysis_date': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: {analysis_file}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 40)
    
    if not potential_traditional or not potential_reasoning:
        print("❌ Could not clearly identify traditional vs reasoning model runs")
        print("   You will need to run the comparison experiment to get accurate data")
        print("   Run: python run_comparison_experiment.py")
    else:
        print("✅ Found potential data for comparison!")
        print(f"   Traditional runs: {[p['date'] for p in potential_traditional[:3]]}")
        print(f"   Reasoning runs: {[p['date'] for p in potential_reasoning[:3]]}")
        print("\n   Update calculate_improvements.py with these file paths to calculate actual percentages")

if __name__ == "__main__":
    main()