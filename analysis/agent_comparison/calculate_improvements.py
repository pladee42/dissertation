#!/usr/bin/env python3
"""
Calculate Actual Improvement Percentages: Traditional vs Reasoning Models
This script analyzes complete_results.json files to compute the real improvement metrics
mentioned in the methodology section.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
from collections import defaultdict

class AgentComparisonAnalyzer:
    def __init__(self):
        self.traditional_results = []
        self.reasoning_results = []
        
    def load_results(self, results_path: str, model_type: str = "reasoning"):
        """Load results from complete_results.json file"""
        with open(results_path, 'r') as f:
            data = json.load(f)
            
        if model_type == "traditional":
            self.traditional_results.append(data)
        else:
            self.reasoning_results.append(data)
            
    def calculate_evaluation_consistency(self, results: List[Dict]) -> Dict:
        """Calculate consistency metrics - how stable are the scores?"""
        scores_by_topic = defaultdict(list)
        
        # Group scores by topic across different runs
        for result_set in results:
            if 'results' in result_set:
                for topic_result in result_set['results']:
                    topic_id = topic_result.get('topic_id', 'unknown')
                    for email in topic_result.get('emails', []):
                        if 'evaluation' in email and 'overall_score' in email['evaluation']:
                            scores_by_topic[topic_id].append(email['evaluation']['overall_score'])
        
        # Calculate variance for each topic
        variances = []
        for topic_id, scores in scores_by_topic.items():
            if len(scores) > 1:
                variance = statistics.variance(scores)
                variances.append(variance)
        
        return {
            'mean_variance': np.mean(variances) if variances else 0,
            'std_variance': np.std(variances) if variances else 0,
            'topics_analyzed': len(variances)
        }
    
    def calculate_analytical_depth(self, results: List[Dict]) -> Dict:
        """Calculate depth of analysis - checklist quality and detail"""
        checklist_metrics = []
        
        for result_set in results:
            if 'results' in result_set:
                for topic_result in result_set['results']:
                    for email in topic_result.get('emails', []):
                        if 'evaluation' in email and 'checklist_scores' in email['evaluation']:
                            checklist = email['evaluation']['checklist_scores']
                            
                            # Metrics for checklist depth
                            num_criteria = len(checklist)
                            avg_description_length = np.mean([len(item.get('description', '')) for item in checklist])
                            confidence_scores = [item.get('confidence', 0) for item in checklist]
                            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
                            
                            checklist_metrics.append({
                                'num_criteria': num_criteria,
                                'avg_description_length': avg_description_length,
                                'avg_confidence': avg_confidence
                            })
        
        if not checklist_metrics:
            return {'error': 'No checklist data found'}
            
        return {
            'avg_criteria_count': np.mean([m['num_criteria'] for m in checklist_metrics]),
            'avg_description_length': np.mean([m['avg_description_length'] for m in checklist_metrics]),
            'avg_confidence': np.mean([m['avg_confidence'] for m in checklist_metrics]),
            'total_checklists': len(checklist_metrics)
        }
    
    def calculate_bias_mitigation(self, results: List[Dict]) -> Dict:
        """Calculate bias mitigation - false positive rate for defective content"""
        false_positives = 0
        total_defective = 0
        high_scores_for_bad_content = []
        
        for result_set in results:
            if 'results' in result_set:
                for topic_result in result_set['results']:
                    for email in topic_result.get('emails', []):
                        content = email.get('email_content', '')
                        score = email.get('evaluation', {}).get('overall_score', 0)
                        
                        # Detect defective content patterns
                        is_defective = any([
                            'COPY WRITER\'S EMAIL' in content.upper(),
                            '[PLACEHOLDER' in content.upper(),
                            len(content) < 100,  # Too short
                            content.count('[') > 5,  # Too many placeholders
                            'YOUR EMAIL HERE' in content.upper()
                        ])
                        
                        if is_defective:
                            total_defective += 1
                            if score > 0.7:  # High score for bad content
                                false_positives += 1
                                high_scores_for_bad_content.append({
                                    'score': score,
                                    'content_preview': content[:100]
                                })
        
        false_positive_rate = (false_positives / total_defective * 100) if total_defective > 0 else 0
        
        return {
            'false_positive_rate': false_positive_rate,
            'total_defective_found': total_defective,
            'false_positives': false_positives,
            'examples': high_scores_for_bad_content[:3]  # First 3 examples
        }
    
    def calculate_improvements(self) -> Dict:
        """Calculate all improvement percentages"""
        if not self.traditional_results or not self.reasoning_results:
            return {'error': 'Need both traditional and reasoning results to compare'}
        
        # Calculate metrics for both
        trad_consistency = self.calculate_evaluation_consistency(self.traditional_results)
        reason_consistency = self.calculate_evaluation_consistency(self.reasoning_results)
        
        trad_depth = self.calculate_analytical_depth(self.traditional_results)
        reason_depth = self.calculate_analytical_depth(self.reasoning_results)
        
        trad_bias = self.calculate_bias_mitigation(self.traditional_results)
        reason_bias = self.calculate_bias_mitigation(self.reasoning_results)
        
        # Calculate improvements
        improvements = {}
        
        # Evaluation Consistency (lower variance is better)
        if trad_consistency['mean_variance'] > 0:
            consistency_improvement = ((trad_consistency['mean_variance'] - reason_consistency['mean_variance']) / 
                                     trad_consistency['mean_variance'] * 100)
            improvements['evaluation_consistency'] = {
                'percentage': round(consistency_improvement, 1),
                'traditional_variance': trad_consistency['mean_variance'],
                'reasoning_variance': reason_consistency['mean_variance']
            }
        
        # Analytical Depth (higher is better)
        if trad_depth.get('avg_criteria_count', 0) > 0:
            depth_improvement = ((reason_depth['avg_criteria_count'] - trad_depth['avg_criteria_count']) / 
                               trad_depth['avg_criteria_count'] * 100)
            improvements['analytical_depth'] = {
                'percentage': round(depth_improvement, 1),
                'traditional_avg_criteria': trad_depth['avg_criteria_count'],
                'reasoning_avg_criteria': reason_depth['avg_criteria_count']
            }
        
        # Bias Mitigation (lower false positive rate is better)
        if trad_bias['false_positive_rate'] > 0:
            bias_improvement = ((trad_bias['false_positive_rate'] - reason_bias['false_positive_rate']) / 
                              trad_bias['false_positive_rate'] * 100)
            improvements['bias_mitigation'] = {
                'percentage': round(bias_improvement, 1),
                'traditional_fp_rate': trad_bias['false_positive_rate'],
                'reasoning_fp_rate': reason_bias['false_positive_rate']
            }
        
        return improvements

def main():
    """Main analysis function"""
    analyzer = AgentComparisonAnalyzer()
    
    # Find result files
    output_dir = Path("../../output/multi_topic_results")
    
    print("🔍 Looking for complete_results.json files...")
    print("=" * 60)
    
    # You'll need to specify which results are traditional vs reasoning
    # For now, this is a template that shows how to use the analyzer
    
    # Example usage (update paths based on your actual data):
    # analyzer.load_results("path/to/traditional_results.json", "traditional")
    # analyzer.load_results("path/to/reasoning_results.json", "reasoning")
    
    # Calculate improvements
    # improvements = analyzer.calculate_improvements()
    
    # Print results
    print("\n📊 IMPROVEMENT ANALYSIS RESULTS")
    print("=" * 60)
    
    # Placeholder for when you have actual data
    print("\n⚠️  To calculate actual improvements, you need to:")
    print("1. Identify which complete_results.json files used traditional models")
    print("2. Identify which used reasoning models")
    print("3. Update this script with the correct file paths")
    print("\nAlternatively, you may need to run new experiments comparing both model types")
    
    # Save results
    results_file = Path("improvement_percentages.json")
    # with open(results_file, 'w') as f:
    #     json.dump(improvements, f, indent=2)
    
    print(f"\n💾 Results will be saved to: {results_file}")

if __name__ == "__main__":
    main()