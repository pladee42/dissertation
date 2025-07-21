#!/usr/bin/env python3
"""
Run Controlled Experiment: Traditional vs Reasoning Models
This script runs a controlled experiment to compare traditional and reasoning models
for the Checklist Creator and Judge Agent roles.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.config import get_model_config
from agents.email_agent import EmailAgent
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent

# Test configurations
TEST_TOPICS = [
    {
        "uid": "T0001",
        "description": "Children's Hospital Cancer Treatment Fund - Support breakthrough pediatric cancer research"
    },
    {
        "uid": "T0002", 
        "description": "Homeless Shelter Winter Support Program - Provide warm meals and beds during cold months"
    },
    {
        "uid": "T0003",
        "description": "Food Bank Community Outreach Initiative - Fight hunger in local communities"
    }
]

# Model configurations for testing
TRADITIONAL_MODELS = {
    "checklist": "gpt-3.5-turbo",  # Traditional model
    "judge": "gpt-3.5-turbo"        # Traditional model
}

REASONING_MODELS = {
    "checklist": "openrouter/anthropic/claude-3.5-sonnet:beta",  # Reasoning model
    "judge": "openrouter/google/gemini-2.0-flash-thinking-exp-1219:free"  # Reasoning model
}

# Email generation models to test
EMAIL_MODELS = ["tinyllama-1.1b", "phi-3-mini", "stablelm-2-1.6b"]

def run_single_evaluation(email_content, checklist_model, judge_model, run_id):
    """Run a single evaluation with specified models"""
    try:
        # Create checklist
        checklist_agent = ChecklistAgent(
            backend_type="openrouter",  # Use OpenRouter for these tests
            model_name=checklist_model
        )
        
        checklist_result = checklist_agent.create_checklist(
            email_content=email_content,
            topic="Charity fundraising email"
        )
        
        if not checklist_result['success']:
            return {
                'success': False,
                'error': f"Checklist creation failed: {checklist_result.get('error')}"
            }
        
        # Judge the email
        judge_agent = JudgeAgent(
            backend_type="openrouter",
            model_name=judge_model
        )
        
        judge_result = judge_agent.evaluate_email(
            email_content=email_content,
            checklist=checklist_result['checklist']
        )
        
        if not judge_result['success']:
            return {
                'success': False,
                'error': f"Judge evaluation failed: {judge_result.get('error')}"
            }
        
        return {
            'success': True,
            'checklist': checklist_result['checklist'],
            'evaluation': judge_result['evaluation'],
            'overall_score': judge_result['overall_score'],
            'run_id': run_id,
            'checklist_model': checklist_model,
            'judge_model': judge_model
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'run_id': run_id
        }

def run_experiment():
    """Run the full comparison experiment"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"experiment_results/traditional_vs_reasoning_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'experiment_id': timestamp,
        'traditional_results': [],
        'reasoning_results': [],
        'comparison_metrics': {}
    }
    
    print("🧪 Starting Traditional vs Reasoning Model Experiment")
    print("=" * 60)
    
    # Test with each topic
    for topic in TEST_TOPICS:
        print(f"\n📝 Testing Topic: {topic['uid']} - {topic['description'][:50]}...")
        
        # Generate emails with different models
        for email_model in EMAIL_MODELS:
            print(f"  📧 Generating email with {email_model}...")
            
            # Generate email
            email_agent = EmailAgent(
                backend_type="vllm",
                model_name=email_model
            )
            
            email_result = email_agent.generate_email(
                topic=topic['description'],
                donor_name="Valued Supporter"
            )
            
            if not email_result['success']:
                print(f"    ❌ Email generation failed: {email_result.get('error')}")
                continue
            
            email_content = email_result['email']
            
            # Test with traditional models
            print("    🔄 Testing with TRADITIONAL models...")
            trad_result = run_single_evaluation(
                email_content,
                TRADITIONAL_MODELS['checklist'],
                TRADITIONAL_MODELS['judge'],
                f"{topic['uid']}_{email_model}_traditional"
            )
            
            if trad_result['success']:
                trad_result['topic'] = topic
                trad_result['email_model'] = email_model
                trad_result['email_content'] = email_content
                results['traditional_results'].append(trad_result)
                print(f"      ✅ Score: {trad_result['overall_score']:.2%}")
            else:
                print(f"      ❌ Failed: {trad_result.get('error')}")
            
            # Add delay to avoid rate limits
            time.sleep(2)
            
            # Test with reasoning models
            print("    🔄 Testing with REASONING models...")
            reason_result = run_single_evaluation(
                email_content,
                REASONING_MODELS['checklist'],
                REASONING_MODELS['judge'],
                f"{topic['uid']}_{email_model}_reasoning"
            )
            
            if reason_result['success']:
                reason_result['topic'] = topic
                reason_result['email_model'] = email_model
                reason_result['email_content'] = email_content
                results['reasoning_results'].append(reason_result)
                print(f"      ✅ Score: {reason_result['overall_score']:.2%}")
            else:
                print(f"      ❌ Failed: {reason_result.get('error')}")
            
            time.sleep(2)
    
    # Save results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate summary
    generate_summary(results, output_dir)
    
    return results

def generate_summary(results, output_dir):
    """Generate a summary of the experiment results"""
    summary = []
    summary.append("TRADITIONAL VS REASONING MODELS EXPERIMENT SUMMARY")
    summary.append("=" * 60)
    summary.append(f"Experiment ID: {results['experiment_id']}")
    summary.append(f"Traditional evaluations: {len(results['traditional_results'])}")
    summary.append(f"Reasoning evaluations: {len(results['reasoning_results'])}")
    
    # Calculate average scores
    if results['traditional_results']:
        trad_scores = [r['overall_score'] for r in results['traditional_results'] if r['success']]
        avg_trad = sum(trad_scores) / len(trad_scores) if trad_scores else 0
        summary.append(f"\nTraditional models average score: {avg_trad:.2%}")
    
    if results['reasoning_results']:
        reason_scores = [r['overall_score'] for r in results['reasoning_results'] if r['success']]
        avg_reason = sum(reason_scores) / len(reason_scores) if reason_scores else 0
        summary.append(f"Reasoning models average score: {avg_reason:.2%}")
    
    # Look for problematic cases
    summary.append("\n\nPROBLEMATIC CASES:")
    summary.append("-" * 30)
    
    for trad_result in results['traditional_results']:
        if trad_result['success']:
            email_content = trad_result['email_content']
            score = trad_result['overall_score']
            
            # Check for placeholder content with high scores
            if any(marker in email_content.upper() for marker in ['[COPY WRITER', 'PLACEHOLDER', 'YOUR EMAIL HERE']):
                if score > 0.7:
                    summary.append(f"\n⚠️  Traditional models gave {score:.2%} to placeholder content!")
                    summary.append(f"   Topic: {trad_result['topic']['uid']}")
                    summary.append(f"   Email model: {trad_result['email_model']}")
    
    # Save summary
    summary_file = output_dir / "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"\n📄 Summary saved to: {summary_file}")
    print("\n" + '\n'.join(summary))

if __name__ == "__main__":
    # Note: This requires OpenRouter API key to be set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("   Please set it to run this experiment")
        sys.exit(1)
    
    run_experiment()