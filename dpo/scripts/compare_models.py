#!/usr/bin/env python3
"""
Simple Comparison Runner - Base vs DPO Models
Generate side-by-side outputs for easy comparison
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.config import (
    get_model_pairs, list_available_comparisons, 
    get_model_config, get_comparison_command
)
from models.orchestrator import ModelOrchestrator
from config.topic_manager import get_topic_by_uid

def create_comparison_output_dir():
    """Create output directory for comparisons"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"../outputs/comparisons/comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_single_model(model_name, topic, output_dir):
    """Run a single model and save output"""
    print(f"🔄 Running {model_name}...")
    
    try:
        # Create orchestrator with single model
        orchestrator = ModelOrchestrator(
            email_models=[model_name],
            checklist_model="gpt-4.1-nano",  # Use fast model for comparison
            judge_model="gemini-2.5-flash"   # Use fast model for comparison
        )
        
        # Run the model
        results = orchestrator.run_topic(topic)
        
        # Save raw output
        model_output_file = output_dir / f"{model_name}_output.txt"
        with open(model_output_file, 'w') as f:
            f.write(f"Model: {model_name}\\n")
            f.write(f"Topic: {topic}\\n")
            f.write(f"Timestamp: {datetime.now()}\\n")
            f.write("="*50 + "\\n\\n")
            
            if results and model_name in results.get('emails', {}):
                email_content = results['emails'][model_name]
                f.write("GENERATED EMAIL:\\n")
                f.write("-"*30 + "\\n")
                f.write(email_content)
                f.write("\\n\\n")
                
                # Add basic metrics
                f.write("BASIC METRICS:\\n")
                f.write("-"*30 + "\\n")
                f.write(f"Length (characters): {len(email_content)}\\n")
                f.write(f"Length (words): {len(email_content.split())}\\n")
                f.write(f"Lines: {len(email_content.split(chr(10)))}\\n")
            else:
                f.write("❌ No output generated\\n")
        
        print(f"✅ {model_name} completed: {model_output_file}")
        return model_output_file, results
        
    except Exception as e:
        error_file = output_dir / f"{model_name}_error.txt"
        with open(error_file, 'w') as f:
            f.write(f"❌ Error running {model_name}: {e}\\n")
        print(f"❌ {model_name} failed: {e}")
        return error_file, None

def create_side_by_side_comparison(base_output, dpo_output, base_model, dpo_model, output_dir):
    """Create side-by-side comparison file"""
    comparison_file = output_dir / f"{base_model}_vs_{dpo_model}_comparison.txt"
    
    # Read outputs
    try:
        with open(base_output, 'r') as f:
            base_content = f.read()
    except:
        base_content = "❌ Base model output not available"
    
    try:
        with open(dpo_output, 'r') as f:
            dpo_content = f.read()
    except:
        dpo_content = "❌ DPO model output not available"
    
    # Create comparison
    with open(comparison_file, 'w') as f:
        f.write(f"SIDE-BY-SIDE COMPARISON\\n")
        f.write(f"{'='*60}\\n\\n")
        f.write(f"BASE MODEL: {base_model}\\n")
        f.write(f"DPO MODEL:  {dpo_model}\\n")
        f.write(f"Date: {datetime.now()}\\n\\n")
        
        f.write(f"{'BASE MODEL OUTPUT':<30} | {'DPO MODEL OUTPUT':<30}\\n")
        f.write(f"{'-'*30} | {'-'*30}\\n")
        
        # Split content into lines for side-by-side
        base_lines = base_content.split('\\n')
        dpo_lines = dpo_content.split('\\n')
        max_lines = max(len(base_lines), len(dpo_lines))
        
        for i in range(max_lines):
            base_line = base_lines[i] if i < len(base_lines) else ""
            dpo_line = dpo_lines[i] if i < len(dpo_lines) else ""
            
            # Truncate long lines for readability
            base_line = base_line[:50] + "..." if len(base_line) > 50 else base_line
            dpo_line = dpo_line[:50] + "..." if len(dpo_line) > 50 else dpo_line
            
            f.write(f"{base_line:<53} | {dpo_line}\\n")
    
    print(f"📊 Comparison created: {comparison_file}")
    return comparison_file

def run_comparison(base_model, dpo_model, topic="Default charity email topic"):
    """Run a complete base vs DPO comparison"""
    print(f"🆚 Starting comparison: {base_model} vs {dpo_model}")
    print(f"📝 Topic: {topic}")
    
    # Create output directory
    output_dir = create_comparison_output_dir()
    print(f"📁 Output directory: {output_dir}")
    
    # Run both models
    base_output, base_results = run_single_model(base_model, topic, output_dir)
    dpo_output, dpo_results = run_single_model(dpo_model, topic, output_dir)
    
    # Create side-by-side comparison
    comparison_file = create_side_by_side_comparison(
        base_output, dpo_output, base_model, dpo_model, output_dir
    )
    
    # Create summary
    summary_file = output_dir / "comparison_summary.json"
    summary = {
        "comparison_id": output_dir.name,
        "timestamp": datetime.now().isoformat(),
        "base_model": base_model,
        "dpo_model": dpo_model,
        "topic": topic,
        "files": {
            "base_output": str(base_output.name),
            "dpo_output": str(dpo_output.name),
            "comparison": str(comparison_file.name),
            "summary": str(summary_file.name)
        },
        "success": {
            "base": base_results is not None,
            "dpo": dpo_results is not None
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n🎉 Comparison complete!")
    print(f"📁 All files saved to: {output_dir}")
    print(f"📊 View comparison: {comparison_file}")
    
    return output_dir

def list_available_pairs():
    """List all available comparison pairs"""
    print("🔍 Available Base vs DPO Comparisons:")
    
    pairs = get_model_pairs()
    if not pairs:
        print("❌ No DPO model pairs found!")
        print("💡 Make sure you have:")
        print("   1. Trained DPO models")
        print("   2. Updated HF username in config")
        print("   3. Uploaded models to HF Hub")
        return
    
    print(f"✅ Found {len(pairs)} comparison pairs:\\n")
    
    for i, (base_model, dpo_model) in enumerate(pairs, 1):
        base_config = get_model_config(base_model)
        dpo_config = get_model_config(dpo_model)
        
        print(f"{i}. {base_model} vs {dpo_model}")
        print(f"   Size: {base_config.get('size', 'unknown')}")
        print(f"   UIDs: {base_config.get('uid', 'N/A')} → {dpo_config.get('uid', 'N/A')}")
        print(f"   Command: {get_comparison_command(base_model, dpo_model)}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Compare base vs DPO models')
    parser.add_argument('--base', help='Base model name')
    parser.add_argument('--dpo', help='DPO model name')
    parser.add_argument('--pair', type=int, help='Use comparison pair by number (see --list)')
    parser.add_argument('--topic', default="Children's Hospital Cancer Treatment Fund", 
                       help='Topic for email generation')
    parser.add_argument('--topic-uid', help='Topic UID (e.g., T0001)')
    parser.add_argument('--list', action='store_true', help='List available comparison pairs')
    parser.add_argument('--all', action='store_true', help='Run all available comparisons')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_pairs()
        return
    
    # Handle topic selection
    topic = args.topic
    if args.topic_uid:
        topic_data = get_topic_by_uid(args.topic_uid)
        if topic_data:
            topic = topic_data['description']
            print(f"🎯 Using topic {args.topic_uid}: {topic}")
    
    pairs = get_model_pairs()
    
    if args.all:
        print(f"🚀 Running all {len(pairs)} comparisons...")
        for i, (base_model, dpo_model) in enumerate(pairs, 1):
            print(f"\\n--- Comparison {i}/{len(pairs)} ---")
            run_comparison(base_model, dpo_model, topic)
        return
    
    if args.pair:
        if 1 <= args.pair <= len(pairs):
            base_model, dpo_model = pairs[args.pair - 1]
            print(f"🎯 Using pair {args.pair}: {base_model} vs {dpo_model}")
        else:
            print(f"❌ Invalid pair number. Use --list to see available pairs.")
            return
    elif args.base and args.dpo:
        base_model, dpo_model = args.base, args.dpo
    else:
        print("❌ Please specify comparison models:")
        print("   --base <model> --dpo <model>  (explicit models)")
        print("   --pair <number>               (use pair from --list)")
        print("   --all                         (run all comparisons)")
        print("   --list                        (show available pairs)")
        return
    
    # Validate models exist
    from config.config import get_model_config
    if not get_model_config(base_model):
        print(f"❌ Base model not found: {base_model}")
        return
    if not get_model_config(dpo_model):
        print(f"❌ DPO model not found: {dpo_model}")
        return
    
    # Run the comparison
    run_comparison(base_model, dpo_model, topic)

if __name__ == "__main__":
    main()