#!/usr/bin/env python3
"""
Quick script to check model availability
"""

import sys
from pathlib import Path

# Add the scripts directory to the path so we can import model_manager
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import ModelManager

def main():
    # Standard 5 models used in the main pipeline
    models_to_check = ['tinyllama', 'vicuna', 'phi3', 'llama3', 'stablelm']
    
    print("🔍 Checking availability of 5 standard models...")
    print(f"Models: {models_to_check}")
    print()
    
    # Initialize model manager
    manager = ModelManager("../downloaded_models")
    
    # Show cache stats first
    stats = manager.get_cache_stats()
    print(f"📊 Cache Statistics:")
    print(f"  Total size: {stats['total_size_gb']} GB")
    print(f"  Model count: {stats['model_count']}")
    print()
    
    # Check availability
    availability = manager.check_model_availability(models_to_check)
    
    available_count = 0
    missing_count = 0
    
    print("📋 Model Status:")
    for model_key, info in availability.items():
        if info['status'] == 'cached':
            print(f"  ✅ {model_key}: Available ({info['size']})")
            available_count += 1
        elif info['status'] == 'missing':
            print(f"  ❌ {model_key}: Missing ({info['size']}) - {info['model_id']}")
            missing_count += 1
        else:
            print(f"  ⚠️ {model_key}: Error - {info.get('error', 'Unknown')}")
            missing_count += 1
    
    print()
    print(f"Summary: {available_count} available, {missing_count} missing")
    
    if missing_count > 0:
        print()
        print("To download missing models, run:")
        missing_models = [k for k, v in availability.items() if v['status'] != 'cached']
        print(f"  python scripts/model_manager.py --download {' '.join(missing_models)}")

if __name__ == "__main__":
    main()