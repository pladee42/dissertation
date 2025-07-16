#!/usr/bin/env python3
"""
Config verification and completion script for DPO models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.config import (
    MODELS, list_dpo_models, list_models_by_size_group,
    get_model_config, is_dpo_model, get_base_model_for_dpo
)

def verify_dpo_config():
    """Verify all DPO model configurations are complete and valid"""
    print("🔍 Verifying DPO Model Configuration...")
    
    dpo_models = list_dpo_models()
    print(f"📊 Found {len(dpo_models)} DPO models in config")
    
    issues = []
    
    for model_name in dpo_models:
        print(f"\\n🔬 Checking {model_name}:")
        config = get_model_config(model_name)
        
        # Check required fields
        required_fields = ['uid', 'model_id', 'recommended_for', 'size', 'base_model', 'is_dpo']
        for field in required_fields:
            if field not in config:
                issues.append(f"{model_name}: Missing field '{field}'")
                print(f"  ❌ Missing: {field}")
            else:
                print(f"  ✅ {field}: {config[field]}")
        
        # Check model_id format
        model_id = config.get('model_id', '')
        if 'your-username' in model_id:
            issues.append(f"{model_name}: Update model_id with real HF username")
            print(f"  ⚠️  Need to update username in model_id")
        
        # Check base model exists
        base_model = config.get('base_model', '')
        if base_model and base_model not in MODELS:
            issues.append(f"{model_name}: Base model '{base_model}' not found in config")
            print(f"  ❌ Base model not found: {base_model}")
        else:
            print(f"  ✅ Base model exists: {base_model}")
    
    if issues:
        print(f"\\n❌ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\\n✅ All {len(dpo_models)} DPO models are properly configured!")
        return True

def test_size_groups():
    """Test the new size group functionality"""
    print("\\n🧪 Testing Size Group Functionality...")
    
    size_groups = ['small', 'medium', 'small-dpo', 'medium-dpo', 'base-only', 'all-dpo']
    
    for group in size_groups:
        models = list_models_by_size_group(group)
        print(f"\\n📋 Size group '{group}': {len(models)} models")
        for model in models:
            is_dpo = is_dpo_model(model)
            base = get_base_model_for_dpo(model) if is_dpo else 'N/A'
            print(f"  - {model} {'(DPO)' if is_dpo else '(BASE)'} {f'← {base}' if base != 'N/A' else ''}")

def suggest_runner_commands():
    """Suggest useful runner commands for testing"""
    print("\\n🚀 Suggested Runner Commands for Testing:")
    
    # Get some examples
    small_dpo = list_models_by_size_group('small-dpo')
    medium_dpo = list_models_by_size_group('medium-dpo')
    all_dpo = list_models_by_size_group('all-dpo')
    
    print("\\n📝 Single DPO model:")
    if small_dpo:
        print(f"  python -m runner --email_models {small_dpo[0]} --topic 'Test DPO'")
    
    print("\\n📝 Compare base vs DPO:")
    if small_dpo:
        base_model = get_base_model_for_dpo(small_dpo[0])
        if base_model:
            print(f"  python -m runner --email_models {base_model} {small_dpo[0]} --topic 'Compare models'")
    
    print("\\n📝 All small DPO models:")
    if len(small_dpo) > 1:
        models_str = ' '.join(small_dpo[:3])  # Limit to first 3
        print(f"  python -m runner --email_models {models_str}")
    
    print("\\n📝 Multi-topic with DPO:")
    if all_dpo:
        print(f"  python -m multi_topic_runner --random_topics 3 --email_models {all_dpo[0]}")
    
    print("\\n📝 Size group usage (if implementing in runner):")
    print(f"  python -m runner --email_generation small-dpo")
    print(f"  python -m runner --email_generation all-dpo")

def check_model_availability():
    """Check if DPO models are available (placeholder for actual availability check)"""
    print("\\n🌐 Model Availability Check:")
    print("(This would check if models exist on HF Hub - skipped for now)")
    
    dpo_models = list_dpo_models()
    for model_name in dpo_models:
        config = get_model_config(model_name)
        model_id = config.get('model_id', '')
        if 'your-username' not in model_id:
            print(f"  📡 Would check: {model_id}")
        else:
            print(f"  ⏳ Pending upload: {model_name}")

def main():
    print("🔧 DPO Configuration Verification Tool")
    print("=" * 50)
    
    # Run all checks
    config_ok = verify_dpo_config()
    test_size_groups()
    check_model_availability()
    suggest_runner_commands()
    
    print("\\n" + "=" * 50)
    if config_ok:
        print("🎉 Configuration verification complete!")
        print("\\n📋 Summary:")
        print(f"  - {len(list_dpo_models())} DPO models configured")
        print(f"  - {len(list_models_by_size_group('small-dpo'))} small DPO models")
        print(f"  - {len(list_models_by_size_group('medium-dpo'))} medium DPO models")
        print("\\n🚀 Ready for Stage 3!")
    else:
        print("❌ Configuration issues found - please fix before proceeding")
        sys.exit(1)

if __name__ == "__main__":
    main()