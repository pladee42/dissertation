MODELS = {
    'tinyllama-1.1b': {
        'uid': 'M0001',
        'model_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'recommended_for': ['email_generation'],
        'size': 'small',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'vicuna-7b': {
        'uid': 'M0002',
        'model_id': 'lmsys/vicuna-7b-v1.5',
        'recommended_for': ['email_generation'],
        'size': 'medium',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'phi-3-mini': {
        'uid': 'M0003',
        'model_id': 'microsoft/Phi-3-mini-4k-instruct',
        'recommended_for': ['email_generation'],
        'size': 'small',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'llama-3-8b': {
        'uid': 'M0004',
        'model_id': 'casperhansen/llama-3-8b-instruct-awq',
        'recommended_for': ['email_generation'],
        'size': 'medium',
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'stablelm-2-1.6b': {
        'uid': 'M0005',
        'model_id': 'stabilityai/stablelm-2-1_6b-chat',
        'recommended_for': ['email_generation'],
        'size': 'small',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'yi-34b': {
        'uid': 'M0006',
        'model_id': '01-ai/Yi-34B-Chat-4bits',
        'recommended_for': ['checklist_generation', 'judge'],
        'size': 'large',
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'llama-3-70b': {
        'uid': 'M0007',
        'model_id': 'casperhansen/llama-3-70b-instruct-awq',
        'recommended_for': ['checklist_generation', 'judge'],
        'size': 'large',
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'deepseek-r1': {
        'uid': 'M0009',
        'model_id': 'deepseek/deepseek-r1-0528',
        'recommended_for': ['email_generation', 'judge'],
        'size': 'api',
        'backend_type': 'openrouter'
    },
    'gpt-4.1-nano': {
        'uid': 'M0010',
        'model_id': 'openai/gpt-4.1-nano-2025-04-14',
        'recommended_for': ['email_generation', 'judge'],
        'size': 'api',
        'backend_type': 'openrouter'
    },
    'o3-mini': {
        'uid': 'M0011',
        'model_id': 'openai/o3-mini',
        'recommended_for': ['judge'],
        'size': 'api',
        'backend_type': 'openrouter'
    },
    # DPO Fine-tuned Models
    # 'tinyllama-1.1b-dpo': {
        # 'uid': 'M0012',
        # 'model_id': 'pladee42/tinyllama-1.1b-dpo',
        # 'recommended_for': ['email_generation'],
        # 'size': 'small',
        # 'quantization': 'experts_int8',
        # 'dtype': 'bfloat16',
        # 'base_model': 'tinyllama-1.1b',
        # 'is_dpo': True
    # },
    # 'vicuna-7b-dpo': {
        # 'uid': 'M0013',
        # 'model_id': 'pladee42/vicuna-7b-dpo',
        # 'recommended_for': ['email_generation'],
        # 'size': 'medium',
        # 'quantization': 'experts_int8',
        # 'dtype': 'bfloat16',
        # 'base_model': 'vicuna-7b',
        # 'is_dpo': True
    # },
    # 'phi-3-mini-dpo': {
        # 'uid': 'M0014',
        # 'model_id': 'pladee42/phi-3-mini-dpo',
        # 'recommended_for': ['email_generation'],
        # 'size': 'small',
        # 'quantization': 'experts_int8',
        # 'dtype': 'bfloat16',
        # 'base_model': 'phi-3-mini',
        # 'is_dpo': True
    # },
    # 'llama-3-8b-dpo': {
        # 'uid': 'M0015',
        # 'model_id': 'pladee42/llama-3-8b-dpo',
        # 'recommended_for': ['email_generation'],
        # 'size': 'medium',
        # 'quantization': 'awq',
        # 'dtype': 'float16',
        # 'base_model': 'llama-3-8b',
        # 'is_dpo': True
    # },
    # 'stablelm-2-1.6b-dpo': {
        # 'uid': 'M0016',
        # 'model_id': 'pladee42/stablelm-2-1.6b-dpo',
        # 'recommended_for': ['email_generation'],
        # 'size': 'small',
        # 'quantization': 'experts_int8',
        # 'dtype': 'bfloat16',
        # 'base_model': 'stablelm-2-1.6b',
        # 'is_dpo': True
    # }
}

# Checklist mode constants
CHECKLIST_MODES = {
    'ENHANCED': 'enhanced',
    'EXTRACT_ONLY': 'extract_only', 
    'PREPROCESS': 'preprocess'
}

# Simple settings
SETTINGS = {
    # Basic settings
    'output_dir': './output',
    'max_retries': 3,
    'temperature': 0.7,
    'max_tokens': 4096,
    
    # Task-specific token limits
    'checklist_max_tokens': 8192,
    'judge_max_tokens': 8192,
    'email_max_tokens': 6144,
    
    # Agent-specific sampling parameters
    'email_temperature': 0.5,      # Consistent but creative emails
    'checklist_temperature': 0.2,  # Structured, focused checklists  
    'judge_temperature': 0.2,      # Deterministic evaluations
    'email_top_p': 0.85,           # Focused vocabulary for emails
    'checklist_top_p': 0.7,        # Narrow for JSON structure
    'judge_top_p': 0.6,            # Very focused for scoring
    
    # Memory settings
    'memory_strategy': 'conservative',  # 'conservative' or 'performance'
    'max_concurrent_models': 2,
    'enable_gpu_cleanup': True,
    
    # Model defaults
    'default_dtype': 'bfloat16',
    'default_quantization': 'experts_int8',
    
    # Paths
    'prompt_dir': './config/prompts',
    'log_dir': './log',
    'models_cache_dir': './downloaded_models',
    
    # Backend settings
    'backend_type': 'vllm',
    
    # Debug settings
    'debug_save_responses': True,  # Set to True to save raw responses for debugging
    
    # vLLM settings
    'vllm_max_parallel': 4,
    'vllm_gpu_memory_utilization': 0.7,
    
    # Consistency sampling settings
    'consistency_samples': 3,
    'consistency_enabled': True,
    'consistency_timeout': 120,
    
    # Checklist mode settings
    'default_checklist_mode': CHECKLIST_MODES['ENHANCED']
}

# Memory requirements by model size (with 30% GPU utilization)
MEMORY_REQUIREMENTS = {
    'small': {'min_gb': 2, 'recommended_gb': 6, 'vram_gb': 14},
    'medium': {'min_gb': 8, 'recommended_gb': 12, 'vram_gb': 16}, 
    'large': {'min_gb': 24, 'recommended_gb': 32, 'vram_gb': 40}
}

def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model"""
    return MODELS.get(model_name, {})

def get_setting(key: str, default=None):
    """Get a setting value"""
    return SETTINGS.get(key, default)

def get_memory_requirement(model_name: str) -> dict:
    """Get memory requirements for a model"""
    model_config = get_model_config(model_name)
    size = model_config.get('size', 'medium')
    return MEMORY_REQUIREMENTS.get(size, MEMORY_REQUIREMENTS['medium'])

def get_model_by_uid(uid: str) -> dict:
    """Get model configuration by UID"""
    for model_name, config in MODELS.items():
        if config.get('uid') == uid:
            return {model_name: config}
    return {}

def get_uid_by_model_name(model_name: str) -> str:
    """Get UID by model name"""
    model_config = MODELS.get(model_name, {})
    return model_config.get('uid', '')

def get_model_name_by_uid(uid: str) -> str:
    """Get model name by UID"""
    for model_name, config in MODELS.items():
        if config.get('uid') == uid:
            return model_name
    return ''

def list_models_by_size(size: str, include_dpo: bool = True) -> list:
    """Get list of model names by size category"""
    models = []
    for model_name, config in MODELS.items():
        if config.get('size') == size:
            # Filter DPO models if requested
            if not include_dpo and config.get('is_dpo', False):
                continue
            models.append(model_name)
    return models

def list_models_by_size_group(size_group: str) -> list:
    """Get models by size group with DPO support"""
    if size_group == 'small':
        return list_models_by_size('small', include_dpo=True)
    elif size_group == 'medium':
        return list_models_by_size('medium', include_dpo=True)
    elif size_group == 'large':
        return list_models_by_size('large', include_dpo=True)
    elif size_group == 'small-dpo':
        return [m for m in list_models_by_size('small') if is_dpo_model(m)]
    elif size_group == 'medium-dpo':
        return [m for m in list_models_by_size('medium') if is_dpo_model(m)]
    elif size_group == 'base-only':
        models = []
        for size in ['small', 'medium']:
            models.extend([m for m in list_models_by_size(size) if not is_dpo_model(m)])
        return models
    elif size_group == 'all-dpo':
        return list_dpo_models()
    else:
        return []

def list_dpo_models() -> list:
    """Get list of DPO model names"""
    dpo_models = []
    for model_name, config in MODELS.items():
        if config.get('is_dpo', False):
            dpo_models.append(model_name)
    return dpo_models

def get_base_model_for_dpo(dpo_model_name: str) -> str:
    """Get base model name for a DPO model"""
    config = get_model_config(dpo_model_name)
    return config.get('base_model', '')

def is_dpo_model(model_name: str) -> bool:
    """Check if a model is a DPO fine-tuned model"""
    config = get_model_config(model_name)
    return config.get('is_dpo', False)

def get_model_pairs() -> list:
    """Get list of (base_model, dpo_model) pairs for comparison"""
    pairs = []
    dpo_models = list_dpo_models()
    
    for dpo_model in dpo_models:
        base_model = get_base_model_for_dpo(dpo_model)
        if base_model and base_model in MODELS:
            pairs.append((base_model, dpo_model))
    
    return pairs

def list_available_comparisons() -> dict:
    """List all available base vs DPO comparisons"""
    pairs = get_model_pairs()
    comparisons = {}
    
    for base_model, dpo_model in pairs:
        base_config = get_model_config(base_model)
        dpo_config = get_model_config(dpo_model)
        
        comparisons[f"{base_model}_vs_{dpo_model}"] = {
            'base': base_model,
            'dpo': dpo_model,
            'size': base_config.get('size', 'unknown'),
            'base_uid': base_config.get('uid', ''),
            'dpo_uid': dpo_config.get('uid', '')
        }
    
    return comparisons

def get_comparison_command(base_model: str, dpo_model: str) -> str:
    """Generate runner command for comparing base and DPO models"""
    return f"python -m runner --email_models {base_model} {dpo_model}"

# Legacy compatibility (for existing imports)
MODELS_CONFIG = MODELS