"""
Simplified Configuration

This module provides all configuration in one place with:
- Basic model configurations
- Simple settings
- Minimal complexity
"""

# Simple model configurations
MODELS = {
    'deepseek-r1-1.5b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'recommended_for': 'email_generation',
        'size': 'small'
    },
    'deepseek-r1-8b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'recommended_for': 'email_generation',
        'size': 'medium'
    },
    'llama-3-3b': {
        'model_id': 'unsloth/Llama-3.2-3B-Instruct',
        'recommended_for': 'email_generation',
        'size': 'small'
    },
    'llama-3-8b': {
        'model_id': 'casperhansen/llama-3-8b-instruct-awq',
        'recommended_for': 'email_generation',
        'size': 'medium'
    },
    'gemma-3-4b': {
        'model_id': 'gaunernst/gemma-3-4b-it-qat-autoawq',
        'recommended_for': 'email_generation',
        'size': 'small'
    },
    'qwen-3-8b': {
        'model_id': 'Qwen/Qwen3-8B-AWQ',
        'recommended_for': 'email_generation',
        'size': 'medium'
    },
    'deepseek-r1-70b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'recommended_for': 'checklist_generation',
        'size': 'large'
    },
    'llama-4-109b': {
        'model_id': 'kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ',
        'recommended_for': 'judge',
        'size': 'large'
    }
}

# Simple settings
SETTINGS = {
    # Basic settings
    'output_dir': './output',
    'max_retries': 3,
    'temperature': 0.7,
    'max_tokens': 2048,
    
    # Memory settings
    'memory_strategy': 'conservative',  # 'conservative' or 'performance'
    'max_concurrent_models': 2,
    'enable_gpu_cleanup': True,
    
    # Model defaults
    'default_dtype': 'bfloat16',
    'default_quantization': 'experts_int8',
    
    # Paths
    'prompt_dir': './prompts',
    'log_dir': './log',
    
    # SGLang settings
    'sglang_server_url': 'http://localhost:30000',
    'sglang_timeout': 60,
    'sglang_health_check': True
}

# Memory requirements by model size
MEMORY_REQUIREMENTS = {
    'small': {'min_gb': 2, 'recommended_gb': 4},
    'medium': {'min_gb': 8, 'recommended_gb': 12},
    'large': {'min_gb': 24, 'recommended_gb': 32}
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

# Legacy compatibility (for existing imports)
MODELS_CONFIG = MODELS