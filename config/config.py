MODELS = {
    'deepseek-r1-1.5b': {
        'uid': 'M0001',
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'recommended_for': ['email_generation'],
        'size': 'small',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'deepseek-r1-8b': {
        'uid': 'M0002',
        'model_id': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'recommended_for': ['email_generation'],
        'size': 'medium',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'llama-3-3b': {
        'uid': 'M0003',
        'model_id': 'unsloth/Llama-3.2-3B-Instruct',
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
    'gemma-3-4b': {
        'uid': 'M0005',
        'model_id': 'gaunernst/gemma-3-4b-it-qat-autoawq',
        'recommended_for': ['email_generation'],
        'size': 'small',
        'quantization': 'awq',
        'dtype': 'bfloat16'
    },
    'qwen-3-8b': {
        'uid': 'M0006',
        'model_id': 'Qwen/Qwen3-8B-AWQ',
        'recommended_for': ['email_generation'],
        'size': 'medium',
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'deepseek-r1-70b': {
        'uid': 'M0007',
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'recommended_for': ['checklist_generation', 'judge'],
        'size': 'large',
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'llama-4-109b': {
        'uid': 'M0008',
        'model_id': 'kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ',
        'recommended_for': ['checklist_generation', 'judge'],
        'size': 'large',
        'quantization': 'auto',
        'dtype': 'float16'
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
    'prompt_dir': './config/prompts',
    'log_dir': './log',
    'models_cache_dir': './downloaded_models',
    
    # Backend settings
    'backend_type': 'vllm',
    
    # vLLM settings
    'vllm_max_parallel': 4,
    'vllm_gpu_memory_utilization': 0.2
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

def list_models_by_size(size: str) -> list:
    """Get list of model names by size category"""
    models = []
    for model_name, config in MODELS.items():
        if config.get('size') == size:
            models.append(model_name)
    return models

# Legacy compatibility (for existing imports)
MODELS_CONFIG = MODELS