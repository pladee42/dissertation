MODELS_CONFIG = {
    'deepseek-r1-1.5b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'recommended_for': ['email_generation'],
        'quantization': 'experts_int8',
        'dtype': 'bfloat16',
        'sglang_config': {
            'tp_size': 1,
            'max_running_requests': 128,
            'context_length': 4096,
            'mem_fraction_static': 0.7,
            'attention_backend': 'flashinfer'
        }
    },
    'deepseek-r1-8b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'recommended_for': ['email_generation'],
        'quantization': 'experts_int8',
        'dtype': 'bfloat16',
        'sglang_config': {
            'tp_size': 2,
            'max_running_requests': 64,
            'context_length': 4096,
            'mem_fraction_static': 0.8,
            'attention_backend': 'flashinfer'
        }
    },
    'llama-3-3b':{
        'model_id': 'unsloth/Llama-3.2-3B-Instruct',
        'recommended_for': ['email_generation'],
        'quantization': 'experts_int8',
        'dtype': 'bfloat16'
    },
    'llama-3-8b':{
        'model_id': 'casperhansen/llama-3-8b-instruct-awq',
        'recommended_for': ['email_generation'],
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'gemma-3-4b':{
        'model_id': 'gaunernst/gemma-3-4b-it-qat-autoawq',
        'recommended_for': ['email_generation'],
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'qwen-3-8b':{
        'model_id': 'Qwen/Qwen3-8B-AWQ',
        'recommended_for': ['email_generation'],
        'quantization': 'awq',
        'dtype': 'float16'
    },
    'deepseek-r1-70b': {
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'recommended_for': ['checklist_generation', 'judge'],
        'quantization': 'experts_int8',
        'dtype': 'bfloat16',
        'sglang_config': {
            'tp_size': 4,
            'max_running_requests': 32,
            'context_length': 4096,
            'mem_fraction_static': 0.85,
            'attention_backend': 'flashinfer'
        }
    },
    'llama-4-109b': {
        'model_id': 'kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ',
        'recommended_for': ['checklist_generation', 'judge'],
        'quantization': 'awq',
        'dtype': 'float16'
    }
}