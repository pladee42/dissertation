"""
SGLang Configuration for Multi-Agent AI System
Cluster-optimized settings for Sheffield University SLURM environment
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SGLangServerConfig:
    """SGLang server configuration for cluster deployment"""
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 30000
    api_key: Optional[str] = None
    
    # Model loading settings
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = True
    
    # Performance settings
    tp_size: int = 4  # Tensor parallelism for 4-GPU setup
    dp_size: int = 1  # Data parallelism
    context_length: int = 4096
    max_running_requests: int = 256
    max_waiting_requests: int = 512
    
    # Memory management
    mem_fraction_static: float = 0.85
    chunk_prefill_budget: int = 512
    disable_radix_cache: bool = False
    enable_mixed_precision: bool = True
    
    # Quantization settings
    quantization: Optional[str] = None  # "fp8", "int4", "int8", None
    load_format: str = "auto"
    
    # Backend optimization
    attention_backend: str = "flashinfer"  # "flashinfer", "triton", "xformers"
    disable_flashinfer: bool = False
    enable_torch_compile: bool = False
    
    # Scheduling and batching
    schedule_policy: str = "lpm"  # "lpm", "random", "fcfs"
    schedule_conservativeness: float = 1.0
    
    # Logging and monitoring
    log_level: str = "info"
    log_requests: bool = True
    show_time_cost: bool = True
    
    # Cluster-specific settings
    cuda_devices: str = "0,1,2,3"
    worker_use_ray: bool = True
    
    def to_server_args(self) -> list[str]:
        """Convert config to SGLang server command line arguments"""
        args = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--tp-size", str(self.tp_size),
            "--dp-size", str(self.dp_size),
            "--context-length", str(self.context_length),
            "--max-running-requests", str(self.max_running_requests),
            "--max-waiting-requests", str(self.max_waiting_requests),
            "--mem-fraction-static", str(self.mem_fraction_static),
            "--chunk-prefill-budget", str(self.chunk_prefill_budget),
            "--attention-backend", self.attention_backend,
            "--schedule-policy", self.schedule_policy,
            "--schedule-conservativeness", str(self.schedule_conservativeness),
            "--log-level", self.log_level,
        ]
        
        if self.tokenizer_path:
            args.extend(["--tokenizer-path", self.tokenizer_path])
        
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        
        if self.disable_radix_cache:
            args.append("--disable-radix-cache")
        
        if self.enable_mixed_precision:
            args.append("--enable-mixed-precision")
        
        if self.disable_flashinfer:
            args.append("--disable-flashinfer")
        
        if self.enable_torch_compile:
            args.append("--enable-torch-compile")
        
        if self.log_requests:
            args.append("--log-requests")
        
        if self.show_time_cost:
            args.append("--show-time-cost")
        
        if self.worker_use_ray:
            args.append("--worker-use-ray")
        
        if self.api_key:
            args.extend(["--api-key", self.api_key])
        
        return args

# Default configurations for different model sizes
SGLANG_CONFIGS = {
    "small_model": SGLangServerConfig(
        port=30000,
        tp_size=1,
        context_length=4096,
        mem_fraction_static=0.7,
        max_running_requests=128,
        quantization="fp8"
    ),
    
    "medium_model": SGLangServerConfig(
        port=30001,
        tp_size=2,
        context_length=4096,
        mem_fraction_static=0.8,
        max_running_requests=64,
        quantization="int8"
    ),
    
    "large_model": SGLangServerConfig(
        port=30002,
        tp_size=4,
        context_length=4096,
        mem_fraction_static=0.85,
        max_running_requests=32,
        quantization=None
    )
}

def get_sglang_config(model_name: str, custom_config: Optional[Dict[str, Any]] = None) -> SGLangServerConfig:
    """Get SGLang configuration for a specific model"""
    
    # Determine config based on model characteristics
    model_lower = model_name.lower()
    
    if any(size in model_lower for size in ["1.5b", "3b", "7b", "8b"]):
        base_config = SGLANG_CONFIGS["small_model"]
    elif any(size in model_lower for size in ["12b", "13b", "14b", "27b", "32b"]):
        base_config = SGLANG_CONFIGS["medium_model"]
    else:  # 70b and larger
        base_config = SGLANG_CONFIGS["large_model"]
    
    # Apply custom overrides
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(base_config, key):
                setattr(base_config, key, value)
    
    return base_config

def get_environment_variables() -> Dict[str, str]:
    """Get SGLang-specific environment variables for cluster deployment"""
    return {
        "SGLANG_BACKEND": "flashinfer",
        "SGLANG_DISABLE_DISK_CACHE": "false",
        "SGLANG_CHUNK_PREFILL_BUDGET": "512",
        "SGLANG_MEM_FRACTION_STATIC": "0.85",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": "${PYTHONPATH}:/path/to/sglang",
        "RAY_DEDUP_LOGS": "0",
        "RAY_DISABLE_IMPORT_WARNING": "1"
    }