from pydantic_settings import BaseSettings
from typing import Dict, Optional, List
from enum import Enum

class MemoryStrategy(str, Enum):
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    ADAPTIVE = "adaptive"

class Settings(BaseSettings):
    # Existing settings
    huggingface_token: Optional[str] = None
    download_dir: str = "./downloaded_models"
    output_dir: str = "./output"
    max_retries: int = 3
    default_temperature: float = 0.7
    default_top_p: float = 0.95
    default_max_tokens: int = 4096
    default_repetition_penalty: float = 1.2
    
    # Advanced Memory Management Settings
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    
    # GPU Memory Limits (in GB)
    max_gpu_memory_per_model: float = 24.0
    gpu_memory_safety_margin: float = 2.0
    gpu_memory_utilization_threshold: float = 0.85
    
    # System Memory Limits (in GB)  
    max_system_memory_per_model: float = 32.0
    system_memory_safety_margin: float = 4.0
    system_memory_utilization_threshold: float = 0.8
    
    # Model Loading Settings
    enable_cpu_fallback: bool = True
    enable_model_quantization: bool = True
    preferred_quantization: List[str] = ["fp8", "int8", "experts_int8"]
    enable_process_isolation: bool = False
    
    # Memory Monitoring
    enable_memory_monitoring: bool = True
    memory_check_interval_seconds: int = 30
    memory_profile_history_limit: int = 100
    
    # Sequential Processing
    force_sequential_processing: bool = False
    max_concurrent_models: int = 2
    sequential_cleanup_delay_seconds: float = 1.0
    
    # Emergency Memory Management
    enable_emergency_cleanup: bool = True
    emergency_memory_threshold: float = 0.95
    emergency_fallback_to_cpu: bool = True
    
    # Process Isolation Settings
    isolation_method: str = "multiprocessing"  # "multiprocessing" or "subprocess"
    isolation_timeout_seconds: int = 3600
    isolation_memory_limit_gb: Optional[float] = None
    
    # Optimization Settings
    enable_memory_profiling: bool = False
    enable_optimization_suggestions: bool = True
    auto_optimize_quantization: bool = True
    auto_optimize_batch_size: bool = True
    
    # Graceful Degradation
    enable_graceful_degradation: bool = True
    degradation_steps: List[str] = [
        "reduce_batch_size",
        "enable_quantization", 
        "switch_to_smaller_model",
        "fallback_to_cpu",
        "enable_process_isolation"
    ]
    
    # Model-specific overrides (can be set via environment)
    model_memory_overrides: Dict[str, float] = {}
    
    class Config:
        env_file = ".env"
        use_enum_values = True

settings = Settings()
