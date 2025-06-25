"""
Model Factory for switching between VLLM and SGLang backends
Provides a unified interface for model loading and inference
"""

import os
import logging
from typing import Dict, Optional, Any, Union
from config.models import MODELS_CONFIG
from config.settings import settings

logger = logging.getLogger(__name__)

# Backend selection via environment variable
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "vllm").lower()

def create_model_instance(model_name: str, 
                         backend: Optional[str] = None,
                         custom_config: Optional[Dict[str, Any]] = None) -> Union['ModelInference', 'SGLangModelInference']:
    """
    Factory function to create model instances with configurable backend
    
    Args:
        model_name: Name of the model from MODELS_CONFIG
        backend: Backend to use ('vllm' or 'sglang'). If None, uses INFERENCE_BACKEND env var
        custom_config: Custom configuration overrides
    
    Returns:
        Model inference instance (VLLM or SGLang)
    """
    
    if backend is None:
        backend = INFERENCE_BACKEND
    
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model {model_name} not found in MODELS_CONFIG")
    
    model_config = MODELS_CONFIG[model_name].copy()
    
    # Apply custom config overrides
    if custom_config:
        model_config.update(custom_config)
    
    model_id = model_config['model_id']
    dtype = model_config.get('dtype', 'bfloat16')
    quantization = model_config.get('quantization', None)
    
    logger.info(f"Creating {backend.upper()} model instance for {model_name}")
    
    if backend == "sglang":
        from models.sglang_llm import SGLangModelInference
        
        # Extract SGLang-specific config
        sglang_config = model_config.get('sglang_config', {})
        
        return SGLangModelInference(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=sglang_config
        )
    
    elif backend == "vllm":
        from models.llm import ModelInference
        
        return ModelInference(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'sglang'")

def get_available_backends() -> list[str]:
    """Get list of available inference backends"""
    backends = []
    
    try:
        import vllm
        backends.append("vllm")
    except ImportError:
        pass
    
    try:
        import sglang
        backends.append("sglang")
    except ImportError:
        pass
    
    return backends

def get_current_backend() -> str:
    """Get the currently configured backend"""
    return INFERENCE_BACKEND

def set_backend(backend: str):
    """Set the inference backend for new model instances"""
    global INFERENCE_BACKEND
    
    available_backends = get_available_backends()
    if backend not in available_backends:
        raise ValueError(f"Backend {backend} not available. Available: {available_backends}")
    
    INFERENCE_BACKEND = backend
    os.environ["INFERENCE_BACKEND"] = backend
    logger.info(f"Inference backend set to: {backend}")

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model configuration information"""
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model {model_name} not found in MODELS_CONFIG")
    
    config = MODELS_CONFIG[model_name].copy()
    
    return {
        "model_name": model_name,
        "model_id": config['model_id'],
        "recommended_for": config.get('recommended_for', []),
        "dtype": config.get('dtype', 'bfloat16'),
        "quantization": config.get('quantization', None),
        "has_sglang_config": 'sglang_config' in config,
        "sglang_config": config.get('sglang_config', {}),
        "available_backends": get_available_backends(),
        "current_backend": get_current_backend()
    }

class ModelManager:
    """Centralized model management with backend switching"""
    
    def __init__(self):
        self._active_models: Dict[str, Any] = {}
        self._backend = get_current_backend()
    
    def load_model(self, model_name: str, 
                   backend: Optional[str] = None,
                   custom_config: Optional[Dict[str, Any]] = None) -> Any:
        """Load a model with specified backend"""
        
        if backend is None:
            backend = self._backend
        
        cache_key = f"{model_name}_{backend}"
        
        if cache_key in self._active_models:
            logger.info(f"Returning cached model instance: {cache_key}")
            return self._active_models[cache_key]
        
        logger.info(f"Loading new model instance: {cache_key}")
        model_instance = create_model_instance(model_name, backend, custom_config)
        
        self._active_models[cache_key] = model_instance
        return model_instance
    
    def unload_model(self, model_name: str, backend: Optional[str] = None):
        """Unload a specific model"""
        
        if backend is None:
            backend = self._backend
        
        cache_key = f"{model_name}_{backend}"
        
        if cache_key in self._active_models:
            model_instance = self._active_models[cache_key]
            try:
                model_instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up model {cache_key}: {e}")
            
            del self._active_models[cache_key]
            logger.info(f"Unloaded model: {cache_key}")
    
    def cleanup_all(self):
        """Cleanup all active models"""
        logger.info("Cleaning up all active models")
        
        for cache_key, model_instance in self._active_models.items():
            try:
                model_instance.cleanup()
                logger.info(f"Cleaned up model: {cache_key}")
            except Exception as e:
                logger.error(f"Error cleaning up model {cache_key}: {e}")
        
        self._active_models.clear()
    
    def get_active_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active models"""
        
        active_info = {}
        
        for cache_key, model_instance in self._active_models.items():
            try:
                model_info = model_instance.get_model_info()
                active_info[cache_key] = model_info
            except Exception as e:
                logger.error(f"Error getting info for model {cache_key}: {e}")
                active_info[cache_key] = {"error": str(e)}
        
        return active_info
    
    def switch_backend(self, new_backend: str):
        """Switch backend and cleanup existing models"""
        
        if new_backend == self._backend:
            logger.info(f"Already using backend: {new_backend}")
            return
        
        logger.info(f"Switching from {self._backend} to {new_backend}")
        
        # Cleanup existing models
        self.cleanup_all()
        
        # Switch backend
        set_backend(new_backend)
        self._backend = new_backend
        
        logger.info(f"Backend switched to: {new_backend}")

# Global model manager instance
model_manager = ModelManager()