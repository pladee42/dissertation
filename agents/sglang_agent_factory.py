"""
SGLang Agent Factory
Unified factory for creating and managing SGLang-optimized agents
"""

import os
import logging
from typing import Dict, Optional, Any, Union
from config.settings import settings

logger = logging.getLogger(__name__)

# Agent backend selection via environment variable
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "vllm").lower()

def create_email_agent(model_id: str, 
                      dtype: str = "bfloat16", 
                      quantization: str = None,
                      backend: Optional[str] = None,
                      custom_config: Optional[Dict[str, Any]] = None) -> Union['EmailAgent', 'SGLangEmailAgent']:
    """
    Factory function to create email agents with configurable backend
    
    Args:
        model_id: Model identifier
        dtype: Data type for model
        quantization: Quantization method
        backend: Backend to use ('vllm' or 'sglang'). If None, uses AGENT_BACKEND env var
        custom_config: Custom configuration overrides
    
    Returns:
        Email agent instance (VLLM or SGLang)
    """
    
    if backend is None:
        backend = AGENT_BACKEND
    
    logger.info(f"Creating {backend.upper()} email agent for model: {model_id}")
    
    if backend == "sglang":
        from agents.sglang_email_agent import SGLangEmailAgent
        return SGLangEmailAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
    
    elif backend == "vllm":
        from agents.email_agent import EmailAgent
        return EmailAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'sglang'")

def create_checklist_agent(model_id: str, 
                          dtype: str = "bfloat16", 
                          quantization: str = None,
                          backend: Optional[str] = None,
                          custom_config: Optional[Dict[str, Any]] = None) -> Union['ChecklistAgent', 'SGLangChecklistAgent']:
    """
    Factory function to create checklist agents with configurable backend
    
    Args:
        model_id: Model identifier
        dtype: Data type for model
        quantization: Quantization method
        backend: Backend to use ('vllm' or 'sglang'). If None, uses AGENT_BACKEND env var
        custom_config: Custom configuration overrides
    
    Returns:
        Checklist agent instance (VLLM or SGLang)
    """
    
    if backend is None:
        backend = AGENT_BACKEND
    
    logger.info(f"Creating {backend.upper()} checklist agent for model: {model_id}")
    
    if backend == "sglang":
        from agents.sglang_checklist_agent import SGLangChecklistAgent
        return SGLangChecklistAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
    
    elif backend == "vllm":
        from agents.checklist_agent import ChecklistAgent
        return ChecklistAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'sglang'")

def create_judge_agent(model_id: str, 
                      dtype: str = "bfloat16", 
                      quantization: str = None,
                      backend: Optional[str] = None,
                      custom_config: Optional[Dict[str, Any]] = None) -> Union['JudgeAgent', 'SGLangJudgeAgent']:
    """
    Factory function to create judge agents with configurable backend
    
    Args:
        model_id: Model identifier
        dtype: Data type for model
        quantization: Quantization method
        backend: Backend to use ('vllm' or 'sglang'). If None, uses AGENT_BACKEND env var
        custom_config: Custom configuration overrides
    
    Returns:
        Judge agent instance (VLLM or SGLang)
    """
    
    if backend is None:
        backend = AGENT_BACKEND
    
    logger.info(f"Creating {backend.upper()} judge agent for model: {model_id}")
    
    if backend == "sglang":
        from agents.sglang_judge_agent import SGLangJudgeAgent
        return SGLangJudgeAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
    
    elif backend == "vllm":
        from agents.judge_agent import JudgeAgent
        return JudgeAgent(
            model_id=model_id,
            dtype=dtype,
            quantization=quantization
        )
    
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'sglang'")

def create_agent_pipeline(email_model_id: str,
                         checklist_model_id: str,
                         judge_model_id: str,
                         backend: Optional[str] = None,
                         custom_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Create a complete agent pipeline with all three agents
    
    Args:
        email_model_id: Model ID for email agent
        checklist_model_id: Model ID for checklist agent
        judge_model_id: Model ID for judge agent
        backend: Backend to use for all agents
        custom_configs: Custom configurations for each agent type
    
    Returns:
        Dictionary containing all three agents and pipeline metadata
    """
    
    if backend is None:
        backend = AGENT_BACKEND
    
    if custom_configs is None:
        custom_configs = {}
    
    logger.info(f"Creating {backend.upper()} agent pipeline")
    
    try:
        # Create agents
        email_agent = create_email_agent(
            email_model_id,
            backend=backend,
            custom_config=custom_configs.get("email", {})
        )
        
        checklist_agent = create_checklist_agent(
            checklist_model_id,
            backend=backend,
            custom_config=custom_configs.get("checklist", {})
        )
        
        judge_agent = create_judge_agent(
            judge_model_id,
            backend=backend,
            custom_config=custom_configs.get("judge", {})
        )
        
        pipeline = {
            "email_agent": email_agent,
            "checklist_agent": checklist_agent,
            "judge_agent": judge_agent,
            "backend": backend,
            "model_ids": {
                "email": email_model_id,
                "checklist": checklist_model_id,
                "judge": judge_model_id
            },
            "created_at": None,  # Will be set by datetime when used
            "features": _get_backend_features(backend)
        }
        
        logger.info(f"Agent pipeline created successfully with {backend.upper()} backend")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to create agent pipeline: {e}")
        raise

def cleanup_agent_pipeline(pipeline: Dict[str, Any]):
    """Cleanup all agents in a pipeline"""
    
    logger.info(f"Cleaning up {pipeline.get('backend', 'unknown')} agent pipeline")
    
    agents = ["email_agent", "checklist_agent", "judge_agent"]
    
    for agent_name in agents:
        if agent_name in pipeline:
            try:
                agent = pipeline[agent_name]
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                    logger.debug(f"Cleaned up {agent_name}")
            except Exception as e:
                logger.error(f"Error cleaning up {agent_name}: {e}")

def get_available_backends() -> list[str]:
    """Get list of available agent backends"""
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
    """Get the currently configured agent backend"""
    return AGENT_BACKEND

def set_agent_backend(backend: str):
    """Set the agent backend for new agent instances"""
    global AGENT_BACKEND
    
    available_backends = get_available_backends()
    if backend not in available_backends:
        raise ValueError(f"Backend {backend} not available. Available: {available_backends}")
    
    AGENT_BACKEND = backend
    os.environ["AGENT_BACKEND"] = backend
    logger.info(f"Agent backend set to: {backend}")

def get_agent_info(agent) -> Dict[str, Any]:
    """Get information about an agent"""
    
    if hasattr(agent, 'get_agent_info'):
        return agent.get_agent_info()
    else:
        # Fallback for agents without get_agent_info method
        return {
            "model_id": getattr(agent, 'model_id', 'unknown'),
            "model_name": getattr(agent, 'model_name', 'unknown'),
            "backend": "unknown"
        }

def _get_backend_features(backend: str) -> list[str]:
    """Get feature list for a specific backend"""
    
    backend_features = {
        "vllm": [
            "tensor_parallelism",
            "quantization",
            "attention_optimization",
            "batch_processing"
        ],
        "sglang": [
            "radix_attention",
            "structured_generation",
            "constrained_output",
            "xgrammar_validation",
            "fork_join_primitives",
            "automatic_cache_reuse",
            "prefix_caching",
            "server_based_architecture"
        ]
    }
    
    return backend_features.get(backend, [])

class AgentManager:
    """Centralized agent management with backend switching"""
    
    def __init__(self):
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
        self._backend = get_current_backend()
    
    def create_pipeline(self, 
                       pipeline_id: str,
                       email_model_id: str,
                       checklist_model_id: str,
                       judge_model_id: str,
                       backend: Optional[str] = None,
                       custom_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create and register a new agent pipeline"""
        
        if pipeline_id in self._active_pipelines:
            logger.warning(f"Pipeline {pipeline_id} already exists, cleaning up first")
            self.cleanup_pipeline(pipeline_id)
        
        pipeline = create_agent_pipeline(
            email_model_id=email_model_id,
            checklist_model_id=checklist_model_id,
            judge_model_id=judge_model_id,
            backend=backend or self._backend,
            custom_configs=custom_configs
        )
        
        pipeline["pipeline_id"] = pipeline_id
        self._active_pipelines[pipeline_id] = pipeline
        
        logger.info(f"Created and registered pipeline: {pipeline_id}")
        
        return pipeline
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get an active pipeline by ID"""
        return self._active_pipelines.get(pipeline_id)
    
    def cleanup_pipeline(self, pipeline_id: str):
        """Cleanup a specific pipeline"""
        
        if pipeline_id in self._active_pipelines:
            pipeline = self._active_pipelines[pipeline_id]
            cleanup_agent_pipeline(pipeline)
            del self._active_pipelines[pipeline_id]
            logger.info(f"Cleaned up pipeline: {pipeline_id}")
    
    def cleanup_all_pipelines(self):
        """Cleanup all active pipelines"""
        
        logger.info("Cleaning up all active pipelines")
        
        for pipeline_id in list(self._active_pipelines.keys()):
            self.cleanup_pipeline(pipeline_id)
    
    def switch_backend(self, new_backend: str):
        """Switch backend and cleanup existing pipelines"""
        
        if new_backend == self._backend:
            logger.info(f"Already using backend: {new_backend}")
            return
        
        logger.info(f"Switching agent backend from {self._backend} to {new_backend}")
        
        # Cleanup existing pipelines
        self.cleanup_all_pipelines()
        
        # Switch backend
        set_agent_backend(new_backend)
        self._backend = new_backend
        
        logger.info(f"Agent backend switched to: {new_backend}")
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active pipelines"""
        
        pipeline_info = {}
        
        for pipeline_id, pipeline in self._active_pipelines.items():
            try:
                pipeline_info[pipeline_id] = {
                    "backend": pipeline.get("backend", "unknown"),
                    "model_ids": pipeline.get("model_ids", {}),
                    "features": pipeline.get("features", []),
                    "agents": {
                        "email": get_agent_info(pipeline.get("email_agent")),
                        "checklist": get_agent_info(pipeline.get("checklist_agent")),
                        "judge": get_agent_info(pipeline.get("judge_agent"))
                    }
                }
            except Exception as e:
                logger.error(f"Error getting info for pipeline {pipeline_id}: {e}")
                pipeline_info[pipeline_id] = {"error": str(e)}
        
        return pipeline_info

# Global agent manager instance
agent_manager = AgentManager()