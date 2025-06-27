"""
Simplified Memory Management and Profiling System

This module provides essential memory management with:
- Basic memory monitoring
- Simple cleanup utilities
- Resource tracking
"""

import gc
import logging
import time
import psutil
import torch
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Simplified memory strategies
MEMORY_STRATEGIES = {
    "conservative": {"safety_margin": 0.2, "aggressive_cleanup": True},
    "performance": {"safety_margin": 0.1, "aggressive_cleanup": False}
}

class SimpleMemoryManager:
    """Simplified memory management system"""
    
    def __init__(self, strategy: str = "conservative"):
        self.strategy = MEMORY_STRATEGIES.get(strategy, MEMORY_STRATEGIES["conservative"])
        self.active_models: Dict[str, Any] = {}
        
        logger.info(f"SimpleMemoryManager initialized with {strategy} strategy")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information"""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info["system_total_gb"] = system_memory.total / (1024**3)
        memory_info["system_available_gb"] = system_memory.available / (1024**3)
        memory_info["system_used_gb"] = system_memory.used / (1024**3)
        
        # GPU memory if available
        if torch.cuda.is_available():
            memory_info["gpu_available"] = True
            memory_info["gpu_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
            memory_info["gpu_free_gb"] = memory_info["gpu_total_gb"] - memory_info["gpu_allocated_gb"]
        else:
            memory_info["gpu_available"] = False
            memory_info["gpu_total_gb"] = 0
            memory_info["gpu_allocated_gb"] = 0
            memory_info["gpu_reserved_gb"] = 0
            memory_info["gpu_free_gb"] = 0
        
        return memory_info
    
    def cleanup_memory(self, force_gpu_cleanup: bool = False):
        """Simple memory cleanup"""
        logger.info("Starting memory cleanup...")
        
        # Python garbage collection
        collected = gc.collect()
        logger.info(f"Collected {collected} objects")
        
        # GPU cleanup if requested
        if force_gpu_cleanup and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        memory_info = self.get_memory_info()
        
        # Check system memory
        system_usage = memory_info["system_used_gb"] / memory_info["system_total_gb"]
        if system_usage > (1 - self.strategy["safety_margin"]):
            return True
        
        # Check GPU memory
        if memory_info["gpu_available"]:
            gpu_usage = memory_info["gpu_allocated_gb"] / memory_info["gpu_total_gb"]
            if gpu_usage > (1 - self.strategy["safety_margin"]):
                return True
        
        return False
    
    def register_model(self, model_name: str, model_instance: Any):
        """Register a model instance for tracking"""
        self.active_models[model_name] = model_instance
        logger.info(f"Registered model: {model_name}")
    
    def unregister_model(self, model_name: str):
        """Unregister a model instance"""
        if model_name in self.active_models:
            del self.active_models[model_name]
            logger.info(f"Unregistered model: {model_name}")
    
    def get_recommendations(self) -> List[str]:
        """Get simple memory optimization recommendations"""
        recommendations = []
        memory_info = self.get_memory_info()
        
        # System memory recommendations
        system_usage = memory_info["system_used_gb"] / memory_info["system_total_gb"]
        if system_usage > 0.8:
            recommendations.append("High system memory usage - consider reducing batch size")
        
        # GPU memory recommendations
        if memory_info["gpu_available"]:
            gpu_usage = memory_info["gpu_allocated_gb"] / memory_info["gpu_total_gb"]
            if gpu_usage > 0.8:
                recommendations.append("High GPU memory usage - consider model quantization")
        
        # General recommendations
        if len(self.active_models) > 3:
            recommendations.append("Many active models - consider sequential processing")
        
        return recommendations

class SimpleMemoryProfiler:
    """Simplified memory profiling"""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def take_snapshot(self, operation_name: str = "unknown") -> Dict[str, Any]:
        """Take a simple memory snapshot"""
        memory_manager = get_memory_manager()
        memory_info = memory_manager.get_memory_info()
        
        snapshot = {
            "timestamp": time.time() - self.start_time,
            "operation": operation_name,
            "system_used_gb": memory_info["system_used_gb"],
            "gpu_allocated_gb": memory_info["gpu_allocated_gb"],
            "gpu_reserved_gb": memory_info["gpu_reserved_gb"]
        }
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_summary(self) -> Dict[str, Any]:
        """Get simple profiling summary"""
        if not self.snapshots:
            return {"error": "No snapshots taken"}
        
        # Calculate basic statistics
        gpu_usage = [s["gpu_allocated_gb"] for s in self.snapshots]
        system_usage = [s["system_used_gb"] for s in self.snapshots]
        
        return {
            "total_snapshots": len(self.snapshots),
            "duration_seconds": self.snapshots[-1]["timestamp"],
            "peak_gpu_usage_gb": max(gpu_usage) if gpu_usage else 0,
            "peak_system_usage_gb": max(system_usage) if system_usage else 0,
            "avg_gpu_usage_gb": sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0,
            "avg_system_usage_gb": sum(system_usage) / len(system_usage) if system_usage else 0
        }

# Global instance
_global_memory_manager = None

def get_memory_manager() -> SimpleMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = SimpleMemoryManager()
    return _global_memory_manager

def get_memory_profiler() -> SimpleMemoryProfiler:
    """Get a new memory profiler instance"""
    return SimpleMemoryProfiler()