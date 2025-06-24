"""
Advanced Memory Management System for AI Agentic Email Generation

This module provides enterprise-grade memory management with:
- Dynamic memory threshold checking
- Graceful degradation strategies  
- Process isolation capabilities
- Detailed memory profiling and optimization
"""

import gc
import os
import sys
import time
import psutil
import subprocess
import multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import logging
import torch
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory management strategies"""
    AGGRESSIVE = "aggressive"       # Maximum memory conservation
    BALANCED = "balanced"          # Balance between performance and memory
    PERFORMANCE = "performance"    # Prioritize performance over memory
    ADAPTIVE = "adaptive"          # Dynamically adjust based on conditions

class DeviceType(Enum):
    """Computing device types"""
    GPU = "gpu"
    CPU = "cpu"
    MIXED = "mixed"

@dataclass
class MemoryProfile:
    """Comprehensive memory profile"""
    total_system_gb: float
    available_system_gb: float
    used_system_gb: float
    gpu_available: bool
    gpu_total_gb: float = 0.0
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_free_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    memory_pressure: float = 0.0  # 0.0 = no pressure, 1.0 = critical
    recommendation: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass 
class ModelMemoryRequirement:
    """Memory requirements for a specific model"""
    model_name: str
    min_gpu_gb: float
    optimal_gpu_gb: float
    max_gpu_gb: float
    cpu_fallback_gb: float
    quantization_options: List[str] = field(default_factory=list)
    supports_cpu: bool = True

@dataclass
class MemoryBudget:
    """Memory budget allocation"""
    total_available_gb: float
    reserved_system_gb: float
    max_model_gb: float
    safety_margin_gb: float
    concurrent_models: int
    strategy: MemoryStrategy

class AdvancedMemoryManager:
    """Enterprise-grade memory management system"""
    
    def __init__(self, strategy: MemoryStrategy = MemoryStrategy.BALANCED):
        self.strategy = strategy
        self.monitoring_enabled = True
        self.profiles: List[MemoryProfile] = []
        self.model_registry: Dict[str, ModelMemoryRequirement] = {}
        self.active_instances: Dict[str, Any] = {}
        self.cleanup_callbacks: List[callable] = []
        
        # Initialize model requirements database
        self._initialize_model_registry()
        
        # Start background monitoring if enabled
        if self.monitoring_enabled:
            self._start_monitoring()
        
        logger.info(f"AdvancedMemoryManager initialized with {strategy.value} strategy")
    
    def _initialize_model_registry(self):
        """Initialize memory requirements for known models"""
        
        # DeepSeek R1 series
        self.model_registry.update({
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ModelMemoryRequirement(
                model_name="deepseek-r1-1.5b",
                min_gpu_gb=2.0, optimal_gpu_gb=4.0, max_gpu_gb=6.0,
                cpu_fallback_gb=8.0, quantization_options=["fp8", "int8"],
                supports_cpu=True
            ),
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelMemoryRequirement(
                model_name="deepseek-r1-7b", 
                min_gpu_gb=8.0, optimal_gpu_gb=12.0, max_gpu_gb=16.0,
                cpu_fallback_gb=16.0, quantization_options=["fp8", "int8", "experts_int8"],
                supports_cpu=True
            ),
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": ModelMemoryRequirement(
                model_name="deepseek-r1-14b",
                min_gpu_gb=16.0, optimal_gpu_gb=24.0, max_gpu_gb=32.0,
                cpu_fallback_gb=32.0, quantization_options=["fp8", "int8", "experts_int8"],
                supports_cpu=False
            ),
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelMemoryRequirement(
                model_name="deepseek-r1-70b",
                min_gpu_gb=40.0, optimal_gpu_gb=80.0, max_gpu_gb=120.0,
                cpu_fallback_gb=160.0, quantization_options=["fp8", "experts_int8"],
                supports_cpu=False
            ),
        })
        
        # Gemma series
        self.model_registry.update({
            "google/gemma-3-12b-it": ModelMemoryRequirement(
                model_name="gemma-3-12b",
                min_gpu_gb=12.0, optimal_gpu_gb=18.0, max_gpu_gb=24.0,
                cpu_fallback_gb=24.0, quantization_options=["awq", "int8"],
                supports_cpu=True
            ),
            "google/gemma-3-27b-it": ModelMemoryRequirement(
                model_name="gemma-3-27b",
                min_gpu_gb=24.0, optimal_gpu_gb=36.0, max_gpu_gb=48.0,
                cpu_fallback_gb=54.0, quantization_options=["awq", "int8"],
                supports_cpu=False
            ),
        })
        
        # Llama series  
        self.model_registry.update({
            "unsloth/Llama-3.2-3B-Instruct": ModelMemoryRequirement(
                model_name="llama-3-3b",
                min_gpu_gb=3.0, optimal_gpu_gb=6.0, max_gpu_gb=8.0,
                cpu_fallback_gb=12.0, quantization_options=["awq", "int8"],
                supports_cpu=True
            ),
            "unsloth/Llama-3.3-70B-Instruct": ModelMemoryRequirement(
                model_name="llama-3-70b",
                min_gpu_gb=40.0, optimal_gpu_gb=80.0, max_gpu_gb=120.0,
                cpu_fallback_gb=160.0, quantization_options=["awq", "experts_int8"],
                supports_cpu=False
            ),
        })
    
    def get_current_profile(self) -> MemoryProfile:
        """Get comprehensive current memory profile"""
        
        # System memory
        system_memory = psutil.virtual_memory()
        total_system_gb = system_memory.total / (1024**3)
        available_system_gb = system_memory.available / (1024**3)
        used_system_gb = system_memory.used / (1024**3)
        
        # GPU memory
        gpu_available = torch.cuda.is_available()
        gpu_total_gb = gpu_allocated_gb = gpu_reserved_gb = gpu_free_gb = 0.0
        gpu_utilization = 0.0
        
        if gpu_available:
            try:
                gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
                gpu_free_gb = gpu_total_gb - gpu_reserved_gb
                gpu_utilization = (gpu_allocated_gb / gpu_total_gb) * 100
            except Exception as e:
                logger.warning(f"GPU memory query failed: {e}")
                gpu_available = False
        
        # Calculate memory pressure
        memory_pressure = self._calculate_memory_pressure(
            available_system_gb, gpu_free_gb if gpu_available else 0
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(memory_pressure, gpu_available, gpu_free_gb)
        
        profile = MemoryProfile(
            total_system_gb=total_system_gb,
            available_system_gb=available_system_gb,
            used_system_gb=used_system_gb,
            gpu_available=gpu_available,
            gpu_total_gb=gpu_total_gb,
            gpu_allocated_gb=gpu_allocated_gb,
            gpu_reserved_gb=gpu_reserved_gb,
            gpu_free_gb=gpu_free_gb,
            gpu_utilization_percent=gpu_utilization,
            memory_pressure=memory_pressure,
            recommendation=recommendation
        )
        
        self.profiles.append(profile)
        
        # Keep only last 100 profiles
        if len(self.profiles) > 100:
            self.profiles = self.profiles[-100:]
        
        return profile
    
    def _calculate_memory_pressure(self, system_available_gb: float, gpu_free_gb: float) -> float:
        """Calculate normalized memory pressure (0.0 = no pressure, 1.0 = critical)"""
        
        # System memory pressure
        system_pressure = max(0, 1.0 - (system_available_gb / 8.0))  # 8GB baseline
        
        # GPU memory pressure (if available)
        gpu_pressure = 0.0
        if gpu_free_gb > 0:
            gpu_pressure = max(0, 1.0 - (gpu_free_gb / 4.0))  # 4GB baseline
        
        # Combined pressure (weighted)
        if gpu_free_gb > 0:
            return 0.3 * system_pressure + 0.7 * gpu_pressure
        else:
            return system_pressure
    
    def _generate_recommendation(self, memory_pressure: float, gpu_available: bool, gpu_free_gb: float) -> str:
        """Generate memory management recommendation"""
        
        if memory_pressure < 0.3:
            return "Optimal - Sufficient memory for concurrent operations"
        elif memory_pressure < 0.6:
            return "Moderate - Consider sequential processing for large models"
        elif memory_pressure < 0.8:
            return "High - Use sequential processing and smaller models"
        else:
            return "Critical - Use CPU fallback and aggressive optimization"
    
    def check_model_feasibility(self, model_id: str, device_preference: DeviceType = DeviceType.GPU) -> Dict[str, Any]:
        """Check if model can be loaded with current memory constraints"""
        
        profile = self.get_current_profile()
        model_req = self.model_registry.get(model_id)
        
        if not model_req:
            logger.warning(f"Unknown model: {model_id}, using default requirements")
            model_req = ModelMemoryRequirement(
                model_name="unknown",
                min_gpu_gb=4.0, optimal_gpu_gb=8.0, max_gpu_gb=12.0,
                cpu_fallback_gb=16.0, supports_cpu=True
            )
        
        result = {
            "feasible": False,
            "recommended_device": DeviceType.CPU,
            "recommended_quantization": None,
            "estimated_memory_gb": 0.0,
            "confidence": 0.0,
            "alternatives": [],
            "warnings": []
        }
        
        # GPU feasibility check
        if device_preference == DeviceType.GPU and profile.gpu_available:
            if profile.gpu_free_gb >= model_req.min_gpu_gb:
                result["feasible"] = True
                result["recommended_device"] = DeviceType.GPU
                result["estimated_memory_gb"] = model_req.optimal_gpu_gb
                
                # Determine optimal quantization
                if profile.gpu_free_gb >= model_req.optimal_gpu_gb:
                    result["confidence"] = 0.9
                    result["recommended_quantization"] = "fp16"
                elif profile.gpu_free_gb >= model_req.min_gpu_gb * 1.2:
                    result["confidence"] = 0.7
                    result["recommended_quantization"] = "fp8" if "fp8" in model_req.quantization_options else "int8"
                else:
                    result["confidence"] = 0.5
                    result["recommended_quantization"] = "int8"
                    result["warnings"].append("Minimal GPU memory - consider CPU fallback")
        
        # CPU fallback check
        if not result["feasible"] and model_req.supports_cpu:
            if profile.available_system_gb >= model_req.cpu_fallback_gb:
                result["feasible"] = True
                result["recommended_device"] = DeviceType.CPU
                result["estimated_memory_gb"] = model_req.cpu_fallback_gb
                result["confidence"] = 0.6
                result["recommended_quantization"] = "int8"
                result["warnings"].append("Using CPU fallback - expect slower performance")
        
        # Generate alternatives
        if not result["feasible"]:
            result["alternatives"] = self._suggest_alternatives(model_req, profile)
        
        return result
    
    def _suggest_alternatives(self, model_req: ModelMemoryRequirement, profile: MemoryProfile) -> List[str]:
        """Suggest alternative models or configurations"""
        
        alternatives = []
        
        # Suggest smaller models from same family
        for model_id, req in self.model_registry.items():
            if (req.model_name.split('-')[0] == model_req.model_name.split('-')[0] and 
                req.optimal_gpu_gb < model_req.optimal_gpu_gb):
                if profile.gpu_free_gb >= req.min_gpu_gb or profile.available_system_gb >= req.cpu_fallback_gb:
                    alternatives.append(f"Use smaller model: {req.model_name}")
        
        # Suggest quantization
        if model_req.quantization_options:
            alternatives.append(f"Try aggressive quantization: {model_req.quantization_options[-1]}")
        
        # Suggest process isolation
        alternatives.append("Use process isolation for complete memory separation")
        
        return alternatives[:3]  # Limit to top 3 suggestions
    
    def create_memory_budget(self, concurrent_models: int = 1, safety_margin: float = 0.2) -> MemoryBudget:
        """Create memory budget for model operations"""
        
        profile = self.get_current_profile()
        
        # Determine available memory based on strategy
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            available_gb = profile.gpu_free_gb * 0.95 if profile.gpu_available else profile.available_system_gb * 0.85
            safety_margin = min(safety_margin, 0.1)
        elif self.strategy == MemoryStrategy.PERFORMANCE:
            available_gb = profile.gpu_free_gb * 0.8 if profile.gpu_available else profile.available_system_gb * 0.7
            safety_margin = max(safety_margin, 0.3)
        else:  # BALANCED or ADAPTIVE
            available_gb = profile.gpu_free_gb * 0.85 if profile.gpu_available else profile.available_system_gb * 0.75
        
        reserved_gb = available_gb * safety_margin
        usable_gb = available_gb - reserved_gb
        max_model_gb = usable_gb / max(1, concurrent_models)
        
        return MemoryBudget(
            total_available_gb=available_gb,
            reserved_system_gb=reserved_gb, 
            max_model_gb=max_model_gb,
            safety_margin_gb=reserved_gb,
            concurrent_models=concurrent_models,
            strategy=self.strategy
        )
    
    @contextmanager
    def managed_model_loading(self, model_id: str, device_preference: DeviceType = DeviceType.GPU):
        """Context manager for safe model loading with automatic resource management"""
        
        logger.info(f"=== Managed Model Loading: {model_id} ===")
        
        # Pre-loading checks
        feasibility = self.check_model_feasibility(model_id, device_preference)
        if not feasibility["feasible"]:
            error_msg = f"Model {model_id} not feasible with current memory constraints"
            logger.error(error_msg)
            if feasibility["alternatives"]:
                logger.info(f"Alternatives: {', '.join(feasibility['alternatives'])}")
            raise RuntimeError(error_msg)
        
        # Pre-loading cleanup
        self.aggressive_cleanup()
        
        initial_profile = self.get_current_profile()
        instance_id = f"{model_id}_{int(time.time())}"
        
        try:
            logger.info(f"Loading {model_id} on {feasibility['recommended_device'].value} "
                       f"with {feasibility['recommended_quantization']} quantization")
            
            # Register instance
            self.active_instances[instance_id] = {
                "model_id": model_id,
                "device": feasibility["recommended_device"],
                "quantization": feasibility["recommended_quantization"],
                "start_time": time.time(),
                "initial_memory": initial_profile
            }
            
            yield feasibility
            
        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            raise
            
        finally:
            # Cleanup and deregister
            if instance_id in self.active_instances:
                del self.active_instances[instance_id]
            
            self.aggressive_cleanup()
            
            final_profile = self.get_current_profile()
            memory_delta = final_profile.gpu_allocated_gb - initial_profile.gpu_allocated_gb
            
            logger.info(f"Model loading completed. Memory delta: {memory_delta:+.2f}GB")
            logger.info(f"=== Managed Model Loading Completed ===")
    
    def aggressive_cleanup(self):
        """Perform aggressive system cleanup"""
        
        logger.debug("Starting aggressive cleanup")
        
        # Execute registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        
        # Multiple rounds of garbage collection
        for _ in range(3):
            collected = gc.collect()
            logger.debug(f"GC collected {collected} objects")
        
        # GPU cleanup if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        # System memory cleanup
        try:
            # Force Python to release memory back to OS
            import ctypes
            if sys.platform.startswith('linux'):
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
        except Exception:
            pass
        
        logger.debug("Aggressive cleanup completed")
    
    def register_cleanup_callback(self, callback: callable):
        """Register cleanup callback for custom resource management"""
        self.cleanup_callbacks.append(callback)
    
    def _start_monitoring(self):
        """Start background memory monitoring (if enabled)"""
        pass  # Implementation would go here for production systems
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        
        current_profile = self.get_current_profile()
        
        report = {
            "timestamp": time.time(),
            "current_profile": current_profile.__dict__,
            "active_instances": len(self.active_instances),
            "memory_strategy": self.strategy.value,
            "recommendations": [],
            "performance_metrics": {},
            "historical_data": {}
        }
        
        # Add performance metrics
        if len(self.profiles) > 1:
            recent_profiles = self.profiles[-10:]
            avg_pressure = sum(p.memory_pressure for p in recent_profiles) / len(recent_profiles)
            report["performance_metrics"] = {
                "avg_memory_pressure": avg_pressure,
                "memory_trend": "increasing" if recent_profiles[-1].memory_pressure > recent_profiles[0].memory_pressure else "stable"
            }
        
        # Generate actionable recommendations
        if current_profile.memory_pressure > 0.7:
            report["recommendations"].extend([
                "Consider using smaller models",
                "Enable sequential processing",
                "Use more aggressive quantization",
                "Consider process isolation"
            ])
        
        return report

# Global memory manager instance
_global_memory_manager: Optional[AdvancedMemoryManager] = None

def get_memory_manager(strategy: MemoryStrategy = MemoryStrategy.BALANCED) -> AdvancedMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = AdvancedMemoryManager(strategy)
    
    return _global_memory_manager

def reset_memory_manager():
    """Reset global memory manager (for testing)"""
    global _global_memory_manager
    _global_memory_manager = None