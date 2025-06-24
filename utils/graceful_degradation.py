"""
Graceful Degradation System for AI Model Loading

This module provides enterprise-grade graceful degradation capabilities:
- Automatic CPU fallback when GPU memory is insufficient
- Dynamic model downsizing based on memory constraints
- Intelligent quantization selection
- Fallback chains with performance optimization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import time

from config.settings import settings, MemoryStrategy
from config.models import MODELS_CONFIG
from utils.memory_manager import get_memory_manager, DeviceType, MemoryProfile

logger = logging.getLogger(__name__)

class DegradationStep(Enum):
    """Available degradation steps"""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    ENABLE_QUANTIZATION = "enable_quantization"
    SWITCH_TO_SMALLER_MODEL = "switch_to_smaller_model"
    FALLBACK_TO_CPU = "fallback_to_cpu"
    ENABLE_PROCESS_ISOLATION = "enable_process_isolation"

@dataclass
class DegradationResult:
    """Result of graceful degradation attempt"""
    success: bool
    final_model_id: str
    final_device: DeviceType
    final_quantization: Optional[str]
    degradation_steps_applied: List[DegradationStep]
    performance_impact: float  # 0.0 = no impact, 1.0 = severe impact
    memory_saved_gb: float
    warnings: List[str]
    fallback_reason: str

@dataclass
class ModelAlternative:
    """Alternative model configuration"""
    model_id: str
    model_name: str
    memory_requirement_gb: float
    performance_score: float  # Relative to original model
    compatibility_score: float  # How well it substitutes original
    quantization_options: List[str]

class GracefulDegradationManager:
    """Manages graceful degradation strategies for model loading"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.model_families = self._build_model_families()
        self.degradation_history: List[DegradationResult] = []
        
    def _build_model_families(self) -> Dict[str, List[ModelAlternative]]:
        """Build model family hierarchies for intelligent downsizing"""
        
        families = {
            "deepseek-r1": [
                ModelAlternative(
                    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    model_name="deepseek-r1-1.5b",
                    memory_requirement_gb=4.0,
                    performance_score=0.6,
                    compatibility_score=0.9,
                    quantization_options=["fp8", "int8"]
                ),
                ModelAlternative(
                    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                    model_name="deepseek-r1-7b",
                    memory_requirement_gb=12.0,
                    performance_score=0.8,
                    compatibility_score=0.95,
                    quantization_options=["fp8", "int8", "experts_int8"]
                ),
                ModelAlternative(
                    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                    model_name="deepseek-r1-14b", 
                    memory_requirement_gb=24.0,
                    performance_score=0.9,
                    compatibility_score=0.98,
                    quantization_options=["fp8", "int8", "experts_int8"]
                ),
                ModelAlternative(
                    model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    model_name="deepseek-r1-70b",
                    memory_requirement_gb=80.0,
                    performance_score=1.0,
                    compatibility_score=1.0,
                    quantization_options=["fp8", "experts_int8"]
                ),
            ],
            "gemma": [
                ModelAlternative(
                    model_id="google/gemma-3-12b-it",
                    model_name="gemma-3-12b",
                    memory_requirement_gb=18.0,
                    performance_score=0.8,
                    compatibility_score=0.9,
                    quantization_options=["awq", "int8"]
                ),
                ModelAlternative(
                    model_id="google/gemma-3-27b-it",
                    model_name="gemma-3-27b",
                    memory_requirement_gb=36.0,
                    performance_score=1.0,
                    compatibility_score=1.0,
                    quantization_options=["awq", "int8"]
                ),
            ],
            "llama": [
                ModelAlternative(
                    model_id="unsloth/Llama-3.2-3B-Instruct",
                    model_name="llama-3-3b",
                    memory_requirement_gb=6.0,
                    performance_score=0.7,
                    compatibility_score=0.85,
                    quantization_options=["awq", "int8"]
                ),
                ModelAlternative(
                    model_id="unsloth/Llama-3.3-70B-Instruct",
                    model_name="llama-3-70b",
                    memory_requirement_gb=80.0,
                    performance_score=1.0,
                    compatibility_score=1.0,
                    quantization_options=["awq", "experts_int8"]
                ),
            ]
        }
        
        return families
    
    def attempt_graceful_degradation(self, 
                                   original_model_id: str,
                                   target_device: DeviceType = DeviceType.GPU,
                                   max_degradation_steps: int = 3) -> DegradationResult:
        """
        Attempt graceful degradation to make model loading feasible
        
        Args:
            original_model_id: The originally requested model
            target_device: Preferred device type
            max_degradation_steps: Maximum number of degradation steps to try
            
        Returns:
            DegradationResult with final configuration or failure
        """
        
        logger.info(f"Attempting graceful degradation for {original_model_id}")
        
        result = DegradationResult(
            success=False,
            final_model_id=original_model_id,
            final_device=target_device,
            final_quantization=None,
            degradation_steps_applied=[],
            performance_impact=0.0,
            memory_saved_gb=0.0,
            warnings=[],
            fallback_reason=""
        )
        
        current_profile = self.memory_manager.get_current_profile()
        initial_memory = current_profile.gpu_free_gb if current_profile.gpu_available else current_profile.available_system_gb
        
        # Try original configuration first
        feasibility = self.memory_manager.check_model_feasibility(original_model_id, target_device)
        if feasibility["feasible"]:
            result.success = True
            result.final_device = feasibility["recommended_device"]
            result.final_quantization = feasibility["recommended_quantization"]
            logger.info("Original model configuration is feasible")
            return result
        
        # Apply degradation steps sequentially
        current_model_id = original_model_id
        current_device = target_device
        degradation_steps_to_try = [DegradationStep(step) for step in settings.degradation_steps[:max_degradation_steps]]
        
        for step in degradation_steps_to_try:
            logger.info(f"Applying degradation step: {step.value}")
            
            try:
                step_result = self._apply_degradation_step(
                    step, current_model_id, current_device, result
                )
                
                if step_result["success"]:
                    current_model_id = step_result["model_id"]
                    current_device = step_result["device"]
                    result.degradation_steps_applied.append(step)
                    result.final_model_id = current_model_id
                    result.final_device = current_device
                    result.final_quantization = step_result.get("quantization")
                    
                    # Check if this configuration is now feasible
                    feasibility = self.memory_manager.check_model_feasibility(current_model_id, current_device)
                    if feasibility["feasible"]:
                        result.success = True
                        result.final_quantization = feasibility["recommended_quantization"]
                        
                        # Calculate performance impact and memory savings
                        result.performance_impact = self._calculate_performance_impact(
                            original_model_id, current_model_id, result.degradation_steps_applied
                        )
                        
                        final_profile = self.memory_manager.get_current_profile() 
                        final_memory = final_profile.gpu_free_gb if final_profile.gpu_available else final_profile.available_system_gb
                        result.memory_saved_gb = final_memory - initial_memory
                        
                        logger.info(f"Graceful degradation successful after {len(result.degradation_steps_applied)} steps")
                        break
                    
            except Exception as e:
                logger.warning(f"Degradation step {step.value} failed: {e}")
                result.warnings.append(f"Step {step.value} failed: {str(e)}")
        
        if not result.success:
            result.fallback_reason = f"All degradation steps exhausted. Last attempted: {step.value if 'step' in locals() else 'none'}"
            logger.error(f"Graceful degradation failed: {result.fallback_reason}")
        
        # Store in history
        self.degradation_history.append(result)
        
        return result
    
    def _apply_degradation_step(self, 
                              step: DegradationStep, 
                              current_model_id: str, 
                              current_device: DeviceType,
                              result: DegradationResult) -> Dict[str, Any]:
        """Apply a specific degradation step"""
        
        step_result = {
            "success": False,
            "model_id": current_model_id,
            "device": current_device,
            "quantization": None
        }
        
        if step == DegradationStep.ENABLE_QUANTIZATION:
            # Try more aggressive quantization
            step_result["success"] = True
            step_result["quantization"] = "int8"  # Most aggressive
            
        elif step == DegradationStep.FALLBACK_TO_CPU:
            if settings.enable_cpu_fallback:
                step_result["success"] = True
                step_result["device"] = DeviceType.CPU
                result.warnings.append("Falling back to CPU - expect slower performance")
            
        elif step == DegradationStep.SWITCH_TO_SMALLER_MODEL:
            smaller_model = self._find_smaller_model(current_model_id)
            if smaller_model:
                step_result["success"] = True
                step_result["model_id"] = smaller_model["model_id"]
                result.warnings.append(f"Switched to smaller model: {smaller_model['model_name']}")
            
        elif step == DegradationStep.ENABLE_PROCESS_ISOLATION:
            if settings.enable_process_isolation:
                step_result["success"] = True
                result.warnings.append("Enabled process isolation for memory separation")
            
        elif step == DegradationStep.REDUCE_BATCH_SIZE:
            # This would be handled at the model configuration level
            step_result["success"] = True
            result.warnings.append("Reduced batch size for memory optimization")
        
        return step_result
    
    def _find_smaller_model(self, current_model_id: str) -> Optional[Dict[str, str]]:
        """Find a smaller model from the same family"""
        
        # Determine model family
        family_name = None
        for family, models in self.model_families.items():
            if any(alt.model_id == current_model_id for alt in models):
                family_name = family
                break
        
        if not family_name:
            logger.warning(f"No family found for model {current_model_id}")
            return None
        
        # Find current model in family
        current_model = None
        family_models = self.model_families[family_name]
        for model in family_models:
            if model.model_id == current_model_id:
                current_model = model
                break
        
        if not current_model:
            return None
        
        # Find smaller models (lower memory requirement)
        smaller_models = [
            model for model in family_models 
            if model.memory_requirement_gb < current_model.memory_requirement_gb
        ]
        
        if not smaller_models:
            return None
        
        # Select the largest among smaller models for best performance
        best_smaller = max(smaller_models, key=lambda m: m.memory_requirement_gb)
        
        return {
            "model_id": best_smaller.model_id,
            "model_name": best_smaller.model_name
        }
    
    def _calculate_performance_impact(self, 
                                    original_model_id: str, 
                                    final_model_id: str, 
                                    steps_applied: List[DegradationStep]) -> float:
        """Calculate estimated performance impact (0.0 = no impact, 1.0 = severe)"""
        
        impact = 0.0
        
        # Model downsizing impact
        if original_model_id != final_model_id:
            original_size = self._estimate_model_size(original_model_id)
            final_size = self._estimate_model_size(final_model_id)
            if original_size and final_size:
                size_ratio = final_size / original_size
                impact += (1.0 - size_ratio) * 0.6  # Up to 60% impact from model size
        
        # Device fallback impact
        if DegradationStep.FALLBACK_TO_CPU in steps_applied:
            impact += 0.3  # 30% performance impact from CPU fallback
        
        # Quantization impact
        if DegradationStep.ENABLE_QUANTIZATION in steps_applied:
            impact += 0.1  # 10% impact from aggressive quantization
        
        # Process isolation impact
        if DegradationStep.ENABLE_PROCESS_ISOLATION in steps_applied:
            impact += 0.05  # 5% impact from process isolation overhead
        
        return min(impact, 1.0)
    
    def _estimate_model_size(self, model_id: str) -> Optional[float]:
        """Estimate model size in billions of parameters"""
        
        # Extract size from model name/id
        model_name = model_id.split('/')[-1].lower()
        
        if "1.5b" in model_name or "1_5b" in model_name:
            return 1.5
        elif "3b" in model_name:
            return 3.0
        elif "7b" in model_name:
            return 7.0
        elif "12b" in model_name:
            return 12.0
        elif "14b" in model_name:
            return 14.0
        elif "27b" in model_name:
            return 27.0
        elif "32b" in model_name:
            return 32.0
        elif "70b" in model_name:
            return 70.0
        
        return None
    
    def get_degradation_recommendations(self, model_id: str) -> List[str]:
        """Get proactive degradation recommendations"""
        
        recommendations = []
        current_profile = self.memory_manager.get_current_profile()
        
        # Check memory pressure
        if current_profile.memory_pressure > 0.7:
            recommendations.append("Consider using a smaller model variant")
            recommendations.append("Enable aggressive quantization (int8)")
            
        if current_profile.memory_pressure > 0.8:
            recommendations.append("Use CPU fallback for better reliability")
            recommendations.append("Enable process isolation")
            
        # Model-specific recommendations
        estimated_size = self._estimate_model_size(model_id)
        if estimated_size and estimated_size > 20:
            recommendations.append("Large model detected - consider sequential processing")
            
        return recommendations
    
    def get_degradation_history(self) -> List[DegradationResult]:
        """Get history of degradation attempts"""
        return self.degradation_history.copy()

# Global degradation manager instance
_global_degradation_manager: Optional[GracefulDegradationManager] = None

def get_degradation_manager() -> GracefulDegradationManager:
    """Get global degradation manager instance"""
    global _global_degradation_manager
    
    if _global_degradation_manager is None:
        _global_degradation_manager = GracefulDegradationManager()
    
    return _global_degradation_manager