"""
Advanced Memory Profiling and Optimization System

This module provides enterprise-grade memory profiling and optimization:
- Detailed memory usage analysis
- Performance bottleneck identification  
- Optimization recommendations
- Memory leak detection
- Resource utilization reporting
"""

import gc
import os
import sys
import time
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import torch
from collections import defaultdict, deque
import tracemalloc
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import settings
from utils.memory_manager import get_memory_manager, MemoryProfile

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Detailed memory snapshot at a point in time"""
    timestamp: datetime
    process_memory_gb: float
    gpu_memory_gb: float
    python_objects_count: int
    pytorch_tensors_count: int
    largest_objects: List[Tuple[str, float]]  # (object_type, size_mb)
    call_stack: Optional[str] = None
    operation_context: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    execution_time_ms: float
    memory_before_gb: float
    memory_after_gb: float
    memory_peak_gb: float
    memory_leaked_gb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    success: bool
    error_message: Optional[str] = None

@dataclass 
class OptimizationRecommendation:
    """Memory optimization recommendation"""
    category: str  # "memory", "performance", "configuration"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    impact_estimate: str
    implementation_difficulty: str  # "easy", "medium", "hard"
    code_example: Optional[str] = None
    related_metrics: List[str] = field(default_factory=list)

class MemoryProfiler:
    """Advanced memory profiler with optimization recommendations"""
    
    def __init__(self, enable_detailed_tracking: bool = True):
        self.enable_detailed_tracking = enable_detailed_tracking
        self.snapshots: deque = deque(maxlen=1000)
        self.performance_metrics: List[PerformanceMetrics] = []
        self.memory_manager = get_memory_manager()
        
        # Profiling state
        self.is_profiling = False
        self.profiling_thread: Optional[threading.Thread] = None
        self.baseline_memory: Optional[MemorySnapshot] = None
        
        # Analysis caches
        self._analysis_cache: Dict[str, Any] = {}
        self._recommendations_cache: List[OptimizationRecommendation] = []
        
        if enable_detailed_tracking:
            tracemalloc.start()
            
        logger.info("MemoryProfiler initialized")
    
    def start_profiling(self, interval_seconds: float = 1.0):
        """Start continuous memory profiling"""
        
        if self.is_profiling:
            logger.warning("Profiling already active")
            return
        
        self.is_profiling = True
        self.baseline_memory = self.take_snapshot("baseline")
        
        def profiling_loop():
            while self.is_profiling:
                try:
                    snapshot = self.take_snapshot("continuous")
                    self.snapshots.append(snapshot)
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in profiling loop: {e}")
                    break
        
        self.profiling_thread = threading.Thread(target=profiling_loop, daemon=True)
        self.profiling_thread.start()
        
        logger.info(f"Memory profiling started with {interval_seconds}s interval")
    
    def stop_profiling(self):
        """Stop continuous memory profiling"""
        
        if not self.is_profiling:
            return
        
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=5)
        
        logger.info("Memory profiling stopped")
    
    def take_snapshot(self, context: str = "manual") -> MemorySnapshot:
        """Take detailed memory snapshot"""
        
        timestamp = datetime.now()
        
        # Process memory
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        # GPU memory
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        
        # Python objects
        python_objects_count = len(gc.get_objects())
        
        # PyTorch tensors
        pytorch_tensors_count = 0
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                pytorch_tensors_count += 1
        
        # Largest objects (if detailed tracking enabled)
        largest_objects = []
        if self.enable_detailed_tracking and tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                for stat in top_stats[:10]:
                    size_mb = stat.size / (1024**2)
                    location = f"{stat.traceback.format()[-1] if stat.traceback else 'unknown'}"
                    largest_objects.append((location, size_mb))
                    
            except Exception as e:
                logger.debug(f"Error collecting detailed memory stats: {e}")
        
        # Call stack (if enabled)
        call_stack = None
        if self.enable_detailed_tracking:
            call_stack = ''.join(traceback.format_stack()[-5:])  # Last 5 frames
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            process_memory_gb=process_memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            python_objects_count=python_objects_count,
            pytorch_tensors_count=pytorch_tensors_count,
            largest_objects=largest_objects,
            call_stack=call_stack,
            operation_context=context
        )
        
        return snapshot
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling specific operations"""
        
        class OperationProfiler:
            def __init__(self, profiler: 'MemoryProfiler', name: str):
                self.profiler = profiler
                self.name = name
                self.start_time = None
                self.start_snapshot = None
                self.peak_memory = 0.0
                
            def __enter__(self):
                self.start_time = time.time()
                self.start_snapshot = self.profiler.take_snapshot(f"start_{self.name}")
                self.peak_memory = self.start_snapshot.process_memory_gb
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                end_snapshot = self.profiler.take_snapshot(f"end_{self.name}")
                
                # Calculate metrics
                execution_time_ms = (end_time - self.start_time) * 1000
                memory_leaked = end_snapshot.process_memory_gb - self.start_snapshot.process_memory_gb
                
                # Get current utilization
                gpu_util = 0.0
                if torch.cuda.is_available():
                    gpu_util = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                
                cpu_util = psutil.cpu_percent()
                
                metrics = PerformanceMetrics(
                    operation_name=self.name,
                    execution_time_ms=execution_time_ms,
                    memory_before_gb=self.start_snapshot.process_memory_gb,
                    memory_after_gb=end_snapshot.process_memory_gb,
                    memory_peak_gb=self.peak_memory,
                    memory_leaked_gb=memory_leaked,
                    gpu_utilization_percent=gpu_util,
                    cpu_utilization_percent=cpu_util,
                    success=exc_type is None,
                    error_message=str(exc_val) if exc_val else None
                )
                
                self.profiler.performance_metrics.append(metrics)
                logger.debug(f"Operation {self.name} profiled: {execution_time_ms:.1f}ms, "
                           f"memory delta: {memory_leaked:+.2f}GB")
        
        return OperationProfiler(self, operation_name)
    
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns and detect issues"""
        
        if len(self.snapshots) < 10:
            return {"error": "Insufficient data for analysis"}
        
        snapshots_list = list(self.snapshots)
        
        # Memory trend analysis
        memory_values = [s.process_memory_gb for s in snapshots_list]
        gpu_memory_values = [s.gpu_memory_gb for s in snapshots_list]
        
        # Detect memory leaks
        leak_detection = self._detect_memory_leaks(memory_values)
        
        # Fragmentation analysis
        fragmentation = self._analyze_fragmentation(snapshots_list)
        
        # Peak memory analysis
        peak_analysis = self._analyze_peak_memory(snapshots_list)
        
        # Object growth analysis
        object_growth = self._analyze_object_growth(snapshots_list)
        
        analysis = {
            "memory_trend": {
                "current_gb": memory_values[-1],
                "baseline_gb": memory_values[0],
                "peak_gb": max(memory_values),
                "growth_gb": memory_values[-1] - memory_values[0],
                "volatility": np.std(memory_values) if len(memory_values) > 1 else 0
            },
            "gpu_memory_trend": {
                "current_gb": gpu_memory_values[-1],
                "peak_gb": max(gpu_memory_values),
                "utilization_percent": (gpu_memory_values[-1] / torch.cuda.get_device_properties(0).total_memory * 1024**3) * 100 if torch.cuda.is_available() else 0
            },
            "leak_detection": leak_detection,
            "fragmentation_analysis": fragmentation,
            "peak_memory_analysis": peak_analysis,
            "object_growth_analysis": object_growth,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Cache analysis
        self._analysis_cache["latest"] = analysis
        
        return analysis
    
    def _detect_memory_leaks(self, memory_values: List[float]) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        
        if len(memory_values) < 20:
            return {"status": "insufficient_data"}
        
        # Simple trend analysis
        recent_values = memory_values[-10:]
        older_values = memory_values[-20:-10]
        
        recent_avg = sum(recent_values) / len(recent_values)
        older_avg = sum(older_values) / len(older_values)
        
        growth_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        
        # Leak indicators
        is_leak_suspected = growth_rate > 0.05  # 5% growth
        leak_severity = "low" if growth_rate < 0.1 else "medium" if growth_rate < 0.2 else "high"
        
        return {
            "status": "leak_suspected" if is_leak_suspected else "no_leak_detected",
            "growth_rate_percent": growth_rate * 100,
            "severity": leak_severity if is_leak_suspected else None,
            "recommendation": "Investigate memory cleanup in recent operations" if is_leak_suspected else None
        }
    
    def _analyze_fragmentation(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze memory fragmentation"""
        
        # Simple fragmentation analysis based on object count vs memory
        if len(snapshots) < 5:
            return {"status": "insufficient_data"}
        
        memory_efficiency = []
        for snapshot in snapshots[-10:]:
            if snapshot.python_objects_count > 0:
                efficiency = snapshot.process_memory_gb / (snapshot.python_objects_count / 1000000)  # GB per million objects
                memory_efficiency.append(efficiency)
        
        if not memory_efficiency:
            return {"status": "no_data"}
        
        avg_efficiency = sum(memory_efficiency) / len(memory_efficiency)
        efficiency_trend = memory_efficiency[-1] - memory_efficiency[0] if len(memory_efficiency) > 1 else 0
        
        return {
            "average_efficiency_gb_per_million_objects": avg_efficiency,
            "efficiency_trend": efficiency_trend,
            "fragmentation_level": "high" if avg_efficiency > 2.0 else "medium" if avg_efficiency > 1.0 else "low"
        }
    
    def _analyze_peak_memory(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze peak memory usage patterns"""
        
        memory_values = [s.process_memory_gb for s in snapshots]
        peak_memory = max(memory_values)
        peak_index = memory_values.index(peak_memory)
        peak_snapshot = snapshots[peak_index]
        
        return {
            "peak_memory_gb": peak_memory,
            "peak_timestamp": peak_snapshot.timestamp.isoformat(),
            "peak_context": peak_snapshot.operation_context,
            "peak_to_current_ratio": peak_memory / memory_values[-1] if memory_values[-1] > 0 else float('inf')
        }
    
    def _analyze_object_growth(self, snapshots: List[MemorySnapshot]) -> Dict[str, Any]:
        """Analyze Python object and tensor growth"""
        
        if len(snapshots) < 5:
            return {"status": "insufficient_data"}
        
        object_counts = [s.python_objects_count for s in snapshots]
        tensor_counts = [s.pytorch_tensors_count for s in snapshots]
        
        object_growth = object_counts[-1] - object_counts[0]
        tensor_growth = tensor_counts[-1] - tensor_counts[0]
        
        return {
            "python_objects_growth": object_growth,
            "pytorch_tensors_growth": tensor_growth,
            "current_python_objects": object_counts[-1],
            "current_pytorch_tensors": tensor_counts[-1],
            "object_growth_rate": object_growth / len(snapshots) if len(snapshots) > 0 else 0
        }
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis"""
        
        recommendations = []
        analysis = self.analyze_memory_patterns()
        
        if "error" in analysis:
            return recommendations
        
        # Memory leak recommendations
        if analysis["leak_detection"]["status"] == "leak_suspected":
            severity = analysis["leak_detection"]["severity"]
            priority = "critical" if severity == "high" else "high"
            
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=priority,
                title="Potential Memory Leak Detected",
                description=f"Memory growth rate of {analysis['leak_detection']['growth_rate_percent']:.1f}% detected. "
                           f"Investigate cleanup routines and ensure proper resource deallocation.",
                impact_estimate="High - could lead to OOM errors",
                implementation_difficulty="medium",
                code_example="""
# Ensure proper cleanup
with agent_session(agent):
    result = agent.process()
    # Automatic cleanup on exit

# Or explicit cleanup
agent.cleanup()
gc.collect()
torch.cuda.empty_cache()
""",
                related_metrics=["memory_growth", "cleanup_effectiveness"]
            ))
        
        # High memory usage recommendations
        if analysis["memory_trend"]["current_gb"] > 16.0:  # 16GB threshold
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high",
                title="High Memory Usage Detected",
                description=f"Current memory usage is {analysis['memory_trend']['current_gb']:.1f}GB. "
                           f"Consider using smaller models or enabling more aggressive cleanup.",
                impact_estimate="Medium - may cause performance degradation",
                implementation_difficulty="easy",
                code_example="""
# Use smaller model variants
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # instead of 70B

# Enable aggressive quantization
quantization = "int8"  # instead of fp16

# Use sequential processing
settings.force_sequential_processing = True
""",
                related_metrics=["current_memory", "peak_memory"]
            ))
        
        # GPU utilization recommendations
        gpu_util = analysis["gpu_memory_trend"]["utilization_percent"]
        if gpu_util > 90:
            recommendations.append(OptimizationRecommendation(
                category="performance",
                priority="high",
                title="High GPU Memory Utilization",
                description=f"GPU memory utilization is {gpu_util:.1f}%. Risk of OOM errors. "
                           f"Consider reducing batch size or using CPU fallback.",
                impact_estimate="High - may cause CUDA OOM errors",
                implementation_difficulty="easy",
                code_example="""
# Enable CPU fallback
settings.enable_cpu_fallback = True

# Reduce memory per model
settings.max_gpu_memory_per_model = 12.0  # GB

# Use more aggressive cleanup
settings.memory_strategy = "aggressive"
""",
                related_metrics=["gpu_utilization", "gpu_memory"]
            ))
        
        # Performance recommendations based on execution times
        if self.performance_metrics:
            avg_execution_time = sum(m.execution_time_ms for m in self.performance_metrics) / len(self.performance_metrics)
            
            if avg_execution_time > 10000:  # 10 seconds
                recommendations.append(OptimizationRecommendation(
                    category="performance",
                    priority="medium",
                    title="Slow Operation Performance",
                    description=f"Average operation time is {avg_execution_time/1000:.1f}s. "
                               f"Consider optimizing model loading or using caching.",
                    impact_estimate="Medium - affects user experience",
                    implementation_difficulty="medium",
                    code_example="""
# Enable model caching
settings.enable_model_caching = True

# Use faster quantization
quantization = "fp8"  # faster than int8

# Optimize batch processing
settings.auto_optimize_batch_size = True
""",
                    related_metrics=["execution_time", "throughput"]
                ))
        
        # Object growth recommendations
        if analysis["object_growth_analysis"]["python_objects_growth"] > 100000:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="medium",
                title="High Object Growth Rate",
                description=f"Python object count grew by {analysis['object_growth_analysis']['python_objects_growth']} objects. "
                           f"Consider more frequent garbage collection.",
                impact_estimate="Medium - may lead to memory fragmentation",
                implementation_difficulty="easy",
                code_example="""
# More frequent garbage collection
import gc
gc.collect()

# Limit object creation in loops
for item in items:
    with limited_scope():
        process(item)
""",
                related_metrics=["object_count", "fragmentation"]
            ))
        
        # Cache recommendations
        self._recommendations_cache = recommendations
        
        return recommendations
    
    def export_profiling_report(self, output_path: str = "./memory_profiling_report.json") -> str:
        """Export comprehensive profiling report"""
        
        analysis = self.analyze_memory_patterns()
        recommendations = self.generate_optimization_recommendations()
        
        report = {
            "profiling_summary": {
                "snapshots_collected": len(self.snapshots),
                "operations_profiled": len(self.performance_metrics),
                "profiling_duration_hours": (datetime.now() - self.baseline_memory.timestamp).total_seconds() / 3600 if self.baseline_memory else 0,
                "generated_at": datetime.now().isoformat()
            },
            "memory_analysis": analysis,
            "performance_metrics": [
                {
                    "operation": m.operation_name,
                    "execution_time_ms": m.execution_time_ms,
                    "memory_delta_gb": m.memory_after_gb - m.memory_before_gb,
                    "success": m.success
                } 
                for m in self.performance_metrics[-50:]  # Last 50 operations
            ],
            "optimization_recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority,
                    "title": r.title,
                    "description": r.description,
                    "impact": r.impact_estimate
                }
                for r in recommendations
            ],
            "system_info": {
                "python_version": sys.version,
                "pytorch_version": torch.__version__ if torch else None,
                "cuda_available": torch.cuda.is_available() if torch else False,
                "total_system_memory_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_count": psutil.cpu_count()
            }
        }
        
        # Write report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Profiling report exported to {output_path}")
        return str(output_file)
    
    def cleanup(self):
        """Cleanup profiler resources"""
        
        self.stop_profiling()
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Clear caches
        self._analysis_cache.clear()
        self._recommendations_cache.clear()
        
        logger.info("MemoryProfiler cleaned up")

# Global profiler instance
_global_profiler: Optional[MemoryProfiler] = None

def get_profiler(enable_detailed_tracking: bool = True) -> MemoryProfiler:
    """Get global memory profiler instance"""
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = MemoryProfiler(enable_detailed_tracking)
    
    return _global_profiler

def profile_operation(operation_name: str):
    """Convenience function for profiling operations"""
    return get_profiler().profile_operation(operation_name)