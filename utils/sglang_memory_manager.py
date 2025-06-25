"""
SGLang-optimized Memory Management System
Extends the existing memory management with SGLang RadixAttention optimization
"""

import gc
import os
import sys
import time
import psutil
import subprocess
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import logging
import torch
from pathlib import Path

from utils.memory_manager import MemoryStrategy, DeviceType, MemoryProfile, get_memory_profile

logger = logging.getLogger(__name__)

@dataclass
class SGLangMemoryProfile(MemoryProfile):
    """Extended memory profile for SGLang with RadixAttention metrics"""
    radix_cache_size_gb: float = 0.0
    kv_cache_size_gb: float = 0.0
    active_requests: int = 0
    cached_prefixes: int = 0
    cache_hit_rate: float = 0.0
    server_processes: List[int] = field(default_factory=list)

class SGLangMemoryManager:
    """SGLang-optimized memory manager with RadixAttention awareness"""
    
    def __init__(self, 
                 strategy: MemoryStrategy = MemoryStrategy.BALANCED,
                 max_gpu_memory_per_model: float = 24.0,
                 radix_cache_limit_gb: float = 8.0):
        """
        Initialize SGLang memory manager
        
        Args:
            strategy: Memory management strategy
            max_gpu_memory_per_model: Maximum GPU memory per model in GB
            radix_cache_limit_gb: Maximum RadixAttention cache size in GB
        """
        self.strategy = strategy
        self.max_gpu_memory_per_model = max_gpu_memory_per_model
        self.radix_cache_limit_gb = radix_cache_limit_gb
        
        # SGLang-specific tracking
        self._active_servers: Dict[int, Dict[str, Any]] = {}  # port -> server info
        self._radix_stats: Dict[int, Dict[str, float]] = {}  # port -> cache stats
        self._lock = threading.RLock()
        
        # Memory thresholds for SGLang
        self._memory_thresholds = {
            MemoryStrategy.AGGRESSIVE: {
                'gpu_utilization_limit': 0.7,
                'system_memory_limit': 0.6,
                'radix_cache_limit': 0.5
            },
            MemoryStrategy.BALANCED: {
                'gpu_utilization_limit': 0.8,
                'system_memory_limit': 0.7,
                'radix_cache_limit': 0.7
            },
            MemoryStrategy.PERFORMANCE: {
                'gpu_utilization_limit': 0.9,
                'system_memory_limit': 0.8,
                'radix_cache_limit': 0.8
            },
            MemoryStrategy.ADAPTIVE: {
                'gpu_utilization_limit': 0.85,
                'system_memory_limit': 0.75,
                'radix_cache_limit': 0.6
            }
        }
        
        logger.info(f"SGLang Memory Manager initialized with {strategy.value} strategy")
    
    def get_sglang_memory_profile(self) -> SGLangMemoryProfile:
        """Get comprehensive SGLang memory profile including RadixAttention metrics"""
        
        # Get base memory profile
        base_profile = get_memory_profile()
        
        # Extend with SGLang-specific metrics
        radix_cache_size = self._calculate_radix_cache_size()
        kv_cache_size = self._calculate_kv_cache_size()
        active_requests = self._count_active_requests()
        cached_prefixes = self._count_cached_prefixes()
        cache_hit_rate = self._calculate_cache_hit_rate()
        server_processes = self._get_active_server_processes()
        
        return SGLangMemoryProfile(
            total_system_gb=base_profile.total_system_gb,
            available_system_gb=base_profile.available_system_gb,
            used_system_gb=base_profile.used_system_gb,
            gpu_available=base_profile.gpu_available,
            gpu_total_gb=base_profile.gpu_total_gb,
            gpu_allocated_gb=base_profile.gpu_allocated_gb,
            radix_cache_size_gb=radix_cache_size,
            kv_cache_size_gb=kv_cache_size,
            active_requests=active_requests,
            cached_prefixes=cached_prefixes,
            cache_hit_rate=cache_hit_rate,
            server_processes=server_processes
        )
    
    def register_sglang_server(self, port: int, model_id: str, config: Dict[str, Any]):
        """Register an active SGLang server for memory tracking"""
        
        with self._lock:
            self._active_servers[port] = {
                'model_id': model_id,
                'config': config,
                'start_time': time.time(),
                'memory_allocated': 0.0,
                'last_health_check': time.time()
            }
            
            # Initialize cache stats
            self._radix_stats[port] = {
                'cache_size_gb': 0.0,
                'cache_hits': 0,
                'cache_misses': 0,
                'active_requests': 0
            }
            
            logger.info(f"Registered SGLang server on port {port} for model {model_id}")
    
    def unregister_sglang_server(self, port: int):
        """Unregister an SGLang server"""
        
        with self._lock:
            if port in self._active_servers:
                del self._active_servers[port]
                logger.info(f"Unregistered SGLang server on port {port}")
            
            if port in self._radix_stats:
                del self._radix_stats[port]
    
    def update_radix_stats(self, port: int, stats: Dict[str, Any]):
        """Update RadixAttention cache statistics for a server"""
        
        with self._lock:
            if port in self._radix_stats:
                self._radix_stats[port].update(stats)
    
    def check_memory_constraints(self, model_name: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if memory constraints allow loading a new SGLang model
        
        Returns:
            (can_load, reason)
        """
        
        profile = self.get_sglang_memory_profile()
        thresholds = self._memory_thresholds[self.strategy]
        
        # Check GPU memory
        if profile.gpu_available:
            gpu_utilization = profile.gpu_allocated_gb / profile.gpu_total_gb
            if gpu_utilization > thresholds['gpu_utilization_limit']:
                return False, f"GPU utilization {gpu_utilization:.2f} exceeds limit {thresholds['gpu_utilization_limit']:.2f}"
        
        # Check system memory
        system_utilization = profile.used_system_gb / profile.total_system_gb
        if system_utilization > thresholds['system_memory_limit']:
            return False, f"System memory utilization {system_utilization:.2f} exceeds limit {thresholds['system_memory_limit']:.2f}"
        
        # Check RadixAttention cache size
        radix_utilization = profile.radix_cache_size_gb / self.radix_cache_limit_gb
        if radix_utilization > thresholds['radix_cache_limit']:
            return False, f"RadixAttention cache utilization {radix_utilization:.2f} exceeds limit {thresholds['radix_cache_limit']:.2f}"
        
        # Check if we have too many active servers
        if len(self._active_servers) >= self._get_max_concurrent_servers():
            return False, f"Maximum concurrent servers ({len(self._active_servers)}) reached"
        
        return True, "Memory constraints satisfied"
    
    def optimize_radix_cache(self) -> Dict[str, Any]:
        """Optimize RadixAttention cache across all active servers"""
        
        optimization_results = {
            'servers_optimized': 0,
            'cache_freed_gb': 0.0,
            'prefixes_evicted': 0,
            'actions_taken': []
        }
        
        profile = self.get_sglang_memory_profile()
        
        # If cache usage is high, trigger optimization
        if profile.radix_cache_size_gb > self.radix_cache_limit_gb * 0.8:
            logger.info("RadixAttention cache usage high, triggering optimization")
            
            with self._lock:
                for port, server_info in self._active_servers.items():
                    try:
                        # Call SGLang server API to optimize cache
                        cache_freed = self._optimize_server_cache(port)
                        
                        if cache_freed > 0:
                            optimization_results['servers_optimized'] += 1
                            optimization_results['cache_freed_gb'] += cache_freed
                            optimization_results['actions_taken'].append(f"Optimized cache on port {port}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to optimize cache for server on port {port}: {e}")
                        optimization_results['actions_taken'].append(f"Failed to optimize port {port}: {e}")
        
        return optimization_results
    
    def cleanup_sglang_resources(self) -> Dict[str, Any]:
        """Comprehensive cleanup of SGLang resources"""
        
        cleanup_results = {
            'servers_stopped': 0,
            'cache_cleared_gb': 0.0,
            'processes_killed': 0,
            'actions_taken': []
        }
        
        logger.info("Starting comprehensive SGLang resource cleanup")
        
        with self._lock:
            # Stop all registered servers
            for port in list(self._active_servers.keys()):
                try:
                    self._stop_sglang_server(port)
                    cleanup_results['servers_stopped'] += 1
                    cleanup_results['actions_taken'].append(f"Stopped server on port {port}")
                except Exception as e:
                    logger.error(f"Error stopping server on port {port}: {e}")
                    cleanup_results['actions_taken'].append(f"Failed to stop port {port}: {e}")
            
            # Clear tracking data
            self._active_servers.clear()
            self._radix_stats.clear()
        
        # Kill any remaining SGLang processes
        killed_processes = self._kill_remaining_sglang_processes()
        cleanup_results['processes_killed'] = killed_processes
        
        # Aggressive memory cleanup
        self._aggressive_memory_cleanup()
        
        # Update cache cleared
        cleanup_results['cache_cleared_gb'] = self._calculate_radix_cache_size()
        
        logger.info(f"SGLang cleanup completed: {cleanup_results}")
        return cleanup_results
    
    def _calculate_radix_cache_size(self) -> float:
        """Calculate total RadixAttention cache size across all servers"""
        
        total_cache_gb = 0.0
        
        with self._lock:
            for port, stats in self._radix_stats.items():
                total_cache_gb += stats.get('cache_size_gb', 0.0)
        
        return total_cache_gb
    
    def _calculate_kv_cache_size(self) -> float:
        """Estimate KV cache size from GPU memory usage"""
        
        if not torch.cuda.is_available():
            return 0.0
        
        # Rough estimation: KV cache is typically 20-30% of allocated GPU memory
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        return allocated_gb * 0.25
    
    def _count_active_requests(self) -> int:
        """Count active requests across all servers"""
        
        total_requests = 0
        
        with self._lock:
            for port, stats in self._radix_stats.items():
                total_requests += stats.get('active_requests', 0)
        
        return total_requests
    
    def _count_cached_prefixes(self) -> int:
        """Count cached prefixes in RadixAttention cache"""
        
        # This would require API calls to SGLang servers
        # For now, return estimated count based on cache size
        cache_size_gb = self._calculate_radix_cache_size()
        
        # Rough estimation: 1MB per cached prefix
        return int(cache_size_gb * 1024)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate across servers"""
        
        total_hits = 0
        total_requests = 0
        
        with self._lock:
            for port, stats in self._radix_stats.items():
                total_hits += stats.get('cache_hits', 0)
                total_requests += stats.get('cache_hits', 0) + stats.get('cache_misses', 0)
        
        if total_requests == 0:
            return 0.0
        
        return total_hits / total_requests
    
    def _get_active_server_processes(self) -> List[int]:
        """Get list of active SGLang server process IDs"""
        
        processes = []
        
        try:
            # Find SGLang server processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'sglang.launch_server' in cmdline:
                        processes.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.warning(f"Error getting SGLang processes: {e}")
        
        return processes
    
    def _get_max_concurrent_servers(self) -> int:
        """Get maximum number of concurrent SGLang servers based on strategy"""
        
        strategy_limits = {
            MemoryStrategy.AGGRESSIVE: 1,
            MemoryStrategy.BALANCED: 2,
            MemoryStrategy.PERFORMANCE: 3,
            MemoryStrategy.ADAPTIVE: 2
        }
        
        return strategy_limits.get(self.strategy, 2)
    
    def _optimize_server_cache(self, port: int) -> float:
        """Optimize cache for a specific SGLang server"""
        
        try:
            import requests
            
            # Call SGLang server optimization endpoint (if available)
            response = requests.post(
                f"http://127.0.0.1:{port}/optimize_cache",
                timeout=10,
                json={"strategy": "lru_eviction", "target_reduction": 0.3}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('cache_freed_gb', 0.0)
            
        except Exception as e:
            logger.debug(f"Cache optimization API call failed for port {port}: {e}")
        
        return 0.0
    
    def _stop_sglang_server(self, port: int):
        """Stop SGLang server on specific port"""
        
        try:
            import requests
            
            # Try graceful shutdown
            requests.post(f"http://127.0.0.1:{port}/shutdown", timeout=5)
            time.sleep(2)
            
        except Exception:
            pass
        
        # Force kill process using the port
        try:
            subprocess.run(
                ['pkill', '-f', f'--port {port}'],
                check=False, timeout=3,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        
        # Unregister the server
        self.unregister_sglang_server(port)
    
    def _kill_remaining_sglang_processes(self) -> int:
        """Kill any remaining SGLang processes"""
        
        killed_count = 0
        
        sglang_process_patterns = [
            'sglang.launch_server',
            'python -m sglang',
            'SGLang',
            'sglang'
        ]
        
        for pattern in sglang_process_patterns:
            try:
                result = subprocess.run(
                    ['pkill', '-f', pattern],
                    check=False, timeout=3,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if result.returncode == 0:
                    killed_count += 1
                    
            except Exception:
                pass
        
        return killed_count
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup optimized for SGLang"""
        
        logger.info("Starting aggressive SGLang memory cleanup")
        
        # Multiple rounds of garbage collection
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC round {i+1}: {collected} objects collected")
        
        if torch.cuda.is_available():
            # CUDA memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
            # Additional cleanup for stubborn memory
            try:
                if hasattr(torch.cuda, 'memory_allocated'):
                    current_memory = torch.cuda.memory_allocated()
                    logger.debug(f"GPU memory after cleanup: {current_memory / 1024**3:.2f} GB")
            except Exception as e:
                logger.debug(f"Memory stat collection failed: {e}")
    
    @contextmanager
    def sglang_memory_context(self, model_name: str, config: Dict[str, Any]):
        """Context manager for SGLang model loading with memory management"""
        
        can_load, reason = self.check_memory_constraints(model_name, config)
        
        if not can_load:
            logger.warning(f"Memory constraints prevent loading {model_name}: {reason}")
            
            # Try optimization
            opt_results = self.optimize_radix_cache()
            logger.info(f"Cache optimization results: {opt_results}")
            
            # Recheck after optimization
            can_load, reason = self.check_memory_constraints(model_name, config)
            
            if not can_load:
                raise RuntimeError(f"Insufficient memory to load {model_name}: {reason}")
        
        # Track memory before loading
        profile_before = self.get_sglang_memory_profile()
        
        try:
            yield self
            
        finally:
            # Track memory after and cleanup if needed
            profile_after = self.get_sglang_memory_profile()
            
            memory_increase = profile_after.gpu_allocated_gb - profile_before.gpu_allocated_gb
            logger.info(f"Memory usage increased by {memory_increase:.2f} GB during model operation")
            
            # If memory usage is high, trigger optimization
            if profile_after.gpu_allocated_gb > self.max_gpu_memory_per_model * 0.8:
                self.optimize_radix_cache()

# Global SGLang memory manager
_sglang_memory_manager = None

def get_sglang_memory_manager(strategy: MemoryStrategy = MemoryStrategy.BALANCED) -> SGLangMemoryManager:
    """Get global SGLang memory manager instance"""
    
    global _sglang_memory_manager
    
    if _sglang_memory_manager is None:
        _sglang_memory_manager = SGLangMemoryManager(strategy=strategy)
    
    return _sglang_memory_manager