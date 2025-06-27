"""
Simplified Resource Management System

This module provides essential resource management with:
- Basic fallback mechanisms
- Simple process utilities
- Resource monitoring
"""

import logging
import time
import psutil
import subprocess
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SimpleResourceManager:
    """Simplified resource management system"""
    
    def __init__(self):
        self.fallback_strategies = ["reduce_batch_size", "enable_quantization", "fallback_to_cpu"]
        
    def check_resource_availability(self, required_memory_gb: float = 8.0) -> bool:
        """Check if required resources are available"""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < required_memory_gb:
                logger.warning(f"Insufficient memory: {available_gb:.1f}GB available, {required_memory_gb:.1f}GB required")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return False
    
    def apply_fallback_strategy(self, strategy: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a simple fallback strategy"""
        fallback_config = model_config.copy()
        
        if strategy == "reduce_batch_size":
            # Reduce batch size by half
            if "batch_size" in fallback_config:
                fallback_config["batch_size"] = max(1, fallback_config["batch_size"] // 2)
            logger.info("Applied fallback: reduced batch size")
        
        elif strategy == "enable_quantization":
            # Enable basic quantization
            fallback_config["quantization"] = "int8"
            logger.info("Applied fallback: enabled quantization")
        
        elif strategy == "fallback_to_cpu":
            # Force CPU usage
            fallback_config["device"] = "cpu"
            logger.info("Applied fallback: switched to CPU")
        
        return fallback_config
    
    def try_with_fallbacks(self, model_config: Dict[str, Any], operation_func: callable) -> Any:
        """Try operation with progressive fallbacks"""
        current_config = model_config.copy()
        
        # First try with original config
        try:
            return operation_func(current_config)
        except Exception as e:
            logger.warning(f"Operation failed with original config: {e}")
        
        # Try fallback strategies
        for strategy in self.fallback_strategies:
            try:
                logger.info(f"Trying fallback strategy: {strategy}")
                current_config = self.apply_fallback_strategy(strategy, current_config)
                return operation_func(current_config)
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy} failed: {e}")
                continue
        
        raise RuntimeError("All fallback strategies failed")

class SimpleProcessManager:
    """Simplified process management"""
    
    def __init__(self):
        self.active_processes: Dict[str, subprocess.Popen] = {}
    
    def run_isolated_command(self, command: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Run command in isolated process with timeout"""
        try:
            logger.info(f"Running isolated command: {' '.join(command)}")
            start_time = time.time()
            
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            result = {
                "success": process.returncode == 0,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "returncode": process.returncode,
                "execution_time": execution_time
            }
            
            if result["success"]:
                logger.info(f"Command completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"Command failed with return code {process.returncode}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "returncode": -1,
                "execution_time": timeout
            }
        except Exception as e:
            logger.error(f"Error running isolated command: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "execution_time": 0
            }
    
    def cleanup_processes(self):
        """Clean up any active processes"""
        for name, process in list(self.active_processes.items()):
            try:
                if process.poll() is None:  # Process still running
                    process.terminate()
                    process.wait(timeout=5)
                del self.active_processes[name]
                logger.info(f"Cleaned up process: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up process {name}: {e}")

# Global instances
_global_resource_manager = None
_global_process_manager = None

def get_resource_manager() -> SimpleResourceManager:
    """Get global resource manager instance"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = SimpleResourceManager()
    return _global_resource_manager

def get_process_manager() -> SimpleProcessManager:
    """Get global process manager instance"""
    global _global_process_manager
    if _global_process_manager is None:
        _global_process_manager = SimpleProcessManager()
    return _global_process_manager