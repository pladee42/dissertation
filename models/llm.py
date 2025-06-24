from dotenv import load_dotenv
import gc
import numpy as np
import os
import subprocess
import torch
import time
import logging
import atexit
import signal
import sys
import threading
import uuid
from contextlib import contextmanager
from typing import Dict, Optional, Union, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from vllm import LLM, SamplingParams
from utils.retry import retry_with_backoff
from config.settings import settings

# Load environment variables
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstanceMetadata:
    """Enhanced metadata for model instances"""
    instance_id: str
    model_id: str
    model_name: str
    created_at: datetime
    dtype: str
    quantization: Optional[str]
    memory_allocated_gb: float
    device_type: str
    process_id: int
    thread_id: int
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0
    cleanup_callbacks: List[callable] = field(default_factory=list)

class GlobalInstanceTracker:
    """Enhanced global instance tracking with coordination"""
    
    def __init__(self):
        self._instances: Dict[str, InstanceMetadata] = {}
        self._lock = threading.RLock()
        self._cleanup_in_progress = False
        self._max_instances = getattr(settings, 'max_concurrent_models', 2)
        self._memory_budget_gb = getattr(settings, 'max_gpu_memory_per_model', 24.0)
        
        # Register cleanup handlers only for main process
        if os.getpid() == os.getpid():  # Simple check, but we need more sophisticated detection
            try:
                # Only register in main process, not in VLLM worker processes
                import multiprocessing
                if multiprocessing.current_process().name == 'MainProcess':
                    atexit.register(self.cleanup_all_instances)
                    signal.signal(signal.SIGTERM, self._signal_handler)
                    signal.signal(signal.SIGINT, self._signal_handler)
                    logger.debug("Cleanup handlers registered for main process")
                else:
                    logger.debug("Skipping cleanup handler registration for worker process")
            except Exception as e:
                logger.debug(f"Error registering cleanup handlers: {e}")
        
        logger.info("GlobalInstanceTracker initialized")
    
    def register_instance(self, 
                         instance: 'ModelInference',
                         memory_allocated_gb: float = 0.0) -> str:
        """Register a new model instance with enhanced metadata"""
        
        with self._lock:
            instance_id = str(uuid.uuid4())
            
            metadata = InstanceMetadata(
                instance_id=instance_id,
                model_id=instance.model_id,
                model_name=instance.model_name,
                created_at=datetime.now(),
                dtype=getattr(instance, 'dtype', 'unknown'),
                quantization=getattr(instance, 'quantization', None),
                memory_allocated_gb=memory_allocated_gb,
                device_type="gpu" if torch.cuda.is_available() else "cpu",
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
                last_used=datetime.now()
            )
            
            self._instances[instance_id] = metadata
            
            logger.info(f"Registered instance {instance_id[:8]} for model {metadata.model_name}")
            
            # Check if we exceed limits
            self._enforce_instance_limits()
            
            return instance_id
    
    def update_instance_usage(self, instance_id: str):
        """Update instance usage statistics"""
        
        with self._lock:
            if instance_id in self._instances:
                metadata = self._instances[instance_id]
                metadata.last_used = datetime.now()
                metadata.usage_count += 1
    
    def deregister_instance(self, instance_id: str):
        """Deregister an instance"""
        
        with self._lock:
            if instance_id in self._instances:
                metadata = self._instances[instance_id]
                metadata.is_active = False
                
                # Execute cleanup callbacks
                for callback in metadata.cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.warning(f"Cleanup callback failed for {instance_id}: {e}")
                
                del self._instances[instance_id]
                logger.info(f"Deregistered instance {instance_id[:8]}")
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Get comprehensive instance statistics"""
        
        with self._lock:
            active_instances = [m for m in self._instances.values() if m.is_active]
            
            total_memory = sum(m.memory_allocated_gb for m in active_instances)
            
            by_model = {}
            for metadata in active_instances:
                model_name = metadata.model_name
                if model_name not in by_model:
                    by_model[model_name] = {"count": 0, "memory_gb": 0.0}
                by_model[model_name]["count"] += 1
                by_model[model_name]["memory_gb"] += metadata.memory_allocated_gb
            
            return {
                "total_instances": len(active_instances),
                "total_memory_gb": total_memory,
                "max_instances": self._max_instances,
                "memory_budget_gb": self._memory_budget_gb,
                "memory_utilization": total_memory / self._memory_budget_gb if self._memory_budget_gb > 0 else 0,
                "instances_by_model": by_model,
                "oldest_instance": min(active_instances, key=lambda m: m.created_at).created_at if active_instances else None,
                "cleanup_in_progress": self._cleanup_in_progress
            }
    
    def _enforce_instance_limits(self):
        """Enforce instance limits by cleaning up oldest instances"""
        
        active_instances = [m for m in self._instances.values() if m.is_active]
        
        # Check instance count limit
        if len(active_instances) > self._max_instances:
            logger.warning(f"Instance limit ({self._max_instances}) exceeded, cleaning up oldest instances")
            
            # Sort by last used (oldest first)
            sorted_instances = sorted(active_instances, key=lambda m: m.last_used or m.created_at)
            
            instances_to_cleanup = len(active_instances) - self._max_instances
            for metadata in sorted_instances[:instances_to_cleanup]:
                logger.info(f"Auto-cleanup of instance {metadata.instance_id[:8]} due to limit")
                self.deregister_instance(metadata.instance_id)
        
        # Check memory limit
        total_memory = sum(m.memory_allocated_gb for m in active_instances)
        if total_memory > self._memory_budget_gb:
            logger.warning(f"Memory budget ({self._memory_budget_gb}GB) exceeded: {total_memory:.2f}GB")
    
    def cleanup_all_instances(self):
        """Cleanup all tracked instances with coordination"""
        
        with self._lock:
            if self._cleanup_in_progress:
                logger.debug("Global cleanup already in progress, skipping")
                return
            
            self._cleanup_in_progress = True
            
        try:
            logger.info("Starting global instance cleanup")
            
            instance_ids = list(self._instances.keys())
            logger.info(f"Cleaning up {len(instance_ids)} instances")
            
            for instance_id in instance_ids:
                try:
                    logger.debug(f"Cleaning up instance {instance_id[:8]}")
                    self.deregister_instance(instance_id)
                except Exception as e:
                    logger.error(f"Error cleaning up instance {instance_id}: {e}")
            
            self._instances.clear()
            logger.info("Global instance cleanup completed")
            
        finally:
            with self._lock:
                self._cleanup_in_progress = False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating cleanup")
        self.cleanup_all_instances()
        sys.exit(0)

# Global instance tracker
_global_tracker = GlobalInstanceTracker()

# Global cleanup coordination
_global_cleanup_lock = threading.Lock()
_global_cleanup_in_progress = False

class ModelInference:
    _instances = []  # Legacy tracking for backward compatibility
    
    def __init__(self,
                 model_id: str,
                 dtype: str = "bfloat16",
                 task: str = "generate",
                 quantization: str = None,
                 custom_config: Optional[Dict[str, Any]] = None):
        """Initialize the model once and keep it in memory with enhanced configuration"""
        
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.dtype = dtype
        self.quantization = quantization
        self.task = task
        self.is_cleaned_up = False
        self.instance_id = None
        
        # Legacy tracking for backward compatibility
        ModelInference._instances.append(self)
        
        logger.info(f"Loading model {model_id}...")
        start_time = time.time()
        
        # Merge default config with custom config
        vllm_config = {
            "model": model_id,
            "dtype": dtype,
            "download_dir": settings.download_dir,
            "tensor_parallel_size": torch.cuda.device_count(),
            "gpu_memory_utilization": 0.9,
            "quantization": quantization,
            "task": task,
            "trust_remote_code": True,
            # Performance optimizations
            "enable_prefix_caching": True,  # Speedup repeated prompts
            "disable_log_stats": True,      # Reduce overhead
            "enable_chunked_prefill": True, # Better memory management
            "max_model_len": 4096,          # Limit context if possible
            "swap_space": 4,                # 4 GB of CPU swap space
            "cpu_offload_gb": 0,           # CPU offloading if needed
            "max_num_seqs": 256,           # Limit concurrent sequences
            "block_size": 16,              # Optimize memory allocation
            # Additional cleanup-friendly settings
            "enforce_eager": False,         # Use CUDA graphs when possible
            "disable_custom_all_reduce": False,  # Better cleanup
        }
        
        if custom_config:
            vllm_config.update(custom_config)
        
        # Load the model with error handling
        try:
            self.llm = LLM(**vllm_config)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
        
        # Default sampling parameters
        self.default_params = SamplingParams(
            temperature=settings.default_temperature,
            top_p=settings.default_top_p,
            repetition_penalty=settings.default_top_p,
            max_tokens=settings.default_max_tokens
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model {self.model_name} loaded in {load_time:.2f} seconds")
        
        # Register with enhanced tracker
        memory_allocated = 0.0
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        
        self.instance_id = _global_tracker.register_instance(self, memory_allocated)
    
    def cleanup(self):
        """Explicitly cleanup the model and associated processes with enhanced VLLM support"""
        global _global_cleanup_in_progress
        
        if self.is_cleaned_up:
            return
        
        # Check if global cleanup is in progress
        with _global_cleanup_lock:
            if _global_cleanup_in_progress:
                logger.info(f"Global cleanup in progress, skipping individual cleanup for {self.model_name}")
                self.is_cleaned_up = True
                return
        
        # Log memory usage before cleanup
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
            logger.info(f"GPU memory before cleanup of {self.model_name}: {memory_before / 1024**3:.2f} GB")
            
        logger.info(f"Starting comprehensive cleanup for model {self.model_name}")
        self.is_cleaned_up = True
        
        try:
            # Enhanced VLLM cleanup with proper coordination
            if hasattr(self, 'llm'):
                try:
                    logger.info("Starting VLLM model cleanup")
                    
                    # First, try graceful VLLM shutdown
                    try:
                        # VLLM models don't have a direct shutdown method, but we can destroy the object
                        logger.info("Destroying VLLM model instance")
                        del self.llm
                        logger.info("VLLM model instance destroyed")
                    except Exception as e:
                        logger.warning(f"Error destroying VLLM model: {e}")
                    
                    # Force Ray cleanup after model destruction
                    self._force_ray_cleanup()
                    
                except Exception as e:
                    logger.warning(f"VLLM cleanup failed: {e}")
            
            # Remove from both tracking systems
            if self in ModelInference._instances:
                ModelInference._instances.remove(self)
                logger.info("Removed from legacy instances list")
            
            # Deregister from enhanced tracker
            if self.instance_id:
                _global_tracker.deregister_instance(self.instance_id)
                self.instance_id = None
            
            # Aggressive memory cleanup sequence
            self._aggressive_memory_cleanup()
            
            # Terminate background processes (but not Ray workers immediately)
            self._terminate_background_processes_safe()
            
            # Final memory verification
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_freed = (memory_before - memory_after) / 1024**3
                logger.info(f"GPU memory after cleanup: {memory_after / 1024**3:.2f} GB (freed: {memory_freed:.2f} GB)")
            
            logger.info(f"Cleanup completed successfully for {self.model_name}")
            
        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}")
            # Even if cleanup fails, mark as cleaned to prevent loops
            self.is_cleaned_up = True
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup with multiple techniques"""
        logger.info("Starting aggressive memory cleanup sequence")
        
        # Multiple rounds of garbage collection
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"Garbage collection round {i+1}: {collected} objects collected")
        
        if torch.cuda.is_available():
            # Multiple CUDA cleanup operations
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Reset memory stats to get accurate readings
            torch.cuda.reset_peak_memory_stats()
            
            # Additional cleanup for stubborn memory
            try:
                # Force memory defragmentation if available
                if hasattr(torch.cuda, 'memory_allocated'):
                    current_memory = torch.cuda.memory_allocated()
                    if current_memory > 0:
                        logger.debug(f"Remaining GPU memory after cleanup: {current_memory / 1024**3:.2f} GB")
            except Exception as e:
                logger.debug(f"Memory stat collection failed: {e}")
    
    def _force_ray_cleanup(self):
        """Force Ray cleanup with timeout and coordination"""
        logger.info("Starting coordinated Ray cleanup")
        
        try:
            import ray
            
            # Check if Ray is initialized
            if ray.is_initialized():
                logger.info("Ray is initialized, attempting shutdown")
                
                # Give a moment for VLLM workers to finish
                time.sleep(2)
                
                # Shutdown Ray with timeout using threading
                try:
                    import threading
                    
                    shutdown_success = [False]  # Use list for mutable reference
                    
                    def shutdown_ray():
                        try:
                            ray.shutdown()
                            shutdown_success[0] = True
                            logger.info("Ray shutdown completed successfully")
                        except Exception as e:
                            logger.warning(f"Ray shutdown failed in thread: {e}")
                    
                    # Start shutdown in separate thread
                    shutdown_thread = threading.Thread(target=shutdown_ray, daemon=True)
                    shutdown_thread.start()
                    
                    # Wait for shutdown with timeout
                    shutdown_thread.join(timeout=10.0)
                    
                    if shutdown_thread.is_alive() or not shutdown_success[0]:
                        logger.warning("Ray shutdown timed out or failed, forcing process termination")
                        self._force_kill_ray_processes()
                    
                except Exception as e:
                    logger.warning(f"Ray shutdown coordination failed: {e}, attempting force cleanup")
                    self._force_kill_ray_processes()
            else:
                logger.info("Ray not initialized, skipping Ray cleanup")
                
        except ImportError:
            logger.debug("Ray not available, skipping Ray cleanup")
        except Exception as e:
            logger.warning(f"Error during Ray cleanup: {e}")
    
    def _force_kill_ray_processes(self):
        """Force kill Ray processes if graceful shutdown fails"""
        logger.warning("Force killing Ray processes")
        
        ray_processes = [
            'ray::IDLE',
            'ray::CoreWorker', 
            'ray::RayletMonitor',
            'ray::NodeManager',
            'ray::ObjectManager',
            'raylet',
            'gcs_server'
        ]
        
        for process_name in ray_processes:
            try:
                subprocess.run(['pkill', '-9', '-f', process_name], 
                             check=False, timeout=3,
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            except Exception:
                pass
        
        # Also kill any remaining VLLM workers
        try:
            subprocess.run(['pkill', '-9', '-f', 'VllmWorker'], 
                         check=False, timeout=3,
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    def _terminate_background_processes_safe(self):
        """Safely terminate background processes without hanging"""
        logger.info("Starting safe background process termination")
        
        # Don't immediately kill Ray processes - let them cleanup naturally
        # Only kill non-Ray processes
        non_ray_processes = [
            'vllm',  # But not VllmWorker
        ]
        
        for process_name in non_ray_processes:
            try:
                subprocess.run(['pkill', '-f', process_name], 
                             check=False, timeout=2,
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            except Exception:
                pass
    
    def _terminate_background_processes(self):
        """Terminate all background processes related to vLLM and Ray"""
        processes_to_kill = [
            'vllm',
            'ray::IDLE',
            'ray::CoreWorker',
            'ray::RayletMonitor', 
            'ray::NodeManager',
            'ray::ObjectManager',
            'ray_serve',
            'raylet',
            'gcs_server'
        ]
        
        for process_name in processes_to_kill:
            try:
                # Kill processes by name
                subprocess.run(['pkill', '-f', process_name], 
                             check=False, timeout=3, 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
            except Exception:
                pass
        
        # Additional cleanup for Ray
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass
        
        # Force kill any remaining Ray processes
        try:
            subprocess.run(['ray', 'stop', '--force'], 
                         check=False, timeout=5,
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    @contextmanager
    def model_context(self):
        """Context manager for automatic cleanup"""
        try:
            yield self
        finally:
            self.cleanup()
    
    def format_prompt(self, query: str, model_name: Optional[str] = None) -> str:
        """Prompt formatting specific for each model"""
        
        if model_name is None:
            model_name = self.model_name
            
        model_lower = model_name.lower()
        
        # DeepSeek models
        if "deepseek" in model_lower:
            if any(word in query.lower() for word in ["calculate", "solve", "math", "equation", "proof"]):
                return f"User: {query}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            else:
                return f"User: {query}"
        
        # Llama models (including Llama 2 and 3)
        elif "llama" in model_lower:
            if "llama-3" in model_lower or "llama-2" in model_lower:
                return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                return f"[INST] {query} [/INST]"
        
        # Mistral models
        elif "mistral" in model_lower:
            return f"[INST] {query} [/INST]"
        
        # Gemma models
        elif "gemma" in model_lower:
            return f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n"
        
        # Default format
        else:
            return f"User: {query}\nAssistant:"
    
    @retry_with_backoff(max_retries=3)
    def generate(self,
                 query: str,
                 model_name: Optional[str] = None,
                 custom_params: Optional[Dict[str, Any]] = None,
                 remove_cot: bool = True,
                 return_full_output: bool = False) -> Union[str, Dict[str, Any]]:
        """Enhanced generation with better error handling and logging"""
        
        if self.is_cleaned_up:
            raise RuntimeError("Model has been cleaned up and is no longer available")
        
        # Update usage tracking
        if self.instance_id:
            _global_tracker.update_instance_usage(self.instance_id)
        
        if model_name is None:
            model_name = self.model_name
            
        prompt = self.format_prompt(query, model_name)
        
        # Prepare sampling parameters
        if custom_params:
            params = SamplingParams(**{**self.default_params.__dict__, **custom_params})
        else:
            params = self.default_params
        
        logger.debug(f"Generating response for model {model_name}")
        start_time = time.time()
        
        try:
            outputs = self.llm.generate([prompt], params)
            output = outputs[0]
            
            # Extract generated text with better error handling
            try:
                output_text = output.outputs[0].text
            except (AttributeError, IndexError) as e:
                logger.warning(f"Standard text extraction failed: {e}, trying alternative method")
                try:
                    output_text = output.outputs.text
                except AttributeError:
                    output_text = str(output)
                    logger.warning("Fallback to string conversion")
            
            gen_time = time.time() - start_time
            logger.debug(f"Generated response in {gen_time:.2f} seconds")
            
            # Remove Chain of Thought for DeepSeek models
            if remove_cot and "</think>" in output_text:
                # Extract content after </think>
                output_text = output_text.split("</think>")[1]
            
            # Remove Chain of Thought for Qwen models
            if remove_cot and "qwen" in model_name:
                pass
            
            if return_full_output:
                return {
                    "text": output_text,
                    "prompt": prompt,
                    "generation_time": gen_time,
                    "model": model_name,
                    "params": params.__dict__
                }
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed for model {model_name}: {e}")
            raise
    
    @retry_with_backoff(max_retries=2)
    def compute_yes_no_probability(self, 
                                   query: str, 
                                   model_name: Optional[str] = None,
                                   method: str = "logprobs") -> Dict[str, float]:
        """Enhanced probability computation with multiple methods"""
        
        if self.is_cleaned_up:
            raise RuntimeError("Model has been cleaned up and is no longer available")
        
        if model_name is None:
            model_name = self.model_name
            
        prompt = self.format_prompt(query, model_name)
        
        if method == "logprobs":
            return self._compute_logprobs_method(prompt, model_name)
        elif method == "completion":
            return self._compute_completion_method(prompt, model_name)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_logprobs_method(self, prompt: str, model_name: str) -> Dict[str, float]:
        """Compute probabilities using token logprobs"""
        
        try:
            # Create sampling parameters for probability computation
            prob_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
                logprobs=10  # Get top 10 token probabilities
            )
            
            # Generate with the prompt
            outputs = self.llm.generate([prompt], prob_params)
            output = outputs[0]
            
            # Get the logprobs for the first token
            first_token_logprobs = output.outputs[0].logprobs[0] if output.outputs[0].logprobs else {}
            
            # Look for "Yes" and "No" tokens (and variations)
            yes_variations = ["Yes", "YES", "yes", "Y", "True", "true"]
            no_variations = ["No", "NO", "no", "N", "False", "false"]
            
            yes_prob = float('-inf')
            no_prob = float('-inf')
            
            for token, logprob in first_token_logprobs.items():
                if any(var in token for var in yes_variations):
                    yes_prob = max(yes_prob, logprob.logprob)
                elif any(var in token for var in no_variations):
                    no_prob = max(no_prob, logprob.logprob)
            
            # If we couldn't find direct matches, use alternative approach
            if yes_prob == float('-inf') or no_prob == float('-inf'):
                return self._compute_completion_method(prompt, model_name)
            
            # Convert logprobs to probabilities and normalize
            yes_exp = np.exp(yes_prob)
            no_exp = np.exp(no_prob)
            total = yes_exp + no_exp
            
            return {
                "yes": float(yes_exp / total),
                "no": float(no_exp / total)
            }
            
        except Exception as e:
            logger.warning(f"Logprobs method failed: {e}, falling back to completion method")
            return self._compute_completion_method(prompt, model_name)
    
    def _compute_completion_method(self, prompt: str, model_name: str) -> Dict[str, float]:
        """Compute probabilities using completion likelihood"""
        
        prob_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=10
        )
        
        # Generate multiple completions and count Yes/No responses
        responses = []
        for _ in range(10):  # Sample 10 times
            output = self.llm.generate([prompt + " Answer:"], prob_params)[0]
            text = output.outputs[0].text.strip().lower()
            responses.append(text)
        
        # Count yes/no responses
        yes_count = sum(1 for r in responses if any(word in r for word in ["yes", "y", "true", "correct"]))
        no_count = sum(1 for r in responses if any(word in r for word in ["no", "n", "false", "incorrect"]))
        
        total = yes_count + no_count
        if total == 0:
            # If no clear yes/no responses, default to 50/50
            return {"yes": 0.5, "no": 0.5}
        
        return {
            "yes": float(yes_count / total),
            "no": float(no_count / total)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "device_count": torch.cuda.device_count(),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "is_cleaned_up": self.is_cleaned_up
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed (fallback)"""
        if not self.is_cleaned_up:
            self.cleanup()

    @classmethod
    def cleanup_all(cls):
        """Cleanup all model instances with coordination"""
        global _global_cleanup_in_progress
        
        with _global_cleanup_lock:
            if _global_cleanup_in_progress:
                logger.info("Global cleanup already in progress, skipping")
                return
            _global_cleanup_in_progress = True
        
        try:
            logger.info("Starting coordinated cleanup of all model instances")
            
            # Legacy cleanup
            for instance in cls._instances.copy():
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up legacy instance: {e}")
            cls._instances.clear()
            
            # Enhanced tracker cleanup
            _global_tracker.cleanup_all_instances()
            
            # Final Ray cleanup
            cls._final_ray_cleanup()
            
            logger.info("Coordinated cleanup completed")
            
        finally:
            with _global_cleanup_lock:
                _global_cleanup_in_progress = False
    
    @classmethod
    def _final_ray_cleanup(cls):
        """Perform final Ray cleanup after all instances are cleaned"""
        logger.info("Performing final Ray cleanup")
        
        try:
            import ray
            if ray.is_initialized():
                logger.info("Final Ray shutdown")
                ray.shutdown()
                
                # Wait a moment for cleanup
                time.sleep(1)
                
        except Exception as e:
            logger.warning(f"Final Ray cleanup failed: {e}")
        
        # Force kill any remaining processes
        try:
            subprocess.run(['pkill', '-9', '-f', 'ray'], 
                         check=False, timeout=3,
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            subprocess.run(['pkill', '-9', '-f', 'VllmWorker'], 
                         check=False, timeout=3,
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    @classmethod
    def get_global_instance_stats(cls) -> Dict[str, Any]:
        """Get comprehensive instance statistics"""
        return _global_tracker.get_instance_stats()
    
    @classmethod
    def get_memory_budget_status(cls) -> Dict[str, Any]:
        """Get memory budget status across all instances"""
        stats = _global_tracker.get_instance_stats()
        
        return {
            "memory_utilization_percent": stats["memory_utilization"] * 100,
            "total_memory_used_gb": stats["total_memory_gb"],
            "memory_budget_gb": stats["memory_budget_gb"],
            "available_memory_gb": max(0, stats["memory_budget_gb"] - stats["total_memory_gb"]),
            "instance_count": stats["total_instances"],
            "max_instances": stats["max_instances"],
            "can_load_new_instance": (
                stats["total_instances"] < stats["max_instances"] and 
                stats["memory_utilization"] < 0.8
            )
        }
