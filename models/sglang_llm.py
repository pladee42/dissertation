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
import requests
import json
from contextlib import contextmanager
from typing import Dict, Optional, Union, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import sglang as sgl
from sglang import Runtime, set_default_backend
from sglang.api import generate, chat_completion
from utils.retry import retry_with_backoff
from config.settings import settings
from config.sglang_config import get_sglang_config, SGLangServerConfig

# Load environment variables
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SGLangInstanceMetadata:
    """Enhanced metadata for SGLang model instances"""
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
    server_port: int
    server_url: str
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0
    cleanup_callbacks: List[callable] = field(default_factory=list)

class SGLangInstanceTracker:
    """Global instance tracking for SGLang models with RadixAttention optimization"""
    
    def __init__(self):
        self._instances: Dict[str, SGLangInstanceMetadata] = {}
        self._lock = threading.RLock()
        self._cleanup_in_progress = False
        self._max_instances = getattr(settings, 'max_concurrent_models', 2)
        self._memory_budget_gb = getattr(settings, 'max_gpu_memory_per_model', 24.0)
        self._port_pool = list(range(30000, 30100))  # Pool of available ports
        self._used_ports = set()
        
        # Register cleanup handlers
        if os.getpid() == os.getpid():
            try:
                import multiprocessing
                if multiprocessing.current_process().name == 'MainProcess':
                    atexit.register(self.cleanup_all_instances)
                    signal.signal(signal.SIGTERM, self._signal_handler)
                    signal.signal(signal.SIGINT, self._signal_handler)
                    logger.debug("SGLang cleanup handlers registered")
            except Exception as e:
                logger.debug(f"Error registering cleanup handlers: {e}")
        
        logger.info("SGLangInstanceTracker initialized")
    
    def get_available_port(self) -> int:
        """Get an available port for SGLang server"""
        with self._lock:
            for port in self._port_pool:
                if port not in self._used_ports:
                    self._used_ports.add(port)
                    return port
            raise RuntimeError("No available ports for SGLang server")
    
    def release_port(self, port: int):
        """Release a port back to the pool"""
        with self._lock:
            self._used_ports.discard(port)
    
    def register_instance(self, instance: 'SGLangModelInference', memory_allocated_gb: float = 0.0) -> str:
        """Register a new SGLang model instance"""
        with self._lock:
            instance_id = str(uuid.uuid4())
            
            metadata = SGLangInstanceMetadata(
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
                server_port=instance.server_port,
                server_url=instance.server_url,
                last_used=datetime.now()
            )
            
            self._instances[instance_id] = metadata
            logger.info(f"Registered SGLang instance {instance_id[:8]} for model {metadata.model_name}")
            
            # Check if we exceed limits
            self._enforce_instance_limits()
            
            return instance_id
    
    def cleanup_all_instances(self):
        """Cleanup all tracked SGLang instances"""
        with self._lock:
            if self._cleanup_in_progress:
                logger.debug("SGLang cleanup already in progress, skipping")
                return
            
            self._cleanup_in_progress = True
            
        try:
            logger.info("Starting SGLang global instance cleanup")
            
            instance_ids = list(self._instances.keys())
            logger.info(f"Cleaning up {len(instance_ids)} SGLang instances")
            
            for instance_id in instance_ids:
                try:
                    metadata = self._instances[instance_id]
                    logger.debug(f"Cleaning up SGLang instance {instance_id[:8]}")
                    
                    # Stop SGLang server
                    self._stop_sglang_server(metadata.server_port)
                    
                    # Release port
                    self.release_port(metadata.server_port)
                    
                    # Execute cleanup callbacks
                    for callback in metadata.cleanup_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.warning(f"Cleanup callback failed for {instance_id}: {e}")
                    
                    del self._instances[instance_id]
                    
                except Exception as e:
                    logger.error(f"Error cleaning up SGLang instance {instance_id}: {e}")
            
            self._instances.clear()
            self._used_ports.clear()
            logger.info("SGLang global instance cleanup completed")
            
        finally:
            with self._lock:
                self._cleanup_in_progress = False
    
    def _stop_sglang_server(self, port: int):
        """Stop SGLang server running on specific port"""
        try:
            # Try graceful shutdown via API
            requests.post(f"http://127.0.0.1:{port}/shutdown", timeout=5)
            time.sleep(2)
        except Exception:
            pass
        
        # Force kill processes using the port
        try:
            subprocess.run(['pkill', '-f', f'--port {port}'], 
                         check=False, timeout=3,
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating SGLang cleanup")
        self.cleanup_all_instances()
        sys.exit(0)
    
    def _enforce_instance_limits(self):
        """Enforce instance limits by cleaning up oldest instances"""
        active_instances = [m for m in self._instances.values() if m.is_active]
        
        if len(active_instances) > self._max_instances:
            logger.warning(f"SGLang instance limit ({self._max_instances}) exceeded")
            sorted_instances = sorted(active_instances, key=lambda m: m.last_used or m.created_at)
            instances_to_cleanup = len(active_instances) - self._max_instances
            
            for metadata in sorted_instances[:instances_to_cleanup]:
                logger.info(f"Auto-cleanup of SGLang instance {metadata.instance_id[:8]} due to limit")
                self._stop_sglang_server(metadata.server_port)
                self.release_port(metadata.server_port)
                del self._instances[metadata.instance_id]

# Global SGLang instance tracker
_sglang_tracker = SGLangInstanceTracker()

class SGLangModelInference:
    """SGLang-based model inference with RadixAttention optimization"""
    
    _instances = []  # Legacy compatibility
    
    def __init__(self,
                 model_id: str,
                 dtype: str = "bfloat16",
                 task: str = "generate",
                 quantization: str = None,
                 custom_config: Optional[Dict[str, Any]] = None):
        """Initialize SGLang model with server-based architecture"""
        
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.dtype = dtype
        self.quantization = quantization
        self.task = task
        self.is_cleaned_up = False
        self.instance_id = None
        
        # Get available port
        self.server_port = _sglang_tracker.get_available_port()
        self.server_url = f"http://127.0.0.1:{self.server_port}"
        
        # Legacy tracking
        SGLangModelInference._instances.append(self)
        
        logger.info(f"Initializing SGLang server for model {model_id} on port {self.server_port}")
        start_time = time.time()
        
        # Get SGLang configuration
        self.sglang_config = get_sglang_config(self.model_name, custom_config)
        self.sglang_config.model_path = model_id
        self.sglang_config.port = self.server_port
        
        if quantization:
            self.sglang_config.quantization = quantization
        
        # Start SGLang server
        try:
            self._start_sglang_server()
            self._wait_for_server_ready()
        except Exception as e:
            logger.error(f"Failed to start SGLang server for {model_id}: {e}")
            _sglang_tracker.release_port(self.server_port)
            raise
        
        # Initialize SGLang backend
        try:
            self.backend = Runtime(
                model_path=model_id,
                tokenizer_path=None,
                base_url=self.server_url,
                api_key=None
            )
            set_default_backend(self.backend)
        except Exception as e:
            logger.error(f"Failed to initialize SGLang backend: {e}")
            self._stop_server()
            raise
        
        load_time = time.time() - start_time
        logger.info(f"SGLang model {self.model_name} loaded in {load_time:.2f} seconds")
        
        # Register with tracker
        memory_allocated = 0.0
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        
        self.instance_id = _sglang_tracker.register_instance(self, memory_allocated)
    
    def _start_sglang_server(self):
        """Start SGLang server process"""
        server_args = self.sglang_config.to_server_args()
        
        logger.info(f"Starting SGLang server with args: {' '.join(server_args)}")
        
        # Start server in background
        self.server_process = subprocess.Popen(
            server_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, **{
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "SGLANG_BACKEND": "flashinfer",
                "SGLANG_MEM_FRACTION_STATIC": str(self.sglang_config.mem_fraction_static)
            })
        )
        
        logger.info(f"SGLang server process started with PID {self.server_process.pid}")
    
    def _wait_for_server_ready(self, timeout: int = 120):
        """Wait for SGLang server to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info(f"SGLang server ready on port {self.server_port}")
                    return
            except Exception:
                pass
            
            time.sleep(2)
        
        raise RuntimeError(f"SGLang server failed to start within {timeout} seconds")
    
    def _stop_server(self):
        """Stop SGLang server process"""
        if hasattr(self, 'server_process') and self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass
    
    def format_prompt(self, query: str, model_name: Optional[str] = None) -> str:
        """Prompt formatting specific for each model (same as VLLM version)"""
        if model_name is None:
            model_name = self.model_name
            
        model_lower = model_name.lower()
        
        # DeepSeek models
        if "deepseek" in model_lower:
            if any(word in query.lower() for word in ["calculate", "solve", "math", "equation", "proof"]):
                return f"User: {query}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            else:
                return f"User: {query}"
        
        # Llama models
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
        """Generate response using SGLang with RadixAttention optimization"""
        
        if self.is_cleaned_up:
            raise RuntimeError("SGLang model has been cleaned up and is no longer available")
        
        # Update usage tracking
        if self.instance_id:
            _sglang_tracker._instances[self.instance_id].last_used = datetime.now()
            _sglang_tracker._instances[self.instance_id].usage_count += 1
        
        if model_name is None:
            model_name = self.model_name
            
        prompt = self.format_prompt(query, model_name)
        
        # Default generation parameters
        gen_params = {
            "temperature": custom_params.get("temperature", settings.default_temperature) if custom_params else settings.default_temperature,
            "top_p": custom_params.get("top_p", settings.default_top_p) if custom_params else settings.default_top_p,
            "max_new_tokens": custom_params.get("max_tokens", settings.default_max_tokens) if custom_params else settings.default_max_tokens,
            "stop": custom_params.get("stop", None) if custom_params else None
        }
        
        logger.debug(f"Generating response for SGLang model {model_name}")
        start_time = time.time()
        
        try:
            # Use SGLang's generate function with RadixAttention
            @sgl.function
            def generate_response(s, prompt_text):
                s += prompt_text
                s += sgl.gen("response", **gen_params)
            
            # Execute generation
            state = generate_response.run(prompt_text=prompt)
            output_text = state["response"]
            
            gen_time = time.time() - start_time
            logger.debug(f"Generated response in {gen_time:.2f} seconds")
            
            # Remove Chain of Thought for DeepSeek models
            if remove_cot and "</think>" in output_text:
                output_text = output_text.split("</think>")[1]
            
            if return_full_output:
                return {
                    "text": output_text,
                    "prompt": prompt,
                    "generation_time": gen_time,
                    "model": model_name,
                    "params": gen_params
                }
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"SGLang generation failed for model {model_name}: {e}")
            raise
    
    @retry_with_backoff(max_retries=2)
    def compute_yes_no_probability(self, 
                                   query: str, 
                                   model_name: Optional[str] = None,
                                   method: str = "logprobs") -> Dict[str, float]:
        """Compute yes/no probabilities using SGLang's constrained generation"""
        
        if self.is_cleaned_up:
            raise RuntimeError("SGLang model has been cleaned up and is no longer available")
        
        if model_name is None:
            model_name = self.model_name
            
        prompt = self.format_prompt(query, model_name)
        
        try:
            # Use SGLang for constrained generation
            @sgl.function
            def compute_probabilities(s, prompt_text):
                s += prompt_text
                s += sgl.gen("answer", choices=["Yes", "No"], temperature=0.0)
            
            # Run multiple times for probability estimation
            yes_count = 0
            total_runs = 10
            
            for _ in range(total_runs):
                state = compute_probabilities.run(prompt_text=prompt)
                if state["answer"].strip().lower().startswith("yes"):
                    yes_count += 1
            
            yes_prob = yes_count / total_runs
            no_prob = 1.0 - yes_prob
            
            return {
                "yes": float(yes_prob),
                "no": float(no_prob)
            }
            
        except Exception as e:
            logger.warning(f"SGLang probability computation failed: {e}, using fallback")
            return {"yes": 0.5, "no": 0.5}
    
    def cleanup(self):
        """Cleanup SGLang model instance and server"""
        if self.is_cleaned_up:
            return
        
        logger.info(f"Starting cleanup for SGLang model {self.model_name}")
        self.is_cleaned_up = True
        
        try:
            # Stop SGLang server
            self._stop_server()
            
            # Release port
            _sglang_tracker.release_port(self.server_port)
            
            # Cleanup backend
            if hasattr(self, 'backend') and self.backend:
                try:
                    del self.backend
                except Exception as e:
                    logger.warning(f"Error cleaning up SGLang backend: {e}")
            
            # Remove from tracking
            if self in SGLangModelInference._instances:
                SGLangModelInference._instances.remove(self)
            
            if self.instance_id and self.instance_id in _sglang_tracker._instances:
                del _sglang_tracker._instances[self.instance_id]
                self.instance_id = None
            
            # Memory cleanup
            self._aggressive_memory_cleanup()
            
            logger.info(f"Cleanup completed for SGLang model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error during SGLang cleanup: {e}")
            self.is_cleaned_up = True
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.info("Starting SGLang memory cleanup")
        
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"Garbage collection round {i+1}: {collected} objects collected")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    
    @contextmanager
    def model_context(self):
        """Context manager for automatic cleanup"""
        try:
            yield self
        finally:
            self.cleanup()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded SGLang model"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "server_url": self.server_url,
            "server_port": self.server_port,
            "device_count": torch.cuda.device_count(),
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "is_cleaned_up": self.is_cleaned_up,
            "backend": "sglang"
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def __del__(self):
        if not self.is_cleaned_up:
            self.cleanup()
    
    @classmethod
    def cleanup_all(cls):
        """Cleanup all SGLang model instances"""
        logger.info("Starting cleanup of all SGLang model instances")
        
        for instance in cls._instances.copy():
            try:
                instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up SGLang instance: {e}")
        
        cls._instances.clear()
        _sglang_tracker.cleanup_all_instances()
        logger.info("SGLang cleanup completed")
    
    @classmethod
    def get_global_instance_stats(cls) -> Dict[str, Any]:
        """Get comprehensive SGLang instance statistics"""
        active_instances = [m for m in _sglang_tracker._instances.values() if m.is_active]
        
        return {
            "total_instances": len(active_instances),
            "used_ports": list(_sglang_tracker._used_ports),
            "backend": "sglang",
            "radix_attention_enabled": True,
            "instances_by_model": {
                m.model_name: {"port": m.server_port, "usage_count": m.usage_count}
                for m in active_instances
            }
        }