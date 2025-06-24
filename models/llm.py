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
from contextlib import contextmanager
from typing import Dict, Optional, Union, Any
from vllm import LLM, SamplingParams
from utils.retry import retry_with_backoff
from config.settings import settings

# Load environment variables
load_dotenv(override=True)
os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    _instances = []  # Track all instances for global cleanup
    
    def __init__(self,
                 model_id: str,
                 dtype: str = "bfloat16",
                 task: str = "generate",
                 quantization: str = None,
                 custom_config: Optional[Dict[str, Any]] = None):
        """Initialize the model once and keep it in memory with enhanced configuration"""
        
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        self.is_cleaned_up = False
        
        # Track this instance for cleanup
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
    
    def cleanup(self):
        """Explicitly cleanup the model and associated processes"""
        if self.is_cleaned_up:
            return
            
        logger.info(f"Cleaning up model {self.model_name}")
        self.is_cleaned_up = True
        
        try:
            # Remove from instances list
            if self in ModelInference._instances:
                ModelInference._instances.remove(self)
            
            # Delete the model instance
            if hasattr(self, 'llm'):
                del self.llm
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache and synchronize
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # More comprehensive process cleanup
            self._terminate_background_processes()
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
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
        """Cleanup all model instances"""
        for instance in cls._instances.copy():
            instance.cleanup()
        cls._instances.clear()
