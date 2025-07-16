import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import gc
import torch
import warnings
import time

# Suppress specific warnings that might occur during model loading
warnings.filterwarnings("ignore", message=".*do_sample.*")
warnings.filterwarnings("ignore", message=".*top_p.*")
warnings.filterwarnings("ignore", message=".*temperature.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

logger = logging.getLogger(__name__)

class VLLMBackend:
    """vLLM backend for LLM inference using direct Python library"""
    
    def __init__(self, max_parallel: int = 4):
        """
        Initialize vLLM backend
        
        Args:
            max_parallel: Maximum parallel requests
        """
        self.max_parallel = max_parallel
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)
        self.engines = {}  # Cache for loaded models
        
        # Set cache directory and GPU memory settings
        cache_dir = "./downloaded_models"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        logger.info(f"vLLM backend initialized with max_parallel: {max_parallel}")
    
    def _get_engine(self, model: str):
        """Get or create vLLM engine for model"""
        if model not in self.engines:
            try:
                # Suppress warnings during model loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from vllm import LLM
                    from config.config import get_model_config
                
                logger.info(f"Loading vLLM model: {model}")
                
                # Get model configuration
                model_config = get_model_config(model)
                model_id = model_config.get('model_id', model)
                quantization = model_config.get('quantization', 'experts_int8')
                dtype = model_config.get('dtype', 'bfloat16')
                
                logger.info(f"Model config - ID: {model_id}, Quantization: {quantization}, Dtype: {dtype}")
                
                # Prepare vLLM initialization parameters
                vllm_kwargs = {
                    "model": model_id,
                    "download_dir": "./downloaded_models",
                    "trust_remote_code": True,
                    "max_model_len": 2048,  # Reduced for memory efficiency
                    "gpu_memory_utilization": 0.85,  # Higher memory usage for large models
                    "dtype": dtype,
                    "enforce_eager": True,  # Disable CUDA graphs for memory efficiency
                    "disable_custom_all_reduce": True  # Better memory management
                }
                
                # Adjust memory utilization for different model sizes
                if 'judgelm' in model.lower() or 'yi' in model.lower():
                    vllm_kwargs["gpu_memory_utilization"] = 0.9  # Very high for 33B+ models
                    vllm_kwargs["max_model_len"] = 1024  # Reduce sequence length for large models
                
                # Add quantization if specified
                if quantization == "awq":
                    vllm_kwargs["quantization"] = "awq"
                elif quantization == "experts_int8":
                    # For experts_int8, we'll use load_format instead
                    vllm_kwargs["load_format"] = "auto"
                
                # Initialize vLLM engine with warning suppression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.engines[model] = LLM(**vllm_kwargs)
                logger.info(f"Successfully loaded vLLM model: {model}")
                
            except Exception as e:
                logger.error(f"Failed to load vLLM model {model}: {e}")
                raise
        
        return self.engines[model]
    
    def generate(self, 
                prompt: str, 
                model: str,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                stop: Optional[list] = None,
                json_schema: Optional[dict] = None,
                guided_json: Optional[dict] = None) -> str:
        """
        Generate text using vLLM
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty to reduce repetitive generation
            stop: Stop sequences
            json_schema: JSON schema for guided decoding
            guided_json: Guided JSON schema for structured output
            
        Returns:
            Generated text
        """
        try:
            from vllm import SamplingParams
            
            # Get vLLM engine
            engine = self._get_engine(model)
            
            # Add model-specific stop tokens
            stop_tokens = stop or []
            
            # Add stop tokens for Qwen models to prevent thinking process
            if 'qwen' in model.lower():
                qwen_stop_tokens = [
                    "<|thinking|>", "</thinking>", "<thinking>", 
                    "<|im_start|>thinking", "<|im_end|>thinking",
                    "Let me think", "I need to think", "Thinking:",
                    "\n\nLet me", "\n\nI need to"
                ]
                stop_tokens.extend(qwen_stop_tokens)
            
            # Add stop tokens for DeepSeek models to prevent thinking process
            if 'deepseek' in model.lower():
                deepseek_stop_tokens = [
                    "<think>", "</think>"
                ]
                stop_tokens.extend(deepseek_stop_tokens)
            
            # Add stop tokens for Yi models to prevent format tags
            if 'yi' in model.lower():
                yi_stop_tokens = [
                    "<|output_json_array|>", "</|output_json_array|>",
                    "<|im_end|>", "<|im_start|>"
                ]
                stop_tokens.extend(yi_stop_tokens)
            
            # No stop tokens for Llama models - let them generate freely
            if 'llama' in model.lower():
                pass  # Don't add any stop tokens for now
            
            # Add stop tokens for JudgeLM models for clean JSON output
            if 'judgelm' in model.lower():
                judgelm_stop_tokens = [
                    "```", "\n\n\n"  # Stop at code blocks or excessive newlines
                ]
                stop_tokens.extend(judgelm_stop_tokens)
            
            # Add email-specific stop tokens to prevent repetition and over-generation
            # Check if this is likely an email generation request based on prompt content
            if any(keyword in prompt.lower() for keyword in ['email', 'subject:', 'dear', 'sincerely', 'best regards']):
                email_stop_tokens = [
                    "\n\nBest regards,", "\n\nSincerely,", "\n\nThank you,",
                    "\n\nWarm regards,", "\n\nKind regards,", "\n\nYours truly,",
                    "\n---\n", "\n\n---\n", "\n\n\n\n",  # Signature separators and excessive newlines
                    "\nSubject:", "\n\nSubject:",  # Prevent starting new emails
                    "\n\n[Your Name]", "\n[Your Name]",  # Template placeholders
                    "\n\nEmail:", "\n\nEmail Generated"  # Meta content indicators
                ]
                stop_tokens.extend(email_stop_tokens)
            
            # Create sampling parameters
            # Note: vLLM uses temperature to control sampling automatically
            # temperature=0.0 -> deterministic, temperature>0.0 -> sampling enabled
            sampling_params_dict = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stop': stop_tokens
            }
            
            # Only add top_p when sampling (temperature > 0)
            if temperature > 0:
                sampling_params_dict['top_p'] = top_p
            
            # Add repetition penalty if specified (only when not 1.0)
            if repetition_penalty != 1.0:
                sampling_params_dict['repetition_penalty'] = repetition_penalty
            
            sampling_params = SamplingParams(**sampling_params_dict)
            
            # Generate with warning suppression
            logger.debug(f"Generating with model: {model}, max_tokens: {max_tokens}, temperature: {temperature}")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = engine.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                output = outputs[0]
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text
                    logger.info(f"vLLM raw output: '{generated_text}' (length: {len(generated_text)})")
                    
                    # Save raw output for debugging
                    import os
                    os.makedirs("output/debug", exist_ok=True)
                    with open(f"output/debug/vllm_raw_{model.replace('/', '_')}_{int(time.time())}.txt", "w") as f:
                        f.write(f"Model: {model}\nGenerated text: '{generated_text}'\nLength: {len(generated_text)}\n")
                    
                    if generated_text.strip():
                        return generated_text.strip()
                    else:
                        logger.error(f"Empty output generated from vLLM for model: {model}")
                        raise Exception(f"Empty output generated from vLLM for model: {model}")
                else:
                    logger.error(f"No outputs in vLLM response for model: {model}")
                    raise Exception(f"No outputs in vLLM response for model: {model}")
            else:
                logger.error(f"No output generated from vLLM for model: {model}")
                raise Exception(f"No output generated from vLLM for model: {model}")
                
        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise Exception(f"vLLM backend error: {e}")
    
    def is_available(self) -> bool:
        """Check if vLLM is available"""
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def generate_parallel(self, requests_data: List[Dict]) -> List[str]:
        """
        Generate text using vLLM in parallel
        
        Args:
            requests_data: List of request dictionaries, each containing:
                - prompt: Input prompt
                - model: Model name (optional)
                - max_tokens: Maximum tokens (optional)
                - temperature: Sampling temperature (optional)
                - stop: Stop sequences (optional)
        
        Returns:
            List of generated text strings
        """
        logger.info(f"Starting parallel generation for {len(requests_data)} requests")
        
        # Submit all requests to thread pool
        future_to_index = {}
        for i, req_data in enumerate(requests_data):
            future = self.executor.submit(
                self.generate,
                req_data.get('prompt', ''),
                req_data.get('model', ''),
                req_data.get('max_tokens', 1000),
                req_data.get('temperature', 0.7),
                req_data.get('stop')
            )
            future_to_index[future] = i
        
        # Collect results in order
        results = [''] * len(requests_data)
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                logger.debug(f"Completed request {index + 1}/{len(requests_data)}")
            except Exception as e:
                logger.error(f"Request {index} failed: {e}")
                results[index] = f"Error: {e}"
        
        logger.info(f"Parallel generation completed for {len(requests_data)} requests")
        return results
    
    def generate_batch(self, 
                      prompts: List[str], 
                      model: str = '',
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      top_p: float = 1.0,
                      repetition_penalty: float = 1.0,
                      stop: Optional[list] = None) -> List[str]:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of input prompts
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            List of generated text strings
        """
        try:
            from vllm import SamplingParams
            
            # Get vLLM engine
            engine = self._get_engine(model)
            
            # Add model-specific stop tokens
            stop_tokens = stop or []
            
            # Add stop tokens for Qwen models to prevent thinking process
            if 'qwen' in model.lower():
                qwen_stop_tokens = [
                    "<|thinking|>", "</thinking>", "<thinking>", 
                    "<|im_start|>thinking", "<|im_end|>thinking",
                    "Let me think", "I need to think", "Thinking:",
                    "\n\nLet me", "\n\nI need to"
                ]
                stop_tokens.extend(qwen_stop_tokens)
            
            # Add stop tokens for DeepSeek models to prevent thinking process
            if 'deepseek' in model.lower():
                deepseek_stop_tokens = [
                    "<think>", "</think>"
                ]
                stop_tokens.extend(deepseek_stop_tokens)
            
            # Add stop tokens for Yi models to prevent format tags
            if 'yi' in model.lower():
                yi_stop_tokens = [
                    "<|output_json_array|>", "</|output_json_array|>",
                    "<|im_end|>", "<|im_start|>"
                ]
                stop_tokens.extend(yi_stop_tokens)
            
            # No stop tokens for Llama models - let them generate freely
            if 'llama' in model.lower():
                pass  # Don't add any stop tokens for now
            
            # Add stop tokens for JudgeLM models for clean JSON output
            if 'judgelm' in model.lower():
                judgelm_stop_tokens = [
                    "```", "\n\n\n"  # Stop at code blocks or excessive newlines
                ]
                stop_tokens.extend(judgelm_stop_tokens)
            
            # Add email-specific stop tokens to prevent repetition and over-generation
            # Check if this is likely an email generation request based on prompt content
            if any(keyword in prompt.lower() for keyword in ['email', 'subject:', 'dear', 'sincerely', 'best regards']):
                email_stop_tokens = [
                    "\n\nBest regards,", "\n\nSincerely,", "\n\nThank you,",
                    "\n\nWarm regards,", "\n\nKind regards,", "\n\nYours truly,",
                    "\n---\n", "\n\n---\n", "\n\n\n\n",  # Signature separators and excessive newlines
                    "\nSubject:", "\n\nSubject:",  # Prevent starting new emails
                    "\n\n[Your Name]", "\n[Your Name]",  # Template placeholders
                    "\n\nEmail:", "\n\nEmail Generated"  # Meta content indicators
                ]
                stop_tokens.extend(email_stop_tokens)
            
            # Create sampling parameters
            # Note: vLLM uses temperature to control sampling automatically
            # temperature=0.0 -> deterministic, temperature>0.0 -> sampling enabled
            sampling_params_dict = {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stop': stop_tokens
            }
            
            # Only add top_p when sampling (temperature > 0)
            if temperature > 0:
                sampling_params_dict['top_p'] = top_p
            
            # Add repetition penalty if specified (only when not 1.0)
            if repetition_penalty != 1.0:
                sampling_params_dict['repetition_penalty'] = repetition_penalty
            
            sampling_params = SamplingParams(**sampling_params_dict)
            
            # Generate batch with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = engine.generate(prompts, sampling_params)
            
            # Extract generated texts
            results = []
            for i, output in enumerate(outputs):
                if output.outputs and len(output.outputs) > 0:
                    generated_text = output.outputs[0].text.strip()
                    if generated_text:
                        results.append(generated_text)
                    else:
                        logger.warning(f"Empty output for prompt {i}")
                        results.append("")
                else:
                    logger.warning(f"No output for prompt {i}")
                    results.append("")
            
            return results
            
        except Exception as e:
            logger.error(f"vLLM batch generation error: {e}")
            # Fallback to parallel generation
            return self.generate_parallel([
                {
                    'prompt': prompt,
                    'model': model,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'stop': stop
                }
                for prompt in prompts
            ])
    
    def get_models(self) -> list:
        """Get available models (returns loaded models)"""
        return list(self.engines.keys())
    
    def get_model_info(self) -> dict:
        """Get model info (compatibility method)"""
        models = self.get_models()
        return {"model_path": models[0] if models else ""}
    
    def unload_model(self, model_name: str):
        """Unload a specific model to free GPU memory"""
        if model_name in self.engines:
            del self.engines[model_name]
            self.cleanup_memory()
            logger.info(f"Unloaded model: {model_name}")
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleaned up")
    
    def __del__(self):
        """Cleanup thread pool and engines on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Clean up vLLM engines
        if hasattr(self, 'engines'):
            for engine in self.engines.values():
                try:
                    del engine
                except:
                    pass
        
        # Clean up GPU memory
        self.cleanup_memory()