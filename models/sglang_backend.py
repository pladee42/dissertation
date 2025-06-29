"""
SGLang LLM Backend

Simple SGLang backend for unified model inference with minimal complexity.
"""

import requests
import logging
from typing import Dict, Any, Optional, List
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class SGLangBackend:
    """SGLang backend for LLM inference with parallel support"""
    
    def __init__(self, base_url: str = "http://localhost:30000", timeout: int = 60, max_parallel: int = 4):
        """
        Initialize SGLang backend
        
        Args:
            base_url: SGLang server URL
            timeout: Request timeout in seconds
            max_parallel: Maximum parallel requests
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_parallel = max_parallel
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)
        
        logger.info(f"SGLang backend initialized with URL: {base_url}, max_parallel: {max_parallel}")
    
    def generate(self, 
                prompt: str, 
                model: str,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                stop: Optional[list] = None) -> str:
        """
        Generate text using SGLang
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        try:
            # Simple SGLang API call
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop or []
                }
            }
            
            url = f"{self.base_url}/generate"
            
            logger.debug(f"Sending request to {url}")
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if "text" in result:
                generated_text = result["text"]
                # Remove the input prompt from output if present
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"SGLang request failed: {e}")
            raise Exception(f"SGLang backend error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SGLang response: {e}")
            raise Exception(f"SGLang response parsing error: {e}")
        except Exception as e:
            logger.error(f"SGLang generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if SGLang server is available"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    def generate_parallel(self, requests_data: List[Dict]) -> List[str]:
        """
        Generate text using SGLang in parallel
        
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
                      stop: Optional[list] = None) -> List[str]:
        """
        Generate text for multiple prompts in parallel
        
        Args:
            prompts: List of input prompts
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            List of generated text strings
        """
        requests_data = [
            {
                'prompt': prompt,
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'stop': stop
            }
            for prompt in prompts
        ]
        
        return self.generate_parallel(requests_data)
    
    def get_models(self) -> list:
        """Get available models from SGLang server"""
        try:
            response = self.session.get(
                f"{self.base_url}/get_model_info",
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return result.get("model_path", [])
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def __del__(self):
        """Cleanup thread pool on destruction"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)