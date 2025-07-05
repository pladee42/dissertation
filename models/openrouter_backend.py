"""
OpenRouter Backend for API-based model inference

Simple HTTP client for OpenRouter API with compatibility interface matching vLLM backend
"""

import logging
import requests
import time
import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterBackend:
    """OpenRouter backend for API-based LLM inference"""
    
    def __init__(self, max_parallel: int = 4):
        """
        Initialize OpenRouter backend
        
        Args:
            max_parallel: Maximum parallel requests (for compatibility)
        """
        self.max_parallel = max_parallel
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please check your .env file.")
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional
            "X-Title": "Email Generation System"  # Optional
        }
        
        logger.info(f"OpenRouter backend initialized with max_parallel: {max_parallel}")
    
    def generate(self, 
                prompt: str, 
                model: str,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                stop: Optional[list] = None,
                json_schema: Optional[dict] = None,
                guided_json: Optional[dict] = None) -> str:
        """
        Generate text using OpenRouter API
        
        Args:
            prompt: Input prompt
            model: Model name (mapped to OpenRouter model ID)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            json_schema: JSON schema (not used in OpenRouter)
            guided_json: Guided JSON (not used in OpenRouter)
            
        Returns:
            Generated text
        """
        try:
            # Map model name to OpenRouter model ID
            model_id = self._get_openrouter_model_id(model)
            
            # Prepare request payload
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            # Add stop sequences if provided
            if stop:
                payload["stop"] = stop
            
            logger.debug(f"Sending request to OpenRouter: model={model_id}, max_tokens={max_tokens}, temperature={temperature}")
            
            # Make API request
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Parse response
            result = response.json()
            
            if 'choices' not in result or len(result['choices']) == 0:
                raise Exception("No choices in OpenRouter response")
            
            generated_text = result['choices'][0]['message']['content']
            
            if generated_text.strip():
                logger.debug(f"Generated text length: {len(generated_text)}")
                return generated_text.strip()
            else:
                raise Exception("Empty response from OpenRouter API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter request error: {e}")
            raise Exception(f"OpenRouter API request failed: {e}")
        except Exception as e:
            logger.error(f"OpenRouter generation error: {e}")
            raise Exception(f"OpenRouter backend error: {e}")
    
    def _get_openrouter_model_id(self, model: str) -> str:
        """Map internal model names to OpenRouter model IDs"""
        model_mapping = {
            'gemini-2.5-flash': 'google/gemini-2.5-flash-lite-preview-06-17',
            'google/gemini-2.5-flash-lite-preview-06-17': 'google/gemini-2.5-flash-lite-preview-06-17'
        }
        
        return model_mapping.get(model, model)
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
        try:
            # Simple test request to check API availability
            test_payload = {
                "model": "google/gemini-2.5-flash-lite-preview-06-17",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=test_payload,
                timeout=10
            )
            
            return response.status_code == 200
        except:
            return False
    
    def generate_parallel(self, requests_data: List[Dict]) -> List[str]:
        """
        Generate text in parallel (simplified sequential for now)
        
        Args:
            requests_data: List of request dictionaries
            
        Returns:
            List of generated text strings
        """
        results = []
        for req_data in requests_data:
            try:
                result = self.generate(
                    req_data.get('prompt', ''),
                    req_data.get('model', ''),
                    req_data.get('max_tokens', 1000),
                    req_data.get('temperature', 0.7),
                    req_data.get('stop')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel request failed: {e}")
                results.append(f"Error: {e}")
        
        return results
    
    def generate_batch(self, 
                      prompts: List[str], 
                      model: str = '',
                      max_tokens: int = 1000,
                      temperature: float = 0.7,
                      stop: Optional[list] = None) -> List[str]:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of input prompts
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            
        Returns:
            List of generated text strings
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, model, max_tokens, temperature, stop)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                results.append("")
        
        return results
    
    def get_models(self) -> list:
        """Get available models (compatibility method)"""
        return ['google/gemini-2.5-flash-lite-preview-06-17']
    
    def get_model_info(self) -> dict:
        """Get model info (compatibility method)"""
        return {"model_path": "openrouter_api"}
    
    def cleanup_memory(self):
        """Clean up memory (no-op for API backend)"""
        logger.debug("OpenRouter cleanup (no-op)")
        pass
    
    def unload_model(self, model_name: str):
        """Unload model (no-op for API backend)"""
        logger.debug(f"OpenRouter unload model: {model_name} (no-op)")
        pass