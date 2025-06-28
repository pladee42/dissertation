"""
SGLang LLM Backend

Simple SGLang backend for unified model inference with minimal complexity.
"""

import requests
import logging
from typing import Dict, Any, Optional
import json

logger = logging.getLogger(__name__)

class SGLangBackend:
    """Simple SGLang backend for LLM inference"""
    
    def __init__(self, base_url: str = "http://localhost:30000", timeout: int = 60):
        """
        Initialize SGLang backend
        
        Args:
            base_url: SGLang server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        logger.info(f"SGLang backend initialized with URL: {base_url}")
    
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