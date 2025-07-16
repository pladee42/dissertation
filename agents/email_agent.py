"""
vLLM Email Generation Agent

This module provides email generation with:
- vLLM backend integration
- Template-based prompts
- Simple retry logic
"""

import logging
import time
from typing import Optional

from models.vllm_backend import VLLMBackend
from models.openrouter_backend import OpenRouterBackend
from utils.template_manager import get_template_manager
from config.config import get_setting, get_model_config

logger = logging.getLogger(__name__)

class EmailAgent:
    """vLLM-based Email Generation Agent"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8", backend_type: str = "vllm", model_key: str = None):
        """Initialize with appropriate backend"""
        self.model_id = model_id
        self.model_key = model_key  # Model configuration key
        self.model_name = model_id.split('/')[-1]
        
        # Detect backend type from model config if not specified
        if model_key:
            model_config = get_model_config(model_key)
            detected_backend_type = model_config.get('backend_type', backend_type)
            logger.info(f"Model key: {model_key}, detected backend_type: {detected_backend_type}")
            backend_type = detected_backend_type
        
        logger.info(f"Final backend_type for {model_key}: {backend_type}")
        
        # Initialize appropriate backend
        if backend_type == 'openrouter':
            self.backend = OpenRouterBackend()
            self.backend_type = 'openrouter'
            logger.info(f"EmailAgent initialized with OpenRouter backend for model: {self.model_name}")
        else:
            self.backend = VLLMBackend()
            self.backend_type = 'vllm'
            logger.info(f"EmailAgent initialized with vLLM backend for model: {self.model_name}")
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
    
    def generate_email(self, prompt: str, topic: str, template_id: str = "1") -> str:
        """Generate email using vLLM backend and templates"""
        start_time = time.time()
        
        try:
            # Get email template
            email_template = self.template_manager.get_email_template(template_id)
            
            # Format template with topic
            formatted_template = self.template_manager.format_template(
                email_template, 
                topic=topic
            )
            
            # Create full prompt
            full_prompt = f"{formatted_template}\n\nEmail:"
            
            # Generate with retry logic
            email_content = self._generate_with_retry(full_prompt, topic)
            
            generation_time = time.time() - start_time
            logger.info(f"Email generated in {generation_time:.2f}s for topic: {topic}")
            
            return email_content
            
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            return self._generate_fallback_email(topic)
    
    def _generate_with_retry(self, prompt: str, topic: str) -> str:
        """Generate text with simple retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Generate using vLLM backend with agent-specific parameters
                model_to_use = self.model_key or self.model_id
                max_tokens = get_setting('email_max_tokens', 2048)
                temperature = get_setting('email_temperature', 0.5)
                top_p = get_setting('email_top_p', 0.85)
                
                # Model-specific adjustments for temperature
                if 'vicuna' in model_to_use.lower():
                    temperature = min(temperature, 0.4)  # Cap at 0.4 for Vicuna
                elif 'llama' in model_to_use.lower():
                    temperature = min(temperature, 0.6)  # Cap at 0.6 for Llama
                
                result = self.backend.generate(
                    prompt=prompt,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1
                )
                
                if result.strip():
                    # Log raw output for debugging
                    logger.info(f"Raw model output (length: {len(result)}): {result[:200]}...")
                    
                    # Clean using simple end token logic
                    cleaned_result = self._clean_with_end_token(result.strip())
                    return cleaned_result
                else:
                    raise Exception("Empty response from backend")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, return fallback email
        logger.warning(f"Backend generation failed, using fallback for topic: {topic}")
        return self._generate_fallback_email(topic)
    
    def _generate_fallback_email(self, topic: str) -> str:
        """Generate fallback email when vLLM is unavailable"""
        return f"""Subject: Regarding {topic}

Dear Recipient,

I hope this email finds you well. I am writing to discuss {topic}.

This email was generated using a fallback mechanism as the primary backend was unavailable. In a production environment, this would be replaced with actual LLM-generated content.

Key points regarding {topic}:
- Important discussion topic
- Requires immediate attention
- Please review and respond

Thank you for your time and consideration.

Best regards,
[Your Name]

---
Generated by EmailAgent (Fallback Mode)"""
    
    def _clean_with_end_token(self, email_content: str) -> str:
        """Clean email content using <END_EMAIL> token for deterministic truncation"""
        import re
        
        original_length = len(email_content)
        
        # Check for END_EMAIL token
        end_token_pos = email_content.find('<END_EMAIL>')
        if end_token_pos != -1:
            # Truncate at END_EMAIL token
            email_content = email_content[:end_token_pos].strip()
            logger.info(f"Truncated at <END_EMAIL> token: {original_length} -> {len(email_content)} chars")
        else:
            # No end token found, use full content but log this
            logger.info(f"No <END_EMAIL> token found, using full content (length: {len(email_content)})")
        
        # Basic cleanup only - normalize whitespace and newlines
        email_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', email_content)  # Max 2 consecutive newlines
        email_content = email_content.strip()  # Remove leading/trailing whitespace
        
        # Ensure proper email formatting
        if email_content and not email_content.endswith('\n'):
            email_content += '\n'
        
        cleaned_length = len(email_content)
        if original_length != cleaned_length:
            logger.info(f"Email cleaned: {original_length} -> {cleaned_length} chars")
        
        return email_content

