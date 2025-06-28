"""
SGLang Email Generation Agent

This module provides email generation with:
- SGLang backend integration
- Template-based prompts
- Simple retry logic
"""

import logging
import time
from typing import Optional

from models.sglang_backend import SGLangBackend
from utils.template_manager import get_template_manager
from config.config import get_setting

logger = logging.getLogger(__name__)

class EmailAgent:
    """SGLang-based Email Generation Agent"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8"):
        """Initialize with SGLang backend"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        # Initialize SGLang backend
        sglang_url = get_setting('sglang_server_url', 'http://localhost:30000')
        sglang_timeout = get_setting('sglang_timeout', 60)
        self.backend = SGLangBackend(base_url=sglang_url, timeout=sglang_timeout)
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
        
        logger.info(f"EmailAgent initialized with model: {self.model_name}")
    
    def generate_email(self, prompt: str, topic: str, template_id: str = "1") -> str:
        """Generate email using SGLang backend and templates"""
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
            email_content = self._generate_with_retry(full_prompt)
            
            generation_time = time.time() - start_time
            logger.info(f"Email generated in {generation_time:.2f}s for topic: {topic}")
            
            return email_content
            
        except Exception as e:
            logger.error(f"Error generating email: {e}")
            return f"Error generating email for topic '{topic}': {str(e)}"
    
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with simple retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Generate using SGLang backend
                result = self.backend.generate(
                    prompt=prompt,
                    model=self.model_id,
                    max_tokens=get_setting('max_tokens', 2048),
                    temperature=get_setting('temperature', 0.7)
                )
                
                if result.strip():
                    return result.strip()
                else:
                    raise Exception("Empty response from SGLang")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, return error message
        return f"Failed to generate email after {self.max_retries} attempts. Last error: {last_error}"

