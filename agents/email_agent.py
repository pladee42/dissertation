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
                    top_p=top_p
                )
                
                if result.strip():
                    # Clean up excessive formatting from certain models
                    cleaned_result = self._clean_email_formatting(result.strip())
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
    
    def _clean_email_formatting(self, email_content: str) -> str:
        """Clean excessive formatting from model outputs and prevent repetition"""
        import re
        
        # Log original length for debugging
        original_length = len(email_content)
        
        # Detect and truncate repetitive content patterns
        email_content = self._remove_repetitive_content(email_content)
        
        # Stop at the first proper email completion
        email_content = self._truncate_at_completion(email_content)
        
        # Remove excessive horizontal rules (---) at the end of emails
        # Keep only one if it appears to be a proper signature separator
        email_content = re.sub(r'\n\s*---+\s*(\n\s*---+\s*)*$', '', email_content)
        
        # Remove multiple consecutive horizontal rules throughout the email
        # Replace 2+ consecutive --- lines with just one
        email_content = re.sub(r'(\n\s*---+\s*){2,}', '\n\n---\n\n', email_content)
        
        # Remove horizontal rules at the very beginning
        email_content = re.sub(r'^(\s*---+\s*\n)+', '', email_content)
        
        # Clean up excessive blank lines (more than 2 consecutive)
        email_content = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', email_content)
        
        # Remove trailing whitespace but preserve structure
        lines = email_content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        email_content = '\n'.join(cleaned_lines)
        
        # Remove excessive trailing newlines
        email_content = email_content.rstrip('\n') + '\n' if email_content.rstrip() else email_content
        
        # Log cleaning results
        cleaned_length = len(email_content)
        if original_length != cleaned_length:
            logger.info(f"Email formatting cleaned: {original_length} -> {cleaned_length} chars")
        
        return email_content
    
    def _remove_repetitive_content(self, email_content: str) -> str:
        """Remove repetitive sentences and paragraphs"""
        import re
        
        # Split into sentences for analysis
        sentences = re.split(r'[.!?]+', email_content)
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            # Normalize sentence for comparison (remove extra whitespace)
            normalized = ' '.join(sentence.strip().split())
            if len(normalized) > 10:  # Only check substantial sentences
                # Check if this sentence is too similar to previous ones
                is_repetitive = False
                for seen in seen_content:
                    # Calculate similarity (simple word overlap check)
                    words1 = set(normalized.lower().split())
                    words2 = set(seen.lower().split())
                    if len(words1 & words2) / max(len(words1), len(words2), 1) > 0.7:
                        is_repetitive = True
                        break
                
                if not is_repetitive:
                    unique_sentences.append(sentence)
                    seen_content.add(normalized)
                else:
                    logger.info(f"Removed repetitive sentence: {normalized[:50]}...")
            else:
                unique_sentences.append(sentence)
        
        return '.'.join(unique_sentences)
    
    def _truncate_at_completion(self, email_content: str) -> str:
        """Truncate email at natural completion points to prevent over-generation"""
        import re
        
        # Common email ending patterns
        completion_patterns = [
            r'\n\n\*\*[^*]+\*\*\s*$',  # Bolded call to action at end
            r'\n\n\*\*[^*]+\*\*\s*\n',  # Bolded call to action with newline
            r'\n\nBest regards,.*$',
            r'\n\nSincerely,.*$', 
            r'\n\nThank you.*$',
            r'\n\nWarm regards,.*$',
            r'\n\nKind regards,.*$',
            r'\n---\n.*$',  # Signature separator
        ]
        
        for pattern in completion_patterns:
            match = re.search(pattern, email_content, re.DOTALL | re.IGNORECASE)
            if match:
                # Include the completion marker but truncate everything after
                end_pos = match.end()
                truncated_content = email_content[:end_pos]
                if len(truncated_content) < len(email_content):
                    logger.info(f"Truncated email at completion marker: {len(email_content)} -> {len(truncated_content)} chars")
                    return truncated_content
        
        # If no completion marker found, check for duplicate subject lines or email starts
        subject_matches = list(re.finditer(r'Subject:', email_content, re.IGNORECASE))
        if len(subject_matches) > 1:
            # Truncate at the second subject line
            truncated_content = email_content[:subject_matches[1].start()]
            logger.info(f"Truncated email at duplicate subject: {len(email_content)} -> {len(truncated_content)} chars")
            return truncated_content
        
        return email_content

