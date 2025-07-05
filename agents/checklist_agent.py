"""
vLLM Checklist Generation Agent

This module provides checklist generation with:
- vLLM backend integration
- Template-based prompts
- Simple retry logic
"""

import logging
import time
import json
from typing import Dict, Any

from models.vllm_backend import VLLMBackend
from utils.template_manager import get_template_manager
from config.config import get_setting

logger = logging.getLogger(__name__)

class ChecklistAgent:
    """vLLM-based Checklist Generation Agent"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8", backend_type: str = "vllm", model_key: str = None):
        """Initialize with vLLM backend"""
        self.model_id = model_id
        self.model_key = model_key  # Model configuration key
        self.model_name = model_id.split('/')[-1]
        
        # Initialize vLLM backend
        self.backend = VLLMBackend()
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
        
        logger.info(f"ChecklistAgent initialized with model: {self.model_name}")
    
    def generate_checklist(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Generate evaluation checklist using vLLM backend and templates"""
        start_time = time.time()
        
        try:
            # Get checklist template
            checklist_template = self.template_manager.get_checklist_template()
            
            # Format template with variables
            formatted_template = self.template_manager.format_template(
                checklist_template,
                topic=topic,
                user_query=user_query
            )
            
            # Create full prompt
            full_prompt = f"{formatted_template}\n\nChecklist (JSON format):"
            
            # Generate with retry logic
            checklist_json = self._generate_with_retry(full_prompt)
            
            # Parse JSON response with better error handling
            try:
                # First try basic JSON parsing
                checklist = json.loads(checklist_json)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response (first 500 chars): {checklist_json[:500]}")
                # Fallback to simple checklist if JSON parsing fails
                checklist = self._create_fallback_checklist(topic)
            
            generation_time = time.time() - start_time
            logger.info(f"Checklist generated in {generation_time:.2f}s for topic: {topic}")
            
            return checklist
            
        except Exception as e:
            logger.error(f"Error generating checklist: {e}")
            return self._create_fallback_checklist(topic)
    
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with simple retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Simplified prompt that works better with Vicuna
                json_prompt = f"""Create an evaluation checklist for an email about: {topic}

Generate a JSON array with 8-10 evaluation questions. Each question should be a yes/no question to evaluate the email quality.

Use this exact format:
[
    {{"question": "Does the email have a clear subject line?", "best_ans": "yes", "priority": "high"}},
    {{"question": "Is the tone professional and appropriate?", "best_ans": "yes", "priority": "medium"}},
    {{"question": "Does the content relate to the topic '{topic}'?", "best_ans": "yes", "priority": "high"}},
    {{"question": "Is there a proper greeting and closing?", "best_ans": "yes", "priority": "medium"}},
    {{"question": "Is the message clear and easy to understand?", "best_ans": "yes", "priority": "high"}},
    {{"question": "Does the email have a call-to-action?", "best_ans": "yes", "priority": "medium"}},
    {{"question": "Is the email free of grammatical errors?", "best_ans": "yes", "priority": "medium"}},
    {{"question": "Is the email length appropriate?", "best_ans": "yes", "priority": "low"}}
]

Return only the JSON array:"""
                
                # Generate using vLLM backend
                model_to_use = self.model_key or self.model_id
                max_tokens = get_setting('checklist_max_tokens', 8192)
                temperature = get_setting('temperature', 0.7)
                
                logger.debug(f"Generating checklist with model: {model_to_use}, max_tokens: {max_tokens}, temperature: {temperature}")
                logger.debug(f"Prompt length: {len(json_prompt)} characters")
                
                result = self.backend.generate(
                    prompt=json_prompt,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if result.strip():
                    # Try to extract JSON if the response contains extra text
                    cleaned_result = self._extract_json_from_response(result.strip())
                    return cleaned_result
                else:
                    raise Exception("Empty response from backend")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, raise exception
        raise Exception(f"Failed to generate checklist after {self.max_retries} attempts. Last error: {last_error}")
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON array from response text"""
        try:
            # Look for JSON array pattern
            import re
            
            # Try to find JSON array pattern [...]
            json_pattern = r'\[.*?\]'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Return the longest match (most likely to be complete)
                json_str = max(matches, key=len)
                logger.debug(f"Extracted JSON: {json_str[:200]}...")
                return json_str
            
            # If no JSON array found, check if the response looks like it should be JSON
            if response.strip().startswith('[') and response.strip().endswith(']'):
                return response.strip()
            
            # If response contains instructions instead of JSON, create fallback
            if any(phrase in response.lower() for phrase in ["provide a list", "json format", "please", "here is"]):
                logger.warning("Model provided instructions instead of JSON, using fallback")
                raise Exception("Model provided instructions instead of JSON")
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to extract JSON from response: {e}")
            raise Exception(f"Failed to extract valid JSON: {e}")
    
    def _create_fallback_checklist(self, topic: str) -> Dict[str, Any]:
        """Create a simple fallback checklist"""
        return {
            "topic": topic,
            "criteria": [
                {
                    "id": 1,
                    "description": "Email has clear subject line",
                    "priority": "high"
                },
                {
                    "id": 2,
                    "description": "Content is relevant to topic",
                    "priority": "high"
                },
                {
                    "id": 3,
                    "description": "Professional tone is maintained",
                    "priority": "medium"
                },
                {
                    "id": 4,
                    "description": "Email has proper greeting and closing",
                    "priority": "medium"
                }
            ],
            "generated_by": self.model_name,
            "timestamp": time.time(),
            "fallback": True
        }

