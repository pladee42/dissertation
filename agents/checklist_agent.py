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
                # Enhanced prompt for better JSON output
                json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON array. Do not include any thinking process, explanations, or additional text. Start immediately with [ and end with ]. Example format:
[
    {{"question": "Does the email...", "best_ans": "yes", "priority": "high"}},
    {{"question": "Is the tone...", "best_ans": "yes", "priority": "medium"}}
]"""
                
                # Generate using vLLM backend
                result = self.backend.generate(
                    prompt=json_prompt,
                    model=self.model_key or self.model_id,
                    max_tokens=get_setting('checklist_max_tokens', 8192),
                    temperature=get_setting('temperature', 0.7)
                )
                
                if result.strip():
                    return result.strip()
                else:
                    raise Exception("Empty response from backend")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, raise exception
        raise Exception(f"Failed to generate checklist after {self.max_retries} attempts. Last error: {last_error}")
    
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

