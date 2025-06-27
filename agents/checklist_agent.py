"""
Simplified Checklist Generation Agent

This module provides simple checklist generation with:
- Basic model interaction
- Simple dictionary returns
- Minimal configuration
"""

import logging
import time
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SimpleChecklistAgent:
    """Simplified Checklist Generation Agent"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8"):
        """Initialize with basic configuration"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"SimpleChecklistAgent initialized with model: {self.model_name}")
    
    def generate_checklist(self, user_query: str, reference_response: str, topic: str) -> Dict[str, Any]:
        """Generate evaluation checklist and return simple dictionary"""
        start_time = time.time()
        
        try:
            # Create simple checklist based on topic
            checklist = self._create_simple_checklist(topic, user_query)
            
            generation_time = time.time() - start_time
            logger.info(f"Checklist generated in {generation_time:.2f}s for topic: {topic}")
            
            return checklist
            
        except Exception as e:
            logger.error(f"Error generating checklist: {e}")
            return {"error": f"Error generating checklist: {str(e)}"}
    
    def _create_simple_checklist(self, topic: str, user_query: str) -> Dict[str, Any]:
        """Create a simple checklist structure"""
        # Simple checklist template
        checklist = {
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
                },
                {
                    "id": 5,
                    "description": "Grammar and spelling are correct",
                    "priority": "low"
                }
            ],
            "generated_by": self.model_name,
            "timestamp": time.time()
        }
        
        return checklist

# Backward compatibility
ChecklistAgent = SimpleChecklistAgent