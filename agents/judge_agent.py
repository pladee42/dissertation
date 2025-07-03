"""
SGLang Judge Agent

This module provides email evaluation with:
- SGLang backend integration
- Template-based prompts
- Simple retry logic
"""

import logging
import time
import json
from typing import Dict, Any

from models.sglang_backend import SGLangBackend
from models.vllm_backend import VLLMBackend
from utils.template_manager import get_template_manager
from config.config import get_setting

logger = logging.getLogger(__name__)

class JudgeAgent:
    """SGLang-based Judge Agent for email evaluation"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8", backend_type: str = "vllm"):
        """Initialize with configurable backend"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        # Initialize backend based on type
        server_url = get_setting('server_url', 'http://localhost:30000')
        server_timeout = get_setting('server_timeout', 60)
        
        if backend_type.lower() == "sglang":
            self.backend = SGLangBackend(base_url=server_url, timeout=server_timeout)
        else:  # default to vllm
            self.backend = VLLMBackend(base_url=server_url, timeout=server_timeout)
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
        
        logger.info(f"JudgeAgent initialized with model: {self.model_name}")
    
    def evaluate_email(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate email using SGLang backend and templates"""
        start_time = time.time()
        
        try:
            # Get judge template
            judge_template = self.template_manager.get_judge_template()
            
            # Format template with variables
            formatted_template = self.template_manager.format_template(
                judge_template,
                email_content=email_content[:1000],  # Truncate for brevity
                checklist=json.dumps(checklist, indent=2)
            )
            
            # Create full prompt
            full_prompt = f"{formatted_template}\n\nEvaluation (JSON format):"
            
            # Generate with retry logic
            evaluation_json = self._generate_with_retry(full_prompt)
            
            # Parse JSON response
            try:
                evaluation = json.loads(evaluation_json)
                # Ensure overall_score exists
                if 'overall_score' not in evaluation:
                    evaluation['overall_score'] = 0.5
            except json.JSONDecodeError:
                # Fallback to simple evaluation if JSON parsing fails
                evaluation = self._create_fallback_evaluation(email_content, checklist)
            
            evaluation_time = time.time() - start_time
            evaluation["evaluation_time"] = evaluation_time
            evaluation["evaluated_by"] = self.model_name
            
            logger.info(f"Email evaluated in {evaluation_time:.2f}s")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating email: {e}")
            return self._create_fallback_evaluation(email_content, checklist)
    
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
                    raise Exception("Empty response from backend")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, raise exception
        raise Exception(f"Failed to evaluate email after {self.max_retries} attempts. Last error: {last_error}")
    
    def _create_fallback_evaluation(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple fallback evaluation"""
        criteria = checklist.get("criteria", [])
        evaluations = []
        total_score = 0
        total_weight = 0
        
        # Weight mapping
        weight_map = {"high": 3, "medium": 2, "low": 1}
        
        for criterion in criteria:
            score = self._score_criterion_simple(email_content, criterion)
            weight = weight_map.get(criterion.get("priority", "medium"), 2)
            
            evaluation_item = {
                "criterion_id": criterion.get("id"),
                "description": criterion.get("description"),
                "score": score,
                "weight": weight,
                "weighted_score": score * weight
            }
            
            evaluations.append(evaluation_item)
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.5
        
        return {
            "overall_score": round(overall_score, 2),
            "detailed_evaluations": evaluations,
            "total_criteria": len(criteria),
            "checklist_topic": checklist.get("topic", "unknown"),
            "fallback": True
        }
    
    def _score_criterion_simple(self, email_content: str, criterion: Dict[str, Any]) -> float:
        """Simple fallback scoring logic"""
        description = criterion.get("description", "").lower()
        email_lower = email_content.lower()
        
        # Simple keyword-based scoring
        if "subject" in description:
            return 0.8 if "subject:" in email_lower else 0.3
        elif "relevant" in description:
            return 0.7 if len(email_content) > 50 else 0.4
        elif "professional" in description:
            return 0.8 if "dear" in email_lower or "regards" in email_lower else 0.5
        elif "greeting" in description:
            return 0.9 if "dear" in email_lower and "regards" in email_lower else 0.4
        else:
            return 0.6  # Default neutral score

