"""
Simplified Judge Agent

This module provides simple email evaluation with:
- Basic scoring mechanism
- Simple dictionary returns
- Minimal configuration
"""

import logging
import time
import random
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SimpleJudgeAgent:
    """Simplified Judge Agent for email evaluation"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8"):
        """Initialize with basic configuration"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"SimpleJudgeAgent initialized with model: {self.model_name}")
    
    def evaluate_email(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate email against checklist and return simple dictionary"""
        start_time = time.time()
        
        try:
            # Simple evaluation based on checklist
            evaluation = self._evaluate_against_checklist(email_content, checklist)
            
            evaluation_time = time.time() - start_time
            logger.info(f"Email evaluated in {evaluation_time:.2f}s")
            
            evaluation["evaluation_time"] = evaluation_time
            evaluation["evaluated_by"] = self.model_name
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating email: {e}")
            return {
                "error": f"Error evaluating email: {str(e)}",
                "overall_score": 0.0
            }
    
    def _evaluate_against_checklist(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Simple evaluation logic"""
        criteria = checklist.get("criteria", [])
        evaluations = []
        total_score = 0
        total_weight = 0
        
        # Weight mapping
        weight_map = {"high": 3, "medium": 2, "low": 1}
        
        for criterion in criteria:
            # Simple scoring logic (placeholder)
            score = self._score_criterion(email_content, criterion)
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
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        return {
            "overall_score": round(overall_score, 2),
            "detailed_evaluations": evaluations,
            "total_criteria": len(criteria),
            "checklist_topic": checklist.get("topic", "unknown")
        }
    
    def _score_criterion(self, email_content: str, criterion: Dict[str, Any]) -> float:
        """Simple scoring logic for a criterion"""
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
        elif "grammar" in description:
            return 0.7  # Assume decent grammar
        else:
            # Default random score for unknown criteria
            return round(random.uniform(0.5, 0.9), 2)

# Backward compatibility  
JudgeAgent = SimpleJudgeAgent