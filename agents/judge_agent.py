"""
vLLM Judge Agent

This module provides email evaluation with:
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

class JudgeAgent:
    """vLLM-based Judge Agent for email evaluation"""
    
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
        
        logger.info(f"JudgeAgent initialized with model: {self.model_name}")
    
    def evaluate_email(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate email using vLLM backend and templates"""
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
                # Convert score to overall_score and normalize to 0-1 range
                if 'score' in evaluation:
                    score = evaluation['score']
                    if isinstance(score, str):
                        score = float(score)
                    evaluation['overall_score'] = score / 10.0  # Convert 1-10 to 0-1
                else:
                    evaluation['overall_score'] = 0.5
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response (first 500 chars): {evaluation_json[:500]}")
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
                # Generate using vLLM backend with model-specific adjustments
                model_to_use = self.model_key or self.model_id
                max_tokens = get_setting('judge_max_tokens', 6144)
                temperature = get_setting('temperature', 0.7)
                
                # Adjust temperature for better JSON output with Vicuna
                if 'vicuna' in model_to_use.lower():
                    temperature = 0.3  # Lower temperature for more focused output
                
                logger.debug(f"Generating evaluation with model: {model_to_use}, max_tokens: {max_tokens}, temperature: {temperature}")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                
                result = self.backend.generate(
                    prompt=prompt,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                if result.strip():
                    # Try to extract JSON if the response contains extra text
                    cleaned_result = self._extract_json_from_response(result.strip())
                    # Fix common JSON formatting issues
                    fixed_result = self._fix_malformed_json(cleaned_result)
                    return fixed_result
                else:
                    raise Exception("Empty response from backend")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)  # Simple delay between retries
        
        # If all retries failed, raise exception
        raise Exception(f"Failed to evaluate email after {self.max_retries} attempts. Last error: {last_error}")
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON object from response text"""
        try:
            import re
            
            logger.debug(f"Original response: {response[:500]}...")
            
            # Check if the response contains instruction patterns instead of JSON
            instruction_patterns = [
                "provide", "analysis", "please", "here is", "i will", "first",
                "let me", "based on", "following", "assessment"
            ]
            
            # If response starts and ends with braces, treat it as JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                json_str = response.strip()
                logger.debug(f"Found complete JSON object: {json_str[:200]}...")
                
                # Basic validation - just check if it looks like a JSON object
                return json_str
            
            # Try to find JSON object pattern {...} as fallback
            json_pattern = r'\{.*\}'  # Use greedy matching
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Return the longest match (most likely to be complete)
                json_str = max(matches, key=len)
                logger.debug(f"Extracted JSON: {json_str[:200]}...")
                
                # Basic validation - just check if it looks like a JSON object
                return json_str
            
            # Check for various instruction patterns that indicate model confusion
            if any(phrase in response.lower() for phrase in instruction_patterns):
                logger.warning("Model provided analysis text instead of JSON, triggering retry")
                raise Exception("Model provided analysis text instead of JSON")
            
            # If we get here, the response doesn't contain recognizable JSON
            logger.warning(f"No valid JSON found in response: {response[:200]}...")
            raise Exception("No valid JSON object found in response")
            
        except Exception as e:
            logger.warning(f"Failed to extract JSON from response: {e}")
            raise Exception(f"Failed to extract valid JSON: {e}")
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        try:
            import re
            
            # Fix empty string values followed by comma
            json_str = re.sub(r':\s*""\s*,', ': "No analysis provided",', json_str)
            
            # Fix missing values after colon (like "score": })
            json_str = re.sub(r':\s*}', ': 5}', json_str)
            json_str = re.sub(r':\s*,', ': "No analysis provided",', json_str)
            
            # Fix missing score value at end
            json_str = re.sub(r'"score":\s*}', '"score": 5}', json_str)
            
            # Ensure score is a number
            score_match = re.search(r'"score":\s*"([^"]*)"', json_str)
            if score_match:
                try:
                    score_val = float(score_match.group(1))
                    json_str = re.sub(r'"score":\s*"[^"]*"', f'"score": {score_val}', json_str)
                except (ValueError, TypeError):
                    json_str = re.sub(r'"score":\s*"[^"]*"', '"score": 5', json_str)
            
            logger.debug(f"Fixed JSON: {json_str[:200]}...")
            return json_str
            
        except Exception as e:
            logger.warning(f"Failed to fix malformed JSON: {e}")
            return json_str
    
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

