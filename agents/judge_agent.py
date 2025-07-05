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
from models.openrouter_backend import OpenRouterBackend
from utils.template_manager import get_template_manager
from config.config import get_setting, get_model_config

logger = logging.getLogger(__name__)

class JudgeAgent:
    """vLLM-based Judge Agent for email evaluation"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8", backend_type: str = "vllm", model_key: str = None):
        """Initialize with appropriate backend"""
        self.model_id = model_id
        self.model_key = model_key  # Model configuration key
        self.model_name = model_id.split('/')[-1]
        
        # Detect backend type from model config if not specified
        if model_key:
            model_config = get_model_config(model_key)
            backend_type = model_config.get('backend_type', backend_type)
        
        # Initialize appropriate backend
        if backend_type == 'openrouter':
            self.backend = OpenRouterBackend()
            self.backend_type = 'openrouter'
            logger.info(f"JudgeAgent initialized with OpenRouter backend for model: {self.model_name}")
        else:
            self.backend = VLLMBackend()
            self.backend_type = 'vllm'
            logger.info(f"JudgeAgent initialized with vLLM backend for model: {self.model_name}")
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
    
    def evaluate_email(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate email using vLLM backend and templates"""
        start_time = time.time()
        
        try:
            # Get judge template
            judge_template = self.template_manager.get_judge_template()
            
            # Format template with variables - handle checklist format
            logger.debug(f"Checklist type: {type(checklist)}, content: {str(checklist)[:200]}...")
            
            # Ensure checklist is properly formatted for the template
            if isinstance(checklist, list):
                # If it's already a list, wrap it in a dict for consistency
                checklist_for_template = {"criteria": checklist}
            else:
                checklist_for_template = checklist
            
            formatted_template = self.template_manager.format_template(
                judge_template,
                email_content=email_content[:1000],  # Truncate for brevity
                checklist=json.dumps(checklist_for_template, indent=2)
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
                
                # Adjust temperature for better JSON output with specific models
                if 'vicuna' in model_to_use.lower():
                    temperature = 0.3  # Lower temperature for more focused output
                elif 'llama' in model_to_use.lower():
                    temperature = 0.3  # Moderate temperature for Llama models to encourage generation
                elif 'gemini' in model_to_use.lower() or self.backend_type == 'openrouter':
                    temperature = 0.1  # Very low temperature for consistent JSON from API models
                
                logger.debug(f"Generating evaluation with model: {model_to_use}, max_tokens: {max_tokens}, temperature: {temperature}")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                
                result = self.backend.generate(
                    prompt=prompt,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                logger.info(f"Raw result from backend: '{result}' (length: {len(result) if result else 0})")
                
                # Save raw response to file for debugging (even if empty)
                import os
                os.makedirs("output/judge", exist_ok=True)
                with open(f"output/judge/raw_response_{int(time.time())}.txt", "w") as f:
                    f.write(result if result else "EMPTY_RESPONSE")
                
                if result.strip():
                    logger.info(f"Judge response (first 500 chars): {result.strip()[:500]}")
                    
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
            
            # Remove any trailing text after the JSON object
            # Look for the first complete JSON object pattern
            import re
            
            # Find JSON object pattern and extract only the first complete one
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: find the last } and cut everything after it
                last_brace = json_str.rfind('}')
                if last_brace != -1:
                    json_str = json_str[:last_brace + 1]
            
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
        # Handle both dict and list formats for checklist
        if isinstance(checklist, list):
            # If checklist is already a list of criteria
            criteria = checklist
        elif isinstance(checklist, dict):
            # If checklist is a dict with 'criteria' key
            criteria = checklist.get("criteria", [])
        else:
            # Fallback if checklist is neither
            criteria = []
            logger.warning(f"Unexpected checklist format: {type(checklist)}")
        evaluations = []
        total_score = 0
        total_weight = 0
        
        # Weight mapping
        weight_map = {"high": 3, "medium": 2, "low": 1}
        
        for criterion in criteria:
            # Handle both dict and simple formats
            if isinstance(criterion, dict):
                # Standard checklist format
                priority = criterion.get("priority", "medium")
                criterion_id = criterion.get("id")
                description = criterion.get("description", "")
            else:
                # Fallback for unexpected formats
                priority = "medium"
                criterion_id = None
                description = str(criterion)
            
            score = self._score_criterion_simple(email_content, criterion)
            weight = weight_map.get(priority, 2)
            
            evaluation_item = {
                "criterion_id": criterion_id,
                "description": description,
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
    
    def _score_criterion_simple(self, email_content: str, criterion: Any) -> float:
        """Simple fallback scoring logic"""
        # Handle both dict and other formats
        if isinstance(criterion, dict):
            description = criterion.get("description", "").lower()
        else:
            description = str(criterion).lower()
        
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

