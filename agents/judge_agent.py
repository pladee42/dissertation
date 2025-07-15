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
            detected_backend_type = model_config.get('backend_type', backend_type)
            logger.info(f"Model key: {model_key}, detected backend_type: {detected_backend_type}")
            backend_type = detected_backend_type
        
        logger.info(f"Final backend_type for {model_key}: {backend_type}")
        
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
            
            # Generate with consistency sampling
            evaluation_json = self._generate_consistent_evaluation(full_prompt)
            
            # Parse JSON response
            try:
                evaluation = json.loads(evaluation_json)
                
                # Process binary evaluation format
                if 'checklist_scores' in evaluation:
                    # Validate binary results and add confidence if available
                    criterion_scores = evaluation['checklist_scores']
                    valid_results = []
                    
                    for item in criterion_scores:
                        if 'result' in item and item['result'].lower() in ['yes', 'no']:
                            valid_results.append(item)
                        else:
                            logger.warning(f"Invalid binary result: {item.get('result', 'missing')}")
                    
                    evaluation['checklist_scores'] = valid_results
                    evaluation['total_criteria'] = len(valid_results)
                    
                    # Calculate weighted score based on priority
                    weighted_result = self._calculate_weighted_score(valid_results, checklist)
                    evaluation['weighted_score'] = weighted_result['weighted_score']
                    evaluation['priority_breakdown'] = weighted_result['priority_breakdown']
                    
                    # Consistency confidence is already added by _process_consistency_results
                    
                    logger.info(f"Binary evaluation: {len(valid_results)} criteria, weighted score: {weighted_result['weighted_score']:.3f}")
                else:
                    evaluation['checklist_scores'] = []
                    evaluation['total_criteria'] = 0
                    evaluation['weighted_score'] = 0.0
                    evaluation['priority_breakdown'] = {}
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
    
    def _calculate_weighted_score(self, judge_results, checklist):
        """Calculate priority-weighted score from binary judge results"""
        PRIORITY_WEIGHTS = {
            "very high": 5, "high": 3, "medium": 2, "low": 1, "very low": 0.5
        }
        
        passed_points = 0
        total_points = 0
        priority_breakdown = {"very high": 0, "high": 0, "medium": 0, "low": 0, "very low": 0}
        
        # Handle different checklist formats
        if isinstance(checklist, dict) and 'criteria' in checklist:
            checklist_criteria = checklist['criteria']
        elif isinstance(checklist, list):
            checklist_criteria = checklist
        else:
            logger.warning("Unknown checklist format, cannot calculate weighted score")
            return {'weighted_score': 0.0, 'priority_breakdown': priority_breakdown}
        
        # Match judge results with checklist criteria
        for i, judge_item in enumerate(judge_results):
            if i < len(checklist_criteria):
                checklist_item = checklist_criteria[i]
                priority = checklist_item.get('priority', 'medium')
                weight = PRIORITY_WEIGHTS.get(priority, 2)
                total_points += weight
                
                # Check if judge result matches expected answer
                judge_result = judge_item.get('result', '').lower()
                expected_result = checklist_item.get('best_ans', 'yes').lower()
                
                if judge_result == expected_result:
                    passed_points += weight
                    priority_breakdown[priority] += 1
        
        weighted_score = passed_points / total_points if total_points > 0 else 0.0
        
        return {
            'weighted_score': weighted_score,
            'priority_breakdown': priority_breakdown
        }
    
    def _generate_consistent_evaluation(self, prompt: str) -> str:
        """Generate evaluation with consistency sampling (3 attempts)"""
        evaluations = []
        generation_times = []
        
        # Generate 3 evaluations with identical parameters
        for attempt in range(3):
            start_time = time.time()
            try:
                result = self._generate_with_retry(prompt)
                generation_time = time.time() - start_time
                
                # Try to parse JSON to validate
                parsed_eval = json.loads(result)
                evaluations.append(parsed_eval)
                generation_times.append(generation_time)
                
                logger.debug(f"Consistency attempt {attempt + 1} successful, time: {generation_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Consistency attempt {attempt + 1} failed: {e}")
                # Continue with other attempts
        
        if not evaluations:
            # All attempts failed, use fallback
            logger.error("All consistency attempts failed, using fallback")
            raise Exception("All consistency sampling attempts failed")
        
        # Calculate consistency and pick best result
        final_evaluation = self._process_consistency_results(evaluations, generation_times)
        
        return json.dumps(final_evaluation)
    
    def _process_consistency_results(self, evaluations, generation_times):
        """Process multiple evaluations to create final consistent result"""
        if len(evaluations) == 1:
            # Only one successful evaluation
            result = evaluations[0]
            result['consistency_confidence'] = 1.0
            result['average_response_time'] = generation_times[0]
            return result
        
        # Calculate consistency confidence
        consistency_confidence = self._calculate_consistency_confidence(evaluations)
        
        # Use majority vote for final decisions
        final_result = self._create_final_evaluation(evaluations)
        
        # Add consistency metadata
        final_result['consistency_confidence'] = consistency_confidence
        final_result['average_response_time'] = sum(generation_times) / len(generation_times)
        final_result['generation_attempts'] = len(evaluations)
        
        logger.info(f"Consistency confidence: {consistency_confidence:.3f}, avg time: {final_result['average_response_time']:.2f}s")
        
        return final_result
    
    def _create_final_evaluation(self, evaluations):
        """Create final evaluation using majority vote"""
        # Use first evaluation as template
        final_eval = evaluations[0].copy()
        
        if 'checklist_scores' in final_eval:
            # Apply majority vote to each criterion
            final_scores = []
            for i in range(len(final_eval['checklist_scores'])):
                # Collect results for this criterion from all evaluations
                results = []
                for eval_data in evaluations:
                    if i < len(eval_data.get('checklist_scores', [])):
                        results.append(eval_data['checklist_scores'][i].get('result', 'no'))
                
                # Majority vote
                yes_count = results.count('yes')
                no_count = results.count('no')
                final_result = 'yes' if yes_count > no_count else 'no'
                
                # Calculate individual criterion confidence
                criterion_confidence = max(yes_count, no_count) / len(results) if results else 0.0
                
                final_score = final_eval['checklist_scores'][i].copy()
                final_score['result'] = final_result
                final_score['confidence'] = criterion_confidence
                final_scores.append(final_score)
            
            final_eval['checklist_scores'] = final_scores
        
        return final_eval
    
    def _calculate_consistency_confidence(self, evaluations_list):
        """Calculate confidence based on agreement across multiple generations"""
        if len(evaluations_list) < 2:
            return 1.0  # Single evaluation assumed confident
        
        # Check if all evaluations have checklist_scores
        valid_evaluations = [e for e in evaluations_list if 'checklist_scores' in e and e['checklist_scores']]
        if not valid_evaluations:
            return 0.0
        
        total_criteria = len(valid_evaluations[0]['checklist_scores'])
        agreements = 0
        
        for i in range(total_criteria):
            # Collect results for this criterion from all evaluations
            results = []
            for eval_data in valid_evaluations:
                if i < len(eval_data['checklist_scores']):
                    result = eval_data['checklist_scores'][i].get('result', 'no')
                    results.append(result)
            
            # Check if all results agree
            if len(set(results)) == 1:  # All same
                agreements += 1
        
        confidence = agreements / total_criteria if total_criteria > 0 else 0.0
        logger.debug(f"Consistency confidence: {agreements}/{total_criteria} = {confidence:.3f}")
        
        return confidence
    
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with simple retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Generate using vLLM backend with model-specific adjustments
                model_to_use = self.model_key or self.model_id
                max_tokens = get_setting('judge_max_tokens', 6144)
                temperature = get_setting('judge_temperature', 0.1)
                top_p = get_setting('judge_top_p', 0.6)
                
                # Model-specific adjustments for temperature (ensure deterministic scoring)
                if 'vicuna' in model_to_use.lower():
                    temperature = min(temperature, 0.1)  # Very low for consistent scoring
                elif 'llama' in model_to_use.lower():
                    temperature = min(temperature, 0.15)  # Slightly higher for Llama
                elif 'gemini' in model_to_use.lower() or self.backend_type == 'openrouter':
                    temperature = 0.05  # Extremely low for API models
                    top_p = 0.5  # Very focused for API models
                
                logger.debug(f"Generating evaluation with model: {model_to_use}, max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}")
                
                result = self.backend.generate(
                    prompt=prompt,
                    model=model_to_use,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
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
            
            logger.debug(f"Extracting JSON from response (length: {len(response)})")
            
            # Check if the response contains instruction patterns instead of JSON
            instruction_patterns = [
                "provide", "analysis", "please", "here is", "i will", "first",
                "let me", "based on", "following", "assessment"
            ]
            
            # If response starts and ends with braces, treat it as JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                json_str = response.strip()
                logger.debug("Found complete JSON object")
                
                # Basic validation - just check if it looks like a JSON object
                return json_str
            
            # Try to find JSON object pattern {...} as fallback
            json_pattern = r'\{.*\}'  # Use greedy matching
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Return the longest match (most likely to be complete)
                json_str = max(matches, key=len)
                logger.debug("Extracted JSON from pattern match")
                
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
            
            logger.debug("Applied JSON formatting fixes")
            return json_str
            
        except Exception as e:
            logger.warning(f"Failed to fix malformed JSON: {e}")
            return json_str
    
    def _create_fallback_evaluation(self, email_content: str, checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Create a simple fallback evaluation with binary format"""
        return {
            "checklist_scores": [],
            "strengths": "Unable to evaluate - using fallback",
            "weaknesses": "Evaluation failed",
            "weighted_score": 0.0,
            "priority_breakdown": {"very high": 0, "high": 0, "medium": 0, "low": 0, "very low": 0},
            "total_criteria": 0,
            "evaluated_by": self.model_name,
            "consistency_confidence": 0.0,
            "average_response_time": 0.0,
            "generation_attempts": 0,
            "fallback": True
        }
    

