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
from models.openrouter_backend import OpenRouterBackend
from utils.template_manager import get_template_manager
from config.config import get_setting, get_model_config

logger = logging.getLogger(__name__)

class ChecklistAgent:
    """vLLM-based Checklist Generation Agent"""
    
    def __init__(self, model_id: str, dtype: str = "bfloat16", quantization: str = "experts_int8", backend_type: str = "vllm", model_key: str = None, checklist_mode: str = "enhanced"):
        """Initialize with appropriate backend"""
        self.model_id = model_id
        self.model_key = model_key  # Model configuration key
        self.model_name = model_id.split('/')[-1]
        self.checklist_mode = checklist_mode
        
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
            logger.info(f"ChecklistAgent initialized with OpenRouter backend for model: {self.model_name}")
        else:
            self.backend = VLLMBackend()
            self.backend_type = 'vllm'
            logger.info(f"ChecklistAgent initialized with vLLM backend for model: {self.model_name}")
        
        # Get template manager
        self.template_manager = get_template_manager()
        
        # Retry settings
        self.max_retries = get_setting('max_retries', 3)
    
    def generate_checklist(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Generate evaluation checklist using vLLM backend and templates"""
        start_time = time.time()
        
        try:
            if self.checklist_mode == "preprocess":
                # Two-step process for preprocess mode
                return self._generate_checklist_preprocess(user_query, topic)
            else:
                # Single-step process for enhanced and extract_only modes
                return self._generate_checklist_single_step(user_query, topic)
                
        except Exception as e:
            logger.error(f"Error generating checklist: {e}")
            return self._create_fallback_checklist(topic)
    
    def _generate_checklist_single_step(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Generate checklist in single step (enhanced/extract_only modes)"""
        start_time = time.time()
        
        # Get checklist template based on mode
        checklist_template = self._get_mode_specific_template()
        
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
        
        # Save raw response for debugging
        self._save_debug_response(checklist_json, "checklist_single", topic)
        
        # Parse JSON response with enhanced error handling
        try:
            # First try basic JSON parsing
            checklist = json.loads(checklist_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed: {e}")
            # Try enhanced JSON extraction before fallback
            try:
                cleaned_result = self._extract_json_from_response(checklist_json)
                fixed_result = self._fix_malformed_json(cleaned_result)
                checklist = json.loads(fixed_result)
                logger.info("Successfully extracted checklist after cleaning")
            except Exception as extract_error:
                logger.warning(f"Enhanced JSON extraction also failed: {extract_error}")
                logger.warning(f"Raw response (first 500 chars): {checklist_json[:500]}")
                # Only use fallback as last resort
                checklist = self._create_fallback_checklist(topic)
        
        generation_time = time.time() - start_time
        logger.info(f"Checklist generated in {generation_time:.2f}s for topic: {topic}")
        
        return checklist
    
    def _generate_checklist_preprocess(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Generate checklist using two-step preprocess method"""
        start_time = time.time()
        
        # Step 1: Analyze example email
        extracted_characteristics = self._analyze_example_email(user_query, topic)
        
        # Step 2: Generate checklist based on extracted characteristics
        checklist_template = self.template_manager.get_template("checklist_preprocess")
        
        # Format template with extracted characteristics
        formatted_template = self.template_manager.format_template(
            checklist_template,
            topic=topic,
            extracted_characteristics=json.dumps(extracted_characteristics, indent=2)
        )
        
        # Create full prompt
        full_prompt = f"{formatted_template}\n\nChecklist (JSON format):"
        
        # Generate with retry logic
        checklist_json = self._generate_with_retry(full_prompt)
        
        # Save raw response for debugging
        self._save_debug_response(checklist_json, "checklist_preprocess", topic)
        
        # Parse JSON response with enhanced error handling
        try:
            checklist = json.loads(checklist_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed in preprocess mode: {e}")
            # Try enhanced JSON extraction before fallback
            try:
                cleaned_result = self._extract_json_from_response(checklist_json)
                fixed_result = self._fix_malformed_json(cleaned_result)
                checklist = json.loads(fixed_result)
                logger.info("Successfully extracted preprocess checklist after cleaning")
            except Exception as extract_error:
                logger.warning(f"Enhanced JSON extraction also failed in preprocess mode: {extract_error}")
                logger.warning(f"Raw response (first 500 chars): {checklist_json[:500]}")
                # Only use fallback as last resort
                checklist = self._create_fallback_checklist(topic)
        
        generation_time = time.time() - start_time
        logger.info(f"Checklist generated in {generation_time:.2f}s for topic: {topic} (preprocess mode)")
        
        return checklist
    
    def _analyze_example_email(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Analyze example email to extract characteristics (Step 1 of preprocess mode)"""
        try:
            # Get example analyzer template
            analyzer_template = self.template_manager.get_template("example_analyzer")
            
            # Format template with user query
            formatted_template = self.template_manager.format_template(
                analyzer_template,
                user_query=user_query
            )
            
            # Create full prompt
            full_prompt = f"{formatted_template}\n\nExtracted characteristics (JSON format):"
            
            # Generate analysis with retry logic
            analysis_json = self._generate_with_retry(full_prompt)
            
            # Save raw response for debugging (especially for problematic models)
            self._save_debug_response(analysis_json, "analysis", topic)
            
            # Parse JSON response with enhanced extraction
            try:
                # Try direct parsing first
                analysis = json.loads(analysis_json)
                logger.info("Successfully analyzed example email characteristics")
                return analysis
            except json.JSONDecodeError as e:
                logger.warning(f"Direct JSON parsing failed: {e}")
                # Try enhanced JSON extraction for objects
                try:
                    cleaned_json = self._extract_json_object_from_response(analysis_json)
                    analysis = json.loads(cleaned_json)
                    logger.info("Successfully extracted JSON object after cleaning")
                    return analysis
                except Exception as extract_error:
                    logger.warning(f"Enhanced JSON extraction also failed: {extract_error}")
                    logger.warning(f"Raw response (first 500 chars): {analysis_json[:500]}")
                    
                    # Check if this looks like truncated JSON (partial structure visible)
                    if self._is_truncated_analysis(analysis_json):
                        logger.info("Detected truncated analysis, attempting intelligent reconstruction")
                        reconstructed = self._reconstruct_truncated_analysis(analysis_json)
                        if reconstructed:
                            return reconstructed
                    
                    # Only use fallback as last resort
                    return self._create_fallback_analysis()
                
        except Exception as e:
            logger.error(f"Error analyzing example email: {e}")
            return self._create_fallback_analysis()
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Create fallback analysis when extraction fails"""
        return {
            "tone_characteristics": {
                "emotional_language": ["professional", "direct"],
                "communication_style": "formal business communication",
                "formality_level": "professional"
            },
            "structure_patterns": {
                "word_count": "approximately 100-200 words",
                "paragraph_count": 3,
                "opening_style": "professional greeting",
                "closing_style": "standard business closing"
            },
            "content_elements": {
                "problem_presentation": "clear issue statement",
                "solution_approach": "direct request or information",
                "cta_style": "clear action request"
            },
            "language_features": {
                "key_phrases": ["please", "thank you", "important"],
                "sentence_structure": "clear and concise sentences",
                "accessibility": "professional business language"
            }
        }
    
    def _is_truncated_analysis(self, response: str) -> bool:
        """Check if response looks like truncated analysis JSON"""
        response = response.strip()
        
        # Signs of truncated analysis
        indicators = [
            # Starts with array or property (missing opening brace)
            (response.startswith('[') or ('"' in response and ':' in response and not response.startswith('{'))),
            # Contains expected analysis keys
            any(key in response for key in ['communication_style', 'formality_level', 'word_count', 'content_elements']),
            # Doesn't end with closing brace or ends abruptly
            not response.rstrip().endswith('}')
        ]
        
        return any(indicators)
    
    def _reconstruct_truncated_analysis(self, response: str) -> dict:
        """Attempt to reconstruct truncated analysis JSON"""
        try:
            response = response.strip()
            
            # Basic reconstruction: add missing braces and complete structure
            if not response.startswith('{'):
                response = '{' + response
            
            if not response.rstrip().endswith('}'):
                response = response.rstrip() + '}'
            
            # Try to parse the reconstructed JSON
            try:
                analysis = json.loads(response)
                logger.info("Successfully reconstructed truncated analysis JSON")
                return analysis
            except json.JSONDecodeError:
                # If simple reconstruction fails, try partial parsing
                return self._partial_parse_analysis(response)
                
        except Exception as e:
            logger.warning(f"Failed to reconstruct truncated analysis: {e}")
            return None
    
    def _partial_parse_analysis(self, response: str) -> dict:
        """Extract what we can from partial JSON and fill in reasonable defaults"""
        import re
        
        try:
            # Start with fallback structure
            analysis = self._create_fallback_analysis()
            
            # Try to extract specific values using regex patterns
            patterns = {
                'communication_style': r'"communication_style":\s*"([^"]*)"',
                'formality_level': r'"formality_level":\s*"([^"]*)"',
                'word_count': r'"word_count":\s*"([^"]*)"',
                'paragraph_count': r'"paragraph_count":\s*(\d+)',
                'emotional_language': r'"emotional_language":\s*\[([^\]]*)\]'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response)
                if match:
                    value = match.group(1)
                    # Update the appropriate section of analysis
                    if key in ['communication_style', 'formality_level']:
                        analysis['tone_characteristics'][key] = value
                    elif key in ['word_count', 'paragraph_count']:
                        analysis['structure_patterns'][key] = int(value) if key == 'paragraph_count' else value
                    elif key == 'emotional_language':
                        # Parse the array content
                        array_content = value.replace('"', '').split(',')
                        analysis['tone_characteristics'][key] = [item.strip() for item in array_content]
            
            logger.info("Successfully performed partial analysis reconstruction")
            return analysis
            
        except Exception as e:
            logger.warning(f"Partial analysis parsing failed: {e}")
            return None
    
    def _save_debug_response(self, response: str, response_type: str, topic: str) -> None:
        """Save raw response for debugging purposes"""
        # Check if debug mode is enabled
        debug_enabled = get_setting('debug_save_responses', False)
        if not debug_enabled:
            return
            
        try:
            from datetime import datetime
            from pathlib import Path
            
            # Create debug directory if it doesn't exist
            debug_dir = Path("debug_responses")
            debug_dir.mkdir(exist_ok=True)
            
            # Create filename with timestamp and details
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            model_name = (self.model_key or self.model_id).replace('/', '_').replace('-', '_')
            safe_topic = topic.replace(' ', '_').replace('/', '_')[:50]  # Limit length
            
            filename = f"{timestamp}_{model_name}_{response_type}_{safe_topic}.txt"
            filepath = debug_dir / filename
            
            # Save the response
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Model: {self.model_key or self.model_id}\n")
                f.write(f"Response Type: {response_type}\n")
                f.write(f"Topic: {topic}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Response Length: {len(response)} characters\n")
                f.write("-" * 80 + "\n")
                f.write("RAW RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(response)
                f.write("\n" + "-" * 80 + "\n")
            
            logger.debug(f"Debug response saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug response: {e}")
    
    def _generate_with_retry(self, prompt: str) -> str:
        """Generate text with simple retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Use the prompt as-is from the template
                json_prompt = prompt
                
                # Generate using vLLM backend with model-specific adjustments
                model_to_use = self.model_key or self.model_id
                max_tokens = get_setting('checklist_max_tokens', 8192)
                temperature = get_setting('temperature', 0.7)
                
                # Adjust temperature for better JSON output with Vicuna
                if 'vicuna' in model_to_use.lower():
                    temperature = 0.3  # Lower temperature for more focused output
                
                logger.debug(f"Generating checklist with model: {model_to_use}, max_tokens: {max_tokens}, temperature: {temperature}")
                
                result = self.backend.generate(
                    prompt=json_prompt,
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
        raise Exception(f"Failed to generate checklist after {self.max_retries} attempts. Last error: {last_error}")
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON array from response text"""
        try:
            import re
            
            logger.debug(f"Extracting JSON from response (length: {len(response)})")
            
            # Check if the response contains the specific problematic text
            if "Please provide a list of questions in the JSON format as described" in response:
                logger.warning("Model provided instructions instead of JSON, triggering retry")
                raise Exception("Model provided instructions instead of JSON")
            
            # If response starts and ends with brackets, treat it as JSON
            if response.strip().startswith('[') and response.strip().endswith(']'):
                json_str = response.strip()
                logger.debug("Found complete JSON array")
                
                # Basic validation - just check if it looks like a JSON array
                return json_str
            
            # Try to find JSON array pattern [...] as fallback
            json_pattern = r'\[.*\]'  # Use greedy matching
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Return the longest match (most likely to be complete)
                json_str = max(matches, key=len)
                logger.debug("Extracted JSON from pattern match")
                
                # Basic validation - just check if it looks like a JSON array
                return json_str
            
            
            # Check for various instruction patterns that indicate model confusion
            instruction_patterns = [
                "provide a list", "json format", "please", "here is", "i can help",
                "create a checklist", "generate a", "based on", "following format"
            ]
            
            if any(phrase in response.lower() for phrase in instruction_patterns):
                logger.warning("Model provided instructions instead of JSON, triggering retry")
                raise Exception("Model provided instructions instead of JSON")
            
            # If we get here, the response doesn't contain recognizable JSON
            logger.warning(f"No valid JSON found in response: {response[:200]}...")
            raise Exception("No valid JSON array found in response")
            
        except Exception as e:
            logger.warning(f"Failed to extract JSON from response: {e}")
            raise Exception(f"Failed to extract valid JSON: {e}")
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""
        try:
            import re
            
            # Fix invalid escape sequences (best\_ans -> best_ans)
            json_str = re.sub(r'best\\_ans', 'best_ans', json_str)
            
            # Fix other common escape issues
            json_str = re.sub(r'\\_', '_', json_str)
            
            # Remove any trailing text after the JSON array
            # Find the last ] and cut everything after it
            last_bracket = json_str.rfind(']')
            if last_bracket != -1:
                json_str = json_str[:last_bracket + 1]
            
            logger.debug("Applied JSON formatting fixes")
            return json_str
            
        except Exception as e:
            logger.warning(f"Failed to fix malformed JSON: {e}")
            return json_str
    
    def _extract_json_object_from_response(self, response: str) -> str:
        """Extract JSON object from response text (for analysis step)"""
        try:
            import re
            
            logger.debug(f"Extracting JSON object from response (length: {len(response)})")
            
            # Remove any explanatory text before JSON
            response = response.strip()
            
            # Handle case where response is truncated JSON missing opening brace
            # Pattern: starts with ["something"] or "key": value indicating middle of object
            if (response.startswith('[') or 
                ('"' in response and ':' in response and not response.startswith('{'))):
                logger.debug("Detected truncated JSON missing opening brace")
                # Try adding opening brace
                candidate = '{' + response
                
                # Check if it ends with a closing brace, if not try to complete it
                if not candidate.rstrip().endswith('}'):
                    # Try to add closing brace if it looks incomplete
                    candidate = candidate.rstrip() + '}'
                
                try:
                    json.loads(candidate)
                    logger.debug("Successfully reconstructed truncated JSON")
                    return candidate
                except json.JSONDecodeError:
                    # If that doesn't work, continue with other methods
                    pass
            
            # Try to find JSON object pattern {...}
            # Look for the first { and last } that create a valid JSON structure
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                # Try each match to see if it's valid JSON
                for match in matches:
                    try:
                        # Test if this is valid JSON
                        json.loads(match)
                        logger.debug("Found valid JSON object")
                        return match
                    except json.JSONDecodeError:
                        continue
                
                # If no valid JSON found, try the longest match with cleaning
                longest_match = max(matches, key=len)
                logger.debug("Trying to clean longest JSON match")
                
                # Basic cleaning - remove extra text after the closing brace
                last_brace = longest_match.rfind('}')
                if last_brace != -1:
                    cleaned = longest_match[:last_brace + 1]
                    try:
                        json.loads(cleaned)
                        return cleaned
                    except json.JSONDecodeError:
                        pass
            
            # Try to extract between first { and last }
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                potential_json = response[first_brace:last_brace + 1]
                try:
                    json.loads(potential_json)
                    logger.debug("Extracted JSON between first and last braces")
                    return potential_json
                except json.JSONDecodeError:
                    pass
            
            # If we get here, no valid JSON object found
            logger.warning(f"No valid JSON object found in response: {response[:200]}...")
            raise Exception("No valid JSON object found in response")
            
        except Exception as e:
            logger.warning(f"Failed to extract JSON object from response: {e}")
            raise Exception(f"Failed to extract valid JSON object: {e}")
    
    def _create_fallback_checklist(self, topic: str) -> list:
        """Create a simple fallback checklist in binary format"""
        return [
            {
                "question": "Does the email have a clear and relevant subject line?",
                "best_ans": "yes", 
                "priority": "very high"
            },
            {
                "question": "Is the content directly relevant to the topic?",
                "best_ans": "yes",
                "priority": "very high" 
            },
            {
                "question": "Does the email include a clear call-to-action?",
                "best_ans": "yes",
                "priority": "high"
            },
            {
                "question": "Is the tone appropriate for the intended audience?", 
                "best_ans": "yes",
                "priority": "high"
            },
            {
                "question": "Does the email have a proper greeting?",
                "best_ans": "yes",
                "priority": "medium"
            },
            {
                "question": "Does the email have a professional closing?",
                "best_ans": "yes", 
                "priority": "medium"
            },
            {
                "question": "Is the email length appropriate for the content?",
                "best_ans": "yes",
                "priority": "low"
            },
            {
                "question": "Does the email avoid spelling and grammar errors?",
                "best_ans": "yes",
                "priority": "low"
            }
        ]
    
    def _get_mode_specific_template(self) -> str:
        """Get template based on checklist mode"""
        if self.checklist_mode == "extract_only":
            return self.template_manager.get_template("checklist_extract")
        elif self.checklist_mode == "preprocess":
            return self.template_manager.get_template("checklist_preprocess")
        else:  # enhanced mode (default)
            return self.template_manager.get_checklist_template()

