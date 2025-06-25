from models.model_factory import create_model_instance
from models.schemas import Checklist, ChecklistItem, Priority
from utils.retry import retry_with_backoff
from config.settings import settings
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import sglang as sgl

logger = logging.getLogger(__name__)

class SGLangChecklistAgent:
    """SGLang-optimized Checklist Generation Agent with structured output and xgrammar validation"""
    
    def __init__(self, model_id: str, dtype: str, quantization: str, custom_config: Dict[str, Any] = None):
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"Initializing SGLangChecklistAgent with model: {model_id}")
        
        # Initialize SGLang model
        self.llm = create_model_instance(
            model_name=self._get_model_name_from_id(model_id),
            backend="sglang",
            custom_config=custom_config
        )
        
        # Checklist-optimized generation parameters
        self.checklist_config = {
            "temperature": 0.3,  # Lower temperature for more consistent structure
            "max_new_tokens": 3000,
            "top_p": 0.9,
            "stop": None
        }
        
        # JSON schema for structured output
        self.checklist_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "correct_answer": {"type": "string", "enum": ["Yes", "No"]},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "explanation": {"type": "string"}
                        },
                        "required": ["question", "correct_answer", "priority"]
                    },
                    "minItems": 1,
                    "maxItems": 20
                },
                "topic": {"type": "string"}
            },
            "required": ["items", "topic"]
        }
    
    def _get_model_name_from_id(self, model_id: str) -> str:
        """Extract model name from model_id for config lookup"""
        id_to_name = {
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 'deepseek-r1-1.5b',
            'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 'deepseek-r1-8b',
            'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 'deepseek-r1-70b',
            'unsloth/Llama-3.2-3B-Instruct': 'llama-3-3b',
            'casperhansen/llama-3-8b-instruct-awq': 'llama-3-8b',
            'gaunernst/gemma-3-4b-it-qat-autoawq': 'gemma-3-4b',
            'Qwen/Qwen3-8B-AWQ': 'qwen-3-8b',
            'kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ': 'llama-4-109b'
        }
        return id_to_name.get(model_id, model_id.split('/')[-1])
    
    @retry_with_backoff(max_retries=settings.max_retries)
    def generate_checklist(self, user_query: str, reference_response: str, topic: str) -> Checklist:
        """Generate structured checklist using SGLang's structured output capabilities"""
        
        logger.info(f"Generating checklist for topic: {topic} using SGLang")
        
        try:
            # Load prompt template
            prompt_path = Path("prompts/checklist/checklist.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            # Use SGLang's structured generation with JSON schema
            @sgl.function
            def generate_structured_checklist(s, user_query_text, reference_response_text, topic_text):
                # Format the prompt
                s += prompt_template.replace('{user_query}', user_query_text)
                s += prompt_template.replace('{reference_response}', reference_response_text)
                s += "\n\nGenerate a comprehensive evaluation checklist in JSON format."
                s += "\n\nThe checklist should include:"
                s += "\n1. Clear, specific evaluation questions"
                s += "\n2. Expected answers (Yes/No)"
                s += "\n3. Priority levels (high/medium/low)"
                s += "\n4. Brief explanations for each item"
                
                # Use SGLang's structured generation with JSON schema constraint
                s += sgl.gen(
                    "checklist_json",
                    regex=self._create_json_regex_pattern(),
                    **self.checklist_config
                )
            
            # Execute SGLang function
            state = generate_structured_checklist.run(
                user_query_text=user_query,
                reference_response_text=reference_response,
                topic_text=topic
            )
            
            # Extract and parse JSON
            checklist_json = state["checklist_json"]
            checklist_data = self._parse_and_validate_checklist(checklist_json, topic)
            
            logger.info(f"SGLang checklist generated successfully with {len(checklist_data.items)} items")
            
            return checklist_data
            
        except Exception as e:
            logger.error(f"SGLang checklist generation failed: {e}")
            # Fallback to template-based generation
            return self._generate_checklist_fallback(user_query, reference_response, topic)
    
    def generate_checklist_with_xgrammar(self, user_query: str, reference_response: str, topic: str) -> Checklist:
        """Generate checklist using SGLang's xgrammar backend for strict JSON validation"""
        
        logger.info(f"Generating checklist with xgrammar validation for topic: {topic}")
        
        try:
            @sgl.function
            def generate_xgrammar_checklist(s, user_query_text, reference_response_text, topic_text):
                # Load and format prompt
                prompt_path = Path("prompts/checklist/checklist.txt")
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt_template = f.read()
                
                s += prompt_template.replace('{user_query}', user_query_text)
                s += prompt_template.replace('{reference_response}', reference_response_text)
                s += "\n\nIMPORTANT: Generate ONLY valid JSON matching the exact schema below:"
                s += f"\n{json.dumps(self.checklist_schema, indent=2)}"
                s += "\n\nJSON Response:"
                
                # Use xgrammar for strict JSON schema validation
                s += sgl.gen(
                    "validated_checklist",
                    grammar=sgl.json_schema(self.checklist_schema),
                    **self.checklist_config
                )
            
            # Execute with xgrammar validation
            state = generate_xgrammar_checklist.run(
                user_query_text=user_query,
                reference_response_text=reference_response,
                topic_text=topic
            )
            
            # Parse validated JSON
            validated_json = state["validated_checklist"]
            checklist_dict = json.loads(validated_json) if isinstance(validated_json, str) else validated_json
            
            # Create Pydantic model
            checklist_data = self._create_checklist_from_dict(checklist_dict, topic)
            
            logger.info(f"xgrammar checklist generated successfully with {len(checklist_data.items)} items")
            
            return checklist_data
            
        except Exception as e:
            logger.error(f"xgrammar checklist generation failed: {e}")
            # Fallback to regular structured generation
            return self.generate_checklist(user_query, reference_response, topic)
    
    def generate_template_based_checklist(self, user_query: str, reference_response: str, topic: str) -> Checklist:
        """Generate checklist using template-based structured generation for consistency"""
        
        logger.info(f"Generating template-based checklist for topic: {topic}")
        
        try:
            @sgl.function
            def generate_template_checklist(s, user_query_text, reference_response_text, topic_text):
                # Use template-based approach for consistent structure
                s += "Generate an evaluation checklist for the following email scenario:\n"
                s += f"User Query: {user_query_text}\n"
                s += f"Reference Response: {reference_response_text}\n"
                s += f"Topic: {topic_text}\n\n"
                
                s += "Create exactly 5-10 evaluation questions following this template:\n\n"
                
                # Generate each checklist item using structured approach
                s += "Number of items: "
                s += sgl.gen("num_items", choices=["5", "6", "7", "8", "9", "10"], temperature=0.0)
                
                num_items = int(s["num_items"])
                items = []
                
                for i in range(num_items):
                    s += f"\n\nItem {i+1}:\n"
                    s += "Question: "
                    s += sgl.gen(f"question_{i}", max_new_tokens=100, stop=["\n"])
                    
                    s += "\nCorrect Answer: "
                    s += sgl.gen(f"answer_{i}", choices=["Yes", "No"], temperature=0.0)
                    
                    s += "\nPriority: "
                    s += sgl.gen(f"priority_{i}", choices=["high", "medium", "low"], temperature=0.3)
                    
                    s += "\nExplanation: "
                    s += sgl.gen(f"explanation_{i}", max_new_tokens=150, stop=["\n\n"])
                    
                    # Collect item data
                    items.append({
                        "question": s[f"question_{i}"].strip(),
                        "answer": s[f"answer_{i}"],
                        "priority": s[f"priority_{i}"],
                        "explanation": s[f"explanation_{i}"].strip()
                    })
                
                return items
            
            # Execute template generation
            items_data = generate_template_checklist.run(
                user_query_text=user_query,
                reference_response_text=reference_response,
                topic_text=topic
            )
            
            # Create checklist items
            checklist_items = []
            for item_data in items_data:
                checklist_item = ChecklistItem(
                    question=item_data["question"],
                    correct_answer=item_data["answer"],
                    priority=Priority(item_data["priority"]),
                    explanation=item_data["explanation"]
                )
                checklist_items.append(checklist_item)
            
            # Create checklist
            checklist = Checklist(items=checklist_items, topic=topic)
            
            logger.info(f"Template-based checklist generated with {len(checklist_items)} items")
            
            return checklist
            
        except Exception as e:
            logger.error(f"Template-based checklist generation failed: {e}")
            raise
    
    def _create_json_regex_pattern(self) -> str:
        """Create regex pattern for JSON structure validation"""
        # Simplified regex for basic JSON structure
        return r'\{.*"items".*\[.*\{.*"question".*"correct_answer".*"priority".*\}.*\].*"topic".*\}'
    
    def _parse_and_validate_checklist(self, checklist_json: str, topic: str) -> Checklist:
        """Parse and validate checklist JSON"""
        
        try:
            # Clean up the JSON string
            checklist_json = checklist_json.strip()
            if not checklist_json.startswith('{'):
                # Find the first { and last }
                start_idx = checklist_json.find('{')
                end_idx = checklist_json.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    checklist_json = checklist_json[start_idx:end_idx+1]
            
            # Parse JSON
            checklist_dict = json.loads(checklist_json)
            
            return self._create_checklist_from_dict(checklist_dict, topic)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}, attempting repair")
            return self._repair_and_parse_checklist(checklist_json, topic)
    
    def _create_checklist_from_dict(self, checklist_dict: Dict[str, Any], topic: str) -> Checklist:
        """Create Checklist object from dictionary"""
        
        # Ensure topic is set
        checklist_dict["topic"] = topic
        
        # Validate and create items
        items = []
        for item_data in checklist_dict.get("items", []):
            try:
                # Normalize priority
                priority_value = item_data.get("priority", "medium").lower()
                if priority_value not in ["high", "medium", "low"]:
                    priority_value = "medium"
                
                # Normalize answer
                answer = item_data.get("correct_answer", "Yes")
                if answer not in ["Yes", "No"]:
                    answer = "Yes"
                
                item = ChecklistItem(
                    question=item_data.get("question", ""),
                    correct_answer=answer,
                    priority=Priority(priority_value),
                    explanation=item_data.get("explanation", "")
                )
                items.append(item)
                
            except Exception as e:
                logger.warning(f"Skipping invalid checklist item: {e}")
                continue
        
        # Ensure we have at least one item
        if not items:
            items = [ChecklistItem(
                question="Does the email address the main topic appropriately?",
                correct_answer="Yes",
                priority=Priority.HIGH,
                explanation="The email should be relevant to the specified topic."
            )]
        
        return Checklist(items=items, topic=topic)
    
    def _repair_and_parse_checklist(self, checklist_json: str, topic: str) -> Checklist:
        """Attempt to repair malformed JSON and parse checklist"""
        
        logger.info("Attempting to repair malformed checklist JSON")
        
        # Basic JSON repair attempts
        repaired_json = checklist_json
        
        # Fix common issues
        repaired_json = repaired_json.replace("'", '"')  # Single to double quotes
        repaired_json = repaired_json.replace('True', 'true')
        repaired_json = repaired_json.replace('False', 'false')
        repaired_json = repaired_json.replace('None', 'null')
        
        try:
            checklist_dict = json.loads(repaired_json)
            return self._create_checklist_from_dict(checklist_dict, topic)
        except:
            # If repair fails, create a default checklist
            logger.warning("JSON repair failed, creating default checklist")
            return self._create_default_checklist(topic)
    
    def _create_default_checklist(self, topic: str) -> Checklist:
        """Create a default checklist when generation fails"""
        
        default_items = [
            ChecklistItem(
                question="Does the email address the main topic appropriately?",
                correct_answer="Yes",
                priority=Priority.HIGH,
                explanation="The email should be relevant to the specified topic."
            ),
            ChecklistItem(
                question="Is the tone appropriate for the context?",
                correct_answer="Yes",
                priority=Priority.MEDIUM,
                explanation="The tone should match the intended audience and purpose."
            ),
            ChecklistItem(
                question="Is the email well-structured and clear?",
                correct_answer="Yes",
                priority=Priority.HIGH,
                explanation="The email should have clear structure and be easy to understand."
            )
        ]
        
        return Checklist(items=default_items, topic=topic)
    
    def _generate_checklist_fallback(self, user_query: str, reference_response: str, topic: str) -> Checklist:
        """Fallback method for checklist generation"""
        
        logger.info("Using fallback checklist generation method")
        
        # Simple prompt-based generation
        prompt = f"""Generate an evaluation checklist for an email about: {topic}

User Query: {user_query}
Reference Response: {reference_response}

Create 5-7 evaluation questions that check if an email response is appropriate.
Each question should be answerable with Yes or No.
"""
        
        try:
            response = self.llm.generate(
                query=prompt,
                model_name=self.model_name,
                custom_params=self.checklist_config,
                remove_cot=True
            )
            
            # Parse response into checklist items
            lines = response.split('\n')
            items = []
            
            for line in lines:
                line = line.strip()
                if line and ('?' in line):
                    # Extract question
                    question = line.split('?')[0] + '?'
                    if len(question) > 10:  # Valid question
                        item = ChecklistItem(
                            question=question,
                            correct_answer="Yes",
                            priority=Priority.MEDIUM,
                            explanation="Generated from fallback method"
                        )
                        items.append(item)
            
            # Ensure we have items
            if not items:
                return self._create_default_checklist(topic)
            
            return Checklist(items=items[:7], topic=topic)  # Limit to 7 items
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return self._create_default_checklist(topic)
    
    def save_checklist(self, checklist: Checklist, filename: str):
        """Save checklist to both JSON and text formats"""
        output_dir = Path(settings.output_dir) / "checklist"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{filename}_sglang.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(checklist.model_dump(), f, indent=2)
        
        # Save readable text
        txt_path = output_dir / f"{filename}_sglang.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"SGLang Checklist for: {checklist.topic}\n")
            f.write(f"Generated with: {self.model_name} (SGLang + RadixAttention)\n\n")
            for i, item in enumerate(checklist.items, 1):
                f.write(f"{i}. {item.question}\n")
                f.write(f"   Expected: {item.correct_answer}\n")
                f.write(f"   Priority: {item.priority.value}\n")
                if item.explanation:
                    f.write(f"   Explanation: {item.explanation}\n")
                f.write("\n")
        
        logger.info(f"SGLang checklist saved: {json_path} and {txt_path}")
    
    def cleanup(self):
        """Cleanup the SGLang checklist agent and release resources"""
        logger.info(f"Cleaning up SGLangChecklistAgent with model: {self.model_name}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("SGLangChecklistAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during SGLangChecklistAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the SGLang checklist agent"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "backend": "sglang",
            "checklist_config": self.checklist_config,
            "model_info": self.llm.get_model_info(),
            "features": [
                "structured_json_output",
                "xgrammar_validation",
                "template_based_generation",
                "json_schema_constraint",
                "radix_attention"
            ]
        }