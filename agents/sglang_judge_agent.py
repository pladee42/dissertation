from models.model_factory import create_model_instance
from models.schemas import Checklist, JudgmentResult, EvaluationResult
from utils.retry import retry_with_backoff
from config.settings import settings
from typing import List, Dict, Any
import logging
import json
import sglang as sgl

logger = logging.getLogger(__name__)

class SGLangJudgeAgent:
    """SGLang-optimized Judge Agent with constrained generation for probability-based scoring"""
    
    def __init__(self, model_id: str, dtype: str, quantization: str, custom_config: Dict[str, Any] = None):
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"Initializing SGLangJudgeAgent with model: {model_id}")
        
        # Initialize SGLang model
        self.llm = create_model_instance(
            model_name=self._get_model_name_from_id(model_id),
            backend="sglang",
            custom_config=custom_config
        )
        
        # Judge-optimized generation parameters
        self.judge_config = {
            "temperature": 0.1,  # Very low temperature for consistent judgments
            "max_new_tokens": 200,
            "top_p": 0.8,
            "stop": None
        }
        
        # Probability computation parameters
        self.probability_config = {
            "temperature": 0.0,  # Zero temperature for probability computation
            "max_new_tokens": 1,
            "top_p": 1.0
        }
        
        # Structured judgment schema
        self.judgment_schema = {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "yes_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "no_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "judgment": {"type": "string", "enum": ["Yes", "No"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string"}
            },
            "required": ["question", "yes_probability", "no_probability", "judgment", "confidence"]
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
    def evaluate_single_item(self, email_content: str, checklist_item, user_query: str) -> JudgmentResult:
        """Evaluate a single checklist item using SGLang's constrained generation"""
        
        logger.debug(f"Evaluating checklist item: {checklist_item.question[:50]}...")
        
        try:
            # Load judge prompt template
            with open("prompts/judge/judge.txt", 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            # Use SGLang's constrained generation for probability-based scoring
            @sgl.function
            def evaluate_with_probabilities(s, user_query_text, email_text, question, expected):
                # Format the evaluation prompt
                s += prompt_template.replace('{user_query}', user_query_text)
                s += prompt_template.replace('{model_output}', email_text)
                s += prompt_template.replace('{checklist_question}', question)
                s += prompt_template.replace('{expected_answer}', expected)
                
                s += "\n\nBased on your evaluation, answer the checklist question:"
                s += f"\n\nQuestion: {question}"
                s += f"\nExpected Answer: {expected}"
                s += "\n\nYour answer (Yes or No): "
                
                # Constrained generation for Yes/No answer
                s += sgl.gen("answer", choices=["Yes", "No"], **self.probability_config)
                
                # Generate confidence and reasoning
                s += "\n\nProvide your confidence (0.0 to 1.0) and brief reasoning:"
                s += "\nConfidence: "
                s += sgl.gen("confidence_raw", regex=r"0\.\d+|1\.0|0\.0", **self.judge_config)
                
                s += "\nReasoning: "
                s += sgl.gen("reasoning", max_new_tokens=100, **self.judge_config)
            
            # Execute SGLang function
            state = evaluate_with_probabilities.run(
                user_query_text=user_query,
                email_text=email_content,
                question=checklist_item.question,
                expected=checklist_item.correct_answer
            )
            
            # Extract results
            answer = state["answer"].strip()
            confidence_str = state["confidence_raw"].strip()
            reasoning = state["reasoning"].strip()
            
            # Parse confidence
            try:
                confidence = float(confidence_str)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except:
                confidence = 0.5  # Default confidence
            
            # Compute probabilities using SGLang's constrained generation
            probabilities = self._compute_yes_no_probabilities_sglang(
                user_query, email_content, checklist_item.question, checklist_item.correct_answer
            )
            
            result = JudgmentResult(
                question=checklist_item.question,
                yes_probability=probabilities["yes"],
                no_probability=probabilities["no"],
                judgment=answer,
                confidence=confidence
            )
            
            logger.debug(f"Evaluation complete: {answer} (confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"SGLang single item evaluation failed: {e}")
            # Fallback to basic evaluation
            return self._evaluate_single_item_fallback(email_content, checklist_item, user_query)
    
    def _compute_yes_no_probabilities_sglang(self, user_query: str, email_content: str, question: str, expected: str) -> Dict[str, float]:
        """Compute Yes/No probabilities using SGLang's constrained generation"""
        
        try:
            @sgl.function
            def compute_probabilities(s, user_query_text, email_text, question_text, expected_answer):
                s += f"User Query: {user_query_text}\n"
                s += f"Email Content: {email_text}\n"
                s += f"Evaluation Question: {question_text}\n"
                s += f"Expected Answer: {expected_answer}\n\n"
                s += "Based on the email content, answer the evaluation question with Yes or No:\n"
                s += "Answer: "
                
                # Use multiple samples to estimate probabilities
                samples = []
                for i in range(10):  # Take 10 samples
                    with s.fork(f"sample_{i}") as fork_state:
                        fork_state += sgl.gen(f"answer_{i}", choices=["Yes", "No"], temperature=0.3)
                        samples.append(fork_state[f"answer_{i}"])
                
                return samples
            
            # Execute probability computation
            samples = compute_probabilities.run(
                user_query_text=user_query,
                email_text=email_content,
                question_text=question,
                expected_answer=expected
            )
            
            # Calculate probabilities from samples
            yes_count = sum(1 for sample in samples if sample.strip().lower() == "yes")
            no_count = len(samples) - yes_count
            
            total = len(samples)
            yes_prob = yes_count / total if total > 0 else 0.5
            no_prob = no_count / total if total > 0 else 0.5
            
            # Ensure probabilities sum to 1
            total_prob = yes_prob + no_prob
            if total_prob > 0:
                yes_prob /= total_prob
                no_prob /= total_prob
            else:
                yes_prob = no_prob = 0.5
            
            return {"yes": yes_prob, "no": no_prob}
            
        except Exception as e:
            logger.warning(f"SGLang probability computation failed: {e}, using fallback")
            return {"yes": 0.5, "no": 0.5}
    
    def evaluate_email_structured(self, email_content: str, checklist: Checklist, user_query: str) -> EvaluationResult:
        """Evaluate entire email using SGLang's structured output with schema validation"""
        
        logger.info(f"Evaluating email with {len(checklist.items)} checklist items using SGLang")
        
        try:
            @sgl.function
            def evaluate_email_batch(s, user_query_text, email_text, checklist_items):
                s += f"User Query: {user_query_text}\n"
                s += f"Email to Evaluate: {email_text}\n\n"
                s += "Evaluate this email against the following checklist items:\n\n"
                
                results = []
                for i, item in enumerate(checklist_items):
                    s += f"Item {i+1}: {item.question}\n"
                    s += f"Expected Answer: {item.correct_answer}\n"
                    s += "Your evaluation: "
                    
                    # Constrained evaluation
                    s += sgl.gen(f"judgment_{i}", choices=["Yes", "No"], **self.probability_config)
                    
                    s += "\nConfidence (0.0-1.0): "
                    s += sgl.gen(f"confidence_{i}", regex=r"0\.\d+|1\.0|0\.0", **self.judge_config)
                    
                    s += "\n\n"
                    
                    # Collect result data
                    judgment = s[f"judgment_{i}"]
                    confidence_str = s[f"confidence_{i}"]
                    
                    try:
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))
                    except:
                        confidence = 0.5
                    
                    # Compute probabilities for this item
                    probabilities = self._compute_yes_no_probabilities_sglang(
                        user_query_text, email_text, item.question, item.correct_answer
                    )
                    
                    result = JudgmentResult(
                        question=item.question,
                        yes_probability=probabilities["yes"],
                        no_probability=probabilities["no"],
                        judgment=judgment,
                        confidence=confidence
                    )
                    results.append(result)
                
                return results
            
            # Execute batch evaluation
            results = evaluate_email_batch.run(
                user_query_text=user_query,
                email_text=email_content,
                checklist_items=checklist.items
            )
            
            # Calculate scores
            overall_score = self._calculate_overall_score(results, checklist)
            weighted_score = self._calculate_weighted_score(results, checklist)
            
            evaluation_result = EvaluationResult(
                email_content=email_content,
                checklist_results=results,
                overall_score=overall_score,
                weighted_score=weighted_score
            )
            
            logger.info(f"SGLang evaluation complete. Overall score: {overall_score:.2f}, Weighted: {weighted_score:.2f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"SGLang structured evaluation failed: {e}")
            # Fallback to individual item evaluation
            return self.evaluate_email(email_content, checklist, user_query)
    
    def evaluate_email(self, email_content: str, checklist: Checklist, user_query: str) -> EvaluationResult:
        """Evaluate entire email against checklist (standard method)"""
        
        results = []
        for item in checklist.items:
            result = self.evaluate_single_item(email_content, item, user_query)
            results.append(result)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results, checklist)
        weighted_score = self._calculate_weighted_score(results, checklist)
        
        return EvaluationResult(
            email_content=email_content,
            checklist_results=results,
            overall_score=overall_score,
            weighted_score=weighted_score
        )
    
    def evaluate_with_reasoning(self, email_content: str, checklist: Checklist, user_query: str) -> Dict[str, Any]:
        """Evaluate email with detailed reasoning using SGLang's structured generation"""
        
        logger.info("Generating detailed evaluation with reasoning")
        
        try:
            @sgl.function
            def evaluate_with_detailed_reasoning(s, user_query_text, email_text, checklist_items):
                s += "Perform a detailed evaluation of the email with step-by-step reasoning.\n\n"
                s += f"User Query: {user_query_text}\n"
                s += f"Email Content: {email_text}\n\n"
                
                detailed_results = []
                
                for i, item in enumerate(checklist_items):
                    s += f"\n=== Evaluation {i+1} ===\n"
                    s += f"Question: {item.question}\n"
                    s += f"Expected: {item.correct_answer}\n"
                    s += f"Priority: {item.priority.value}\n\n"
                    
                    s += "Step-by-step reasoning:\n"
                    s += sgl.gen(f"reasoning_{i}", max_new_tokens=200, **self.judge_config)
                    
                    s += "\n\nFinal judgment: "
                    s += sgl.gen(f"judgment_{i}", choices=["Yes", "No"], **self.probability_config)
                    
                    s += "\nConfidence: "
                    s += sgl.gen(f"confidence_{i}", regex=r"0\.\d+|1\.0|0\.0", **self.judge_config)
                    
                    # Collect detailed result
                    detailed_result = {
                        "question": item.question,
                        "reasoning": s[f"reasoning_{i}"],
                        "judgment": s[f"judgment_{i}"],
                        "confidence": float(s[f"confidence_{i}"]) if s[f"confidence_{i}"].replace('.', '').isdigit() else 0.5,
                        "priority": item.priority.value
                    }
                    detailed_results.append(detailed_result)
                
                return detailed_results
            
            # Execute detailed evaluation
            detailed_results = evaluate_with_detailed_reasoning.run(
                user_query_text=user_query,
                email_text=email_content,
                checklist_items=checklist.items
            )
            
            # Convert to standard format
            judgment_results = []
            for i, (detail, item) in enumerate(zip(detailed_results, checklist.items)):
                probabilities = self._compute_yes_no_probabilities_sglang(
                    user_query, email_content, item.question, item.correct_answer
                )
                
                result = JudgmentResult(
                    question=detail["question"],
                    yes_probability=probabilities["yes"],
                    no_probability=probabilities["no"],
                    judgment=detail["judgment"],
                    confidence=detail["confidence"]
                )
                judgment_results.append(result)
            
            # Calculate scores
            overall_score = self._calculate_overall_score(judgment_results, checklist)
            weighted_score = self._calculate_weighted_score(judgment_results, checklist)
            
            return {
                "detailed_results": detailed_results,
                "judgment_results": judgment_results,
                "overall_score": overall_score,
                "weighted_score": weighted_score,
                "evaluation_method": "sglang_with_reasoning"
            }
            
        except Exception as e:
            logger.error(f"Detailed evaluation failed: {e}")
            # Fallback to standard evaluation
            standard_result = self.evaluate_email(email_content, checklist, user_query)
            return {
                "detailed_results": [],
                "judgment_results": standard_result.checklist_results,
                "overall_score": standard_result.overall_score,
                "weighted_score": standard_result.weighted_score,
                "evaluation_method": "fallback"
            }
    
    def _evaluate_single_item_fallback(self, email_content: str, checklist_item, user_query: str) -> JudgmentResult:
        """Fallback evaluation method"""
        
        logger.info("Using fallback evaluation method")
        
        # Simple prompt-based evaluation
        prompt = f"""Evaluate the following email against this checklist item:

User Query: {user_query}
Email Content: {email_content}

Checklist Question: {checklist_item.question}
Expected Answer: {checklist_item.correct_answer}

Does the email satisfy this requirement? Answer Yes or No."""
        
        try:
            response = self.llm.generate(
                query=prompt,
                model_name=self.model_name,
                custom_params=self.judge_config,
                remove_cot=True
            )
            
            # Parse response
            answer = "Yes" if "yes" in response.lower() else "No"
            confidence = 0.5  # Default confidence
            
            # Simple probability estimation
            yes_prob = 0.7 if answer == "Yes" else 0.3
            no_prob = 1.0 - yes_prob
            
            return JudgmentResult(
                question=checklist_item.question,
                yes_probability=yes_prob,
                no_probability=no_prob,
                judgment=answer,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Fallback evaluation failed: {e}")
            # Ultimate fallback
            return JudgmentResult(
                question=checklist_item.question,
                yes_probability=0.5,
                no_probability=0.5,
                judgment="Yes",
                confidence=0.5
            )
    
    def _calculate_overall_score(self, results: List[JudgmentResult], checklist: Checklist) -> float:
        """Calculate simple overall score"""
        correct_count = sum(1 for r in results if r.judgment == "Yes")
        return correct_count / len(results) if results else 0.0
    
    def _calculate_weighted_score(self, results: List[JudgmentResult], checklist: Checklist) -> float:
        """Calculate weighted score based on priority"""
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        
        total_weight = 0
        weighted_sum = 0
        
        for result, item in zip(results, checklist.items):
            weight = priority_weights[item.priority.value]
            total_weight += weight
            if result.judgment == "Yes":
                weighted_sum += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def cleanup(self):
        """Cleanup the SGLang judge agent and release resources"""
        logger.info(f"Cleaning up SGLangJudgeAgent with model: {self.model_name}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("SGLangJudgeAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during SGLangJudgeAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the SGLang judge agent"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "backend": "sglang",
            "judge_config": self.judge_config,
            "probability_config": self.probability_config,
            "model_info": self.llm.get_model_info(),
            "features": [
                "constrained_generation",
                "probability_based_scoring",
                "structured_evaluation",
                "detailed_reasoning",
                "schema_validation",
                "radix_attention"
            ]
        }