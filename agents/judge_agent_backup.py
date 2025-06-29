from models.llm import ModelInference
from models.schemas import Checklist, JudgmentResult, EvaluationResult
from utils.retry import retry_with_backoff
from config.settings import settings
from typing import List

class JudgeAgent:
    def __init__(self, model_id: str, dtype: str, quantization: str):
        self.llm = ModelInference(model_id=model_id, dtype=dtype, quantization=quantization)
        self.model_id = model_id
    
    @retry_with_backoff(max_retries=settings.max_retries)
    def evaluate_single_item(self, email_content: str, checklist_item, user_query: str) -> JudgmentResult:
        """Evaluate a single checklist item"""
        
        # Load judge prompt template
        with open("config/prompts/judge/judge.txt", 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format prompt
        prompt = prompt_template.replace('{user_query}', user_query)
        prompt = prompt.replace('{model_output}', email_content)
        prompt = prompt.replace('{checklist_question}', checklist_item.question)
        prompt = prompt.replace('{expected_answer}', checklist_item.correct_answer)
        
        # Get probabilities
        probabilities = self.llm.compute_yes_no_probability(
            query=prompt, 
            model_name=self.model_id.split('/')[-1]
        )
        
        # Calculate confidence
        confidence = abs(probabilities["yes"] - probabilities["no"])
        
        return JudgmentResult(
            question=checklist_item.question,
            yes_probability=probabilities["yes"],
            no_probability=probabilities["no"],
            judgment="Yes" if probabilities["yes"] > probabilities["no"] else "No",
            confidence=confidence
        )
    
    def evaluate_email(self, email_content: str, checklist: Checklist, user_query: str) -> EvaluationResult:
        """Evaluate entire email against checklist"""
        
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
    
    def _calculate_overall_score(self, results: List[JudgmentResult], checklist: Checklist) -> float:
        """Calculate simple overall score"""
        correct_count = sum(1 for r in results if r.judgment == "Yes")
        return correct_count / len(results)
    
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
        """Cleanup the judge agent and release resources"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaning up JudgeAgent with model: {self.model_id.split('/')[-1]}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("JudgeAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during JudgeAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
