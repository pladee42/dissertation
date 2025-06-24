from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import logging
from pathlib import Path

from agents.email_agent import EmailAgent, EmailGenerationResult
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent, EvaluationResult
from models.schemas import Checklist
from config.models import MODELS_CONFIG

logger = logging.getLogger(__name__)

@dataclass
class EmailCandidate:
    """Represents a generated email candidate with evaluation results"""
    email_content: str
    model_name: str
    model_id: str
    generation_result: EmailGenerationResult
    checklist_evaluation: Optional[EvaluationResult] = None
    overall_score: Optional[float] = None
    weighted_score: Optional[float] = None
    rank: Optional[int] = None

@dataclass
class MultiModelResults:
    """Container for all multi-model generation and evaluation results"""
    candidates: List[EmailCandidate]
    best_candidate: EmailCandidate
    checklist_used: Checklist
    prompt_used: str
    topic: str
    generation_time: float
    evaluation_time: float

class MultiModelEmailGenerator:
    """Orchestrates multiple models for email generation and ranking"""
    
    def __init__(self, 
                 email_models: List[str],
                 checklist_model: str,
                 judge_model: str,
                 max_concurrent: int = 3):
        """
        Initialize with multiple email models and evaluation models
        
        Args:
            email_models: List of model identifiers for email generation
            checklist_model: Model for checklist generation
            judge_model: Model for evaluation/judging
            max_concurrent: Maximum concurrent model executions
        """
        self.email_models = email_models
        self.checklist_model = checklist_model
        self.judge_model = judge_model
        self.max_concurrent = max_concurrent
        
        # Initialize agents
        logger.info("Initializing multi-model email generation system...")
        self.email_agents = {}
        self.checklist_agent = None
        self.judge_agent = None
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents with error handling"""
        # Initialize email agents
        for model_key in self.email_models:
            try:
                model_id = MODELS_CONFIG[model_key]['model_id']
                self.email_agents[model_key] = EmailAgent(model_id)
                logger.info(f"Initialized EmailAgent for {model_key}")
            except Exception as e:
                logger.error(f"Failed to initialize EmailAgent for {model_key}: {e}")
        
        # Initialize evaluation agents
        try:
            checklist_model_id = MODELS_CONFIG[self.checklist_model]['model_id']
            self.checklist_agent = ChecklistAgent(checklist_model_id)
            logger.info(f"Initialized ChecklistAgent with {self.checklist_model}")
        except Exception as e:
            logger.error(f"Failed to initialize ChecklistAgent: {e}")
            
        try:
            judge_model_id = MODELS_CONFIG[self.judge_model]['model_id']
            self.judge_agent = JudgeAgent(judge_model_id)
            logger.info(f"Initialized JudgeAgent with {self.judge_model}")
        except Exception as e:
            logger.error(f"Failed to initialize JudgeAgent: {e}")
    
    def generate_and_rank_emails(self,
                                prompt: str,
                                topic: str,
                                style: str = "professional",
                                length: str = "medium",
                                ranking_method: str = "weighted") -> MultiModelResults:
        """
        Generate emails with multiple models and rank them
        
        Args:
            prompt: The email generation prompt
            topic: Email topic
            style: Email style preference
            length: Email length preference
            ranking_method: 'simple', 'weighted', or 'hybrid'
        """
        start_time = time.time()
        
        # Step 1: Generate emails concurrently
        logger.info(f"Generating emails with {len(self.email_agents)} models...")
        candidates = self._generate_emails_concurrent(prompt, topic, style, length)
        
        generation_time = time.time() - start_time
        eval_start_time = time.time()
        
        # Step 2: Generate checklist based on first successful email
        if not candidates:
            raise ValueError("No emails were successfully generated")
        
        reference_email = candidates[0].email_content
        checklist = self.checklist_agent.generate_checklist(prompt, reference_email, topic)
        
        # Step 3: Evaluate all candidates
        logger.info("Evaluating all email candidates...")
        self._evaluate_candidates(candidates, checklist, prompt)
        
        # Step 4: Rank candidates
        ranked_candidates = self._rank_candidates(candidates, ranking_method)
        
        evaluation_time = time.time() - eval_start_time
        
        # Step 5: Prepare results
        results = MultiModelResults(
            candidates=ranked_candidates,
            best_candidate=ranked_candidates[0],
            checklist_used=checklist,
            prompt_used=prompt,
            topic=topic,
            generation_time=generation_time,
            evaluation_time=evaluation_time
        )
        
        logger.info(f"Multi-model generation completed. Best model: {results.best_candidate.model_name}")
        return results
    
    def _generate_emails_concurrent(self,
                                  prompt: str,
                                  topic: str,
                                  style: str,
                                  length: str) -> List[EmailCandidate]:
        """Generate emails using multiple models concurrently"""
        
        candidates = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit generation tasks
            future_to_model = {}
            for model_key, agent in self.email_agents.items():
                future = executor.submit(
                    self._generate_single_email,
                    agent, model_key, prompt, topic, style, length
                )
                future_to_model[future] = model_key
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_key = future_to_model[future]
                try:
                    candidate = future.result(timeout=120)  # 2 minute timeout
                    if candidate:
                        candidates.append(candidate)
                        logger.info(f"Successfully generated email with {model_key}")
                except Exception as e:
                    logger.error(f"Email generation failed for {model_key}: {e}")
        
        return candidates
    
    def _generate_single_email(self,
                             agent: EmailAgent,
                             model_key: str,
                             prompt: str,
                             topic: str,
                             style: str,
                             length: str) -> Optional[EmailCandidate]:
        """Generate a single email with error handling"""
        
        try:
            result = agent.generate_email_with_metadata(prompt, topic, style, length)
            
            return EmailCandidate(
                email_content=result.content,
                model_name=model_key,
                model_id=agent.model_id,
                generation_result=result
            )
        except Exception as e:
            logger.error(f"Failed to generate email with {model_key}: {e}")
            return None
    
    def _evaluate_candidates(self,
                           candidates: List[EmailCandidate],
                           checklist: Checklist,
                           prompt: str):
        """Evaluate all candidates using the judge agent"""
        
        for candidate in candidates:
            try:
                evaluation = self.judge_agent.evaluate_email(
                    candidate.email_content,
                    checklist,
                    prompt
                )
                
                candidate.checklist_evaluation = evaluation
                candidate.overall_score = evaluation.overall_score
                candidate.weighted_score = evaluation.weighted_score
                
                logger.debug(f"Evaluated {candidate.model_name}: "
                           f"Overall={candidate.overall_score:.3f}, "
                           f"Weighted={candidate.weighted_score:.3f}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {candidate.model_name}: {e}")
                candidate.overall_score = 0.0
                candidate.weighted_score = 0.0
    
    def _rank_candidates(self,
                        candidates: List[EmailCandidate],
                        method: str) -> List[EmailCandidate]:
        """Rank candidates based on evaluation scores"""
        
        if method == "simple":
            # Rank by overall score
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.overall_score or 0.0,
                reverse=True
            )
        elif method == "weighted":
            # Rank by weighted score
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.weighted_score or 0.0,
                reverse=True
            )
        elif method == "hybrid":
            # Combine both scores with equal weight
            sorted_candidates = sorted(
                candidates,
                key=lambda x: ((x.overall_score or 0.0) + (x.weighted_score or 0.0)) / 2,
                reverse=True
            )
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Assign ranks
        for i, candidate in enumerate(sorted_candidates):
            candidate.rank = i + 1
        
        return sorted_candidates
    
    def compare_top_candidates(self,
                             results: MultiModelResults,
                             top_n: int = 3) -> Dict[str, Any]:
        """Compare top N candidates with detailed analysis"""
        
        top_candidates = results.candidates[:top_n]
        
        comparison = {
            "top_candidates": [],
            "score_analysis": {},
            "model_performance": {}
        }
        
        for candidate in top_candidates:
            candidate_info = {
                "rank": candidate.rank,
                "model_name": candidate.model_name,
                "overall_score": candidate.overall_score,
                "weighted_score": candidate.weighted_score,
                "generation_time": candidate.generation_result.generation_time,
                "checklist_results": []
            }
            
            if candidate.checklist_evaluation:
                for result in candidate.checklist_evaluation.checklist_results:
                    candidate_info["checklist_results"].append({
                        "question": result.question,
                        "judgment": result.judgment,
                        "yes_probability": result.yes_probability,
                        "confidence": result.confidence
                    })
            
            comparison["top_candidates"].append(candidate_info)
        
        # Score analysis
        scores = [c.overall_score for c in results.candidates if c.overall_score is not None]
        weighted_scores = [c.weighted_score for c in results.candidates if c.weighted_score is not None]
        
        comparison["score_analysis"] = {
            "overall_score_range": [min(scores), max(scores)] if scores else [0, 0],
            "weighted_score_range": [min(weighted_scores), max(weighted_scores)] if weighted_scores else [0, 0],
            "score_gap": scores[0] - scores[-1] if len(scores) > 1 else 0
        }
        
        # Model performance summary
        for candidate in results.candidates:
            comparison["model_performance"][candidate.model_name] = {
                "rank": candidate.rank,
                "overall_score": candidate.overall_score,
                "weighted_score": candidate.weighted_score,
                "model_id": candidate.model_id
            }
        
        return comparison
    
    def save_results(self, results: MultiModelResults, output_dir: Optional[str] = None):
        """Save comprehensive results to files"""
        
        if output_dir is None:
            output_dir = Path("output")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual emails
        emails_dir = output_dir / "emails"
        emails_dir.mkdir(exist_ok=True)
        
        for candidate in results.candidates:
            filename = f"rank_{candidate.rank}_{candidate.model_name}.txt"
            with open(emails_dir / filename, 'w', encoding='utf-8') as f:
                f.write(f"Rank: {candidate.rank}\n")
                f.write(f"Model: {candidate.model_name}\n")
                f.write(f"Overall Score: {candidate.overall_score:.3f}\n")
                f.write(f"Weighted Score: {candidate.weighted_score:.3f}\n")
                f.write("-" * 50 + "\n\n")
                f.write(candidate.email_content)
        
        # Save comparison report
        comparison = self.compare_top_candidates(results)
        with open(output_dir / "comparison_report.json", 'w') as f:
            import json
            json.dump(comparison, f, indent=2)
        
        # Save checklist
        with open(output_dir / "checklist.json", 'w') as f:
            import json
            json.dump(results.checklist_used.dict(), f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
