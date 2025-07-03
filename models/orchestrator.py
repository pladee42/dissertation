"""
Simplified Model Orchestrator

This module provides simplified multi-model orchestration with:
- Simple dictionary-based results
- Basic parallel execution
- Minimal abstractions
"""

import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.email_agent import EmailAgent
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent
from config.config import MODELS_CONFIG

logger = logging.getLogger(__name__)

class ModelOrchestrator:
    """Orchestrator for multiple models"""
    
    def __init__(self, 
                 email_models: List[str],
                 checklist_model: str,
                 judge_model: str,
                 max_concurrent: int = 2):
        """
        Initialize with model configurations
        
        Args:
            email_models: List of model names for email generation
            checklist_model: Model name for checklist generation
            judge_model: Model name for evaluation
            max_concurrent: Maximum concurrent executions (simplified)
        """
        self.email_models = email_models
        self.checklist_model = checklist_model
        self.judge_model = judge_model
        self.max_concurrent = min(max_concurrent, 2)  # Keep it simple
        
        logger.info(f"SimpleModelOrchestrator initialized with {len(email_models)} email models")
    
    def generate_emails(self, prompt: str, topic: str) -> List[Dict[str, Any]]:
        """Generate emails from multiple models"""
        logger.info(f"Generating emails for topic: {topic}")
        start_time = time.time()
        
        email_results = []
        
        # Use parallel execution for improved performance
        if self.max_concurrent > 1 and len(self.email_models) > 1:
            logger.info(f"Using parallel execution with {self.max_concurrent} concurrent models")
            with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
                future_to_model = {}
                
                for model_name in self.email_models:
                    model_config = MODELS_CONFIG.get(model_name, {})
                    if not model_config:
                        logger.warning(f"No config found for model: {model_name}")
                        continue
                    
                    future = executor.submit(
                        self._generate_single_email,
                        model_name, model_config, prompt, topic, start_time
                    )
                    future_to_model[future] = model_name
                
                # Collect results
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        email_results.append(result)
                        logger.info(f"Successfully generated email with {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to generate email with {model_name}: {e}")
                        email_results.append({
                            "model_name": model_name,
                            "email_content": "",
                            "generation_time": 0,
                            "success": False,
                            "error": str(e)
                        })
        else:
            # Sequential execution fallback
            for model_name in self.email_models:
                try:
                    model_config = MODELS_CONFIG.get(model_name, {})
                    if not model_config:
                        logger.warning(f"No config found for model: {model_name}")
                        continue
                    
                    result = self._generate_single_email(
                        model_name, model_config, prompt, topic, start_time
                    )
                    email_results.append(result)
                    logger.info(f"Successfully generated email with {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate email with {model_name}: {e}")
                    email_results.append({
                        "model_name": model_name,
                        "email_content": "",
                        "generation_time": 0,
                        "success": False,
                        "error": str(e)
                    })
        
        total_time = time.time() - start_time
        logger.info(f"Email generation completed in {total_time:.2f}s")
        
        return email_results
    
    def _generate_single_email(self, model_name: str, model_config: dict, prompt: str, topic: str, start_time: float) -> Dict[str, Any]:
        """Generate email with a single model (for parallel execution)"""
        logger.info(f"Generating email with model: {model_name}")
        
        # Create agent and generate email
        agent = EmailAgent(
            model_id=model_config['model_id'],
            dtype=model_config.get('dtype', 'bfloat16'),
            quantization=model_config.get('quantization', 'experts_int8'),
            model_key=model_name
        )
        
        # Extract template_id from prompt if available, default to "1"
        template_id = "1"
        email_content = agent.generate_email(prompt, topic, template_id)
        
        # Simple result dictionary
        result = {
            "model_name": model_name,
            "model_id": model_config['model_id'],
            "email_content": email_content,
            "generation_time": time.time() - start_time,
            "success": True
        }
        
        return result
    
    def create_checklist(self, user_query: str, topic: str) -> Dict[str, Any]:
        """Create evaluation checklist"""
        logger.info("Creating evaluation checklist")
        start_time = time.time()
        
        try:
            # Get model config
            model_config = MODELS_CONFIG.get(self.checklist_model, {})
            if not model_config:
                raise ValueError(f"No config found for checklist model: {self.checklist_model}")
            
            # Create agent and generate checklist
            agent = ChecklistAgent(
                model_id=model_config['model_id'],
                dtype=model_config.get('dtype', 'bfloat16'),
                quantization=model_config.get('quantization', 'experts_int8'),
                model_key=self.checklist_model
            )
            
            checklist = agent.generate_checklist(user_query, topic)
            
            result = {
                "checklist": checklist,
                "model_name": self.checklist_model,
                "generation_time": time.time() - start_time,
                "success": True
            }
            
            logger.info("Checklist created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create checklist: {e}")
            return {
                "checklist": None,
                "model_name": self.checklist_model,
                "generation_time": 0,
                "success": False,
                "error": str(e)
            }
    
    def evaluate_emails(self, email_results: List[Dict[str, Any]], checklist: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate emails against checklist"""
        logger.info("Evaluating emails against checklist")
        start_time = time.time()
        
        evaluated_results = []
        
        try:
            # Get model config
            model_config = MODELS_CONFIG.get(self.judge_model, {})
            if not model_config:
                raise ValueError(f"No config found for judge model: {self.judge_model}")
            
            # Create judge agent
            judge_agent = JudgeAgent(
                model_id=model_config['model_id'],
                dtype=model_config.get('dtype', 'bfloat16'),
                quantization=model_config.get('quantization', 'experts_int8'),
                model_key=self.judge_model
            )
            
            # Evaluate each email
            for email_result in email_results:
                if not email_result.get("success", False):
                    # Skip failed email generations
                    evaluated_results.append(email_result)
                    continue
                
                try:
                    evaluation = judge_agent.evaluate_email(
                        email_content=email_result["email_content"],
                        checklist=checklist["checklist"]
                    )
                    
                    # Add evaluation to result
                    email_result["evaluation"] = evaluation
                    email_result["overall_score"] = evaluation.get('overall_score', 0.0) if isinstance(evaluation, dict) else 0.0
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate email from {email_result['model_name']}: {e}")
                    email_result["evaluation"] = None
                    email_result["overall_score"] = 0.0
                
                evaluated_results.append(email_result)
            
            # Sort by score (highest first)
            evaluated_results.sort(key=lambda x: x.get("overall_score", 0.0), reverse=True)
            
            # Add rankings
            for i, result in enumerate(evaluated_results):
                result["rank"] = i + 1
            
            total_time = time.time() - start_time
            logger.info(f"Email evaluation completed in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to evaluate emails: {e}")
            # Return original results without evaluation
            evaluated_results = email_results
        
        return evaluated_results
    
    def generate_and_rank_emails(self, prompt: str, topic: str, user_query: str = "") -> Dict[str, Any]:
        """Complete pipeline: generate, evaluate, and rank emails"""
        logger.info("Starting complete email generation and ranking pipeline")
        start_time = time.time()
        
        # Step 1: Generate emails
        email_results = self.generate_emails(prompt, topic)
        
        # Step 2: Create checklist
        checklist_result = self.create_checklist(user_query or f"Email about {topic}", topic)
        
        # Step 3: Evaluate emails
        if checklist_result.get("success"):
            evaluated_results = self.evaluate_emails(email_results, checklist_result)
        else:
            evaluated_results = email_results
        
        # Simple final result
        result = {
            "emails": evaluated_results,
            "checklist": checklist_result,
            "best_email": evaluated_results[0] if evaluated_results else None,
            "topic": topic,
            "total_time": time.time() - start_time,
            "success": len([r for r in evaluated_results if r.get("success")]) > 0
        }
        
        logger.info(f"Pipeline completed in {result['total_time']:.2f}s")
        return result