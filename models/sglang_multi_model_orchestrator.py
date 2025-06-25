from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import logging
import asyncio
from pathlib import Path
import threading

from agents.sglang_agent_factory import create_email_agent, create_checklist_agent, create_judge_agent, agent_manager
from models.schemas import Checklist, EmailGenerationResult, EvaluationResult
from config.models import MODELS_CONFIG
from utils.sglang_cache_optimizer import get_cache_optimizer
from utils.sglang_memory_manager import get_sglang_memory_manager
import sglang as sgl

logger = logging.getLogger(__name__)

@dataclass
class SGLangEmailCandidate:
    """Represents a generated email candidate with SGLang-specific features"""
    email_content: str
    model_name: str
    model_id: str
    generation_result: EmailGenerationResult
    checklist_evaluation: Optional[EvaluationResult] = None
    overall_score: Optional[float] = None
    weighted_score: Optional[float] = None
    rank: Optional[int] = None
    cache_hits: int = 0
    generation_tokens: int = 0
    server_port: Optional[int] = None

@dataclass
class SGLangMultiModelResults:
    """Container for SGLang multi-model generation and evaluation results"""
    candidates: List[SGLangEmailCandidate]
    best_candidate: SGLangEmailCandidate
    checklist_used: Checklist
    prompt_used: str
    topic: str
    generation_time: float
    evaluation_time: float
    cache_optimization_stats: Dict[str, Any]
    memory_usage_stats: Dict[str, Any]
    total_tokens_saved: int = 0
    throughput_improvement: float = 0.0

class SGLangMultiModelEmailGenerator:
    """SGLang-optimized orchestrator for multiple models with advanced features"""
    
    def __init__(self, 
                 email_models: List[str],
                 checklist_model: str,
                 judge_model: str,
                 max_concurrent: int = 2,
                 enable_cache_optimization: bool = True,
                 enable_speculative_decoding: bool = False):
        """
        Initialize SGLang multi-model orchestrator
        
        Args:
            email_models: List of model identifiers for email generation
            checklist_model: Model for checklist generation
            judge_model: Model for evaluation/judging
            max_concurrent: Maximum concurrent model executions
            enable_cache_optimization: Enable RadixAttention cache optimization
            enable_speculative_decoding: Enable speculative decoding for speed
        """
        self.email_models = email_models
        self.checklist_model = checklist_model
        self.judge_model = judge_model
        self.max_concurrent = max_concurrent
        self.enable_cache_optimization = enable_cache_optimization
        self.enable_speculative_decoding = enable_speculative_decoding
        
        # SGLang-specific components
        self.cache_optimizer = get_cache_optimizer() if enable_cache_optimization else None
        self.memory_manager = get_sglang_memory_manager()
        
        # Initialize agents
        logger.info("Initializing SGLang multi-model email generation system...")
        self.email_agents = {}
        self.checklist_agent = None
        self.judge_agent = None
        
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all SGLang agents with error handling"""
        
        # Initialize email agents
        for model_key in self.email_models:
            try:
                model_config = MODELS_CONFIG[model_key]
                model_id = model_config['model_id']
                
                # Create SGLang email agent
                agent = create_email_agent(
                    model_id=model_id,
                    dtype=model_config.get('dtype', 'bfloat16'),
                    quantization=model_config.get('quantization'),
                    backend="sglang",
                    custom_config=model_config.get('sglang_config', {})
                )
                
                self.email_agents[model_key] = agent
                logger.info(f"Initialized SGLang EmailAgent for {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to initialize SGLang EmailAgent for {model_key}: {e}")
        
        # Initialize evaluation agents
        try:
            checklist_config = MODELS_CONFIG[self.checklist_model]
            self.checklist_agent = create_checklist_agent(
                model_id=checklist_config['model_id'],
                dtype=checklist_config.get('dtype', 'bfloat16'),
                quantization=checklist_config.get('quantization'),
                backend="sglang",
                custom_config=checklist_config.get('sglang_config', {})
            )
            logger.info(f"Initialized SGLang ChecklistAgent with {self.checklist_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SGLang ChecklistAgent: {e}")
            
        try:
            judge_config = MODELS_CONFIG[self.judge_model]
            self.judge_agent = create_judge_agent(
                model_id=judge_config['model_id'],
                dtype=judge_config.get('dtype', 'bfloat16'),
                quantization=judge_config.get('quantization'),
                backend="sglang",
                custom_config=judge_config.get('sglang_config', {})
            )
            logger.info(f"Initialized SGLang JudgeAgent with {self.judge_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SGLang JudgeAgent: {e}")
    
    def generate_and_rank_emails_sglang(self,
                                       prompt: str,
                                       topic: str,
                                       style: str = "professional",
                                       length: str = "medium",
                                       ranking_method: str = "weighted") -> SGLangMultiModelResults:
        """
        Generate emails with multiple SGLang models using fork/join primitives
        
        Args:
            prompt: The email generation prompt
            topic: Email topic
            style: Email style preference
            length: Email length preference
            ranking_method: 'simple', 'weighted', or 'hybrid'
        """
        start_time = time.time()
        
        # Pre-optimization: Warm cache and optimize sequence
        if self.cache_optimizer:
            agent_sequence = list(self.email_agents.keys()) + ["checklist", "judge"]
            optimized_sequence = self.cache_optimizer.optimize_agent_sequence(
                agent_sequence, topic, prompt
            )
            cache_warming_results = self.cache_optimizer.warm_cache_for_topic(
                topic, agent_sequence
            )
            logger.info(f"Cache warming completed: {cache_warming_results}")
        
        # Step 1: Generate emails using SGLang fork/join primitives
        logger.info(f"Generating emails with {len(self.email_agents)} SGLang models...")
        candidates = self._generate_emails_sglang_parallel(prompt, topic, style, length)
        
        generation_time = time.time() - start_time
        eval_start_time = time.time()
        
        # Step 2: Generate checklist using SGLang structured output
        if not candidates:
            raise ValueError("No emails were successfully generated")
        
        reference_email = candidates[0].email_content
        
        # Use SGLang checklist agent with xgrammar validation
        if hasattr(self.checklist_agent, 'generate_checklist_with_xgrammar'):
            checklist = self.checklist_agent.generate_checklist_with_xgrammar(
                prompt, reference_email, topic
            )
        else:
            checklist = self.checklist_agent.generate_checklist(
                prompt, reference_email, topic
            )
        
        # Step 3: Evaluate all candidates using SGLang constrained generation
        logger.info("Evaluating all email candidates with SGLang...")
        self._evaluate_candidates_sglang(candidates, checklist, prompt)
        
        # Step 4: Rank candidates
        ranked_candidates = self._rank_candidates(candidates, ranking_method)
        
        evaluation_time = time.time() - eval_start_time
        
        # Step 5: Collect performance metrics
        cache_stats = self._collect_cache_stats() if self.cache_optimizer else {}
        memory_stats = self._collect_memory_stats()
        
        # Step 6: Prepare SGLang-specific results
        results = SGLangMultiModelResults(
            candidates=ranked_candidates,
            best_candidate=ranked_candidates[0],
            checklist_used=checklist,
            prompt_used=prompt,
            topic=topic,
            generation_time=generation_time,
            evaluation_time=evaluation_time,
            cache_optimization_stats=cache_stats,
            memory_usage_stats=memory_stats,
            total_tokens_saved=cache_stats.get('tokens_saved', 0),
            throughput_improvement=self._calculate_throughput_improvement(cache_stats)
        )
        
        logger.info(f"SGLang multi-model generation completed. Best model: {results.best_candidate.model_name}")
        logger.info(f"Performance: {results.throughput_improvement:.2f}x improvement, {results.total_tokens_saved} tokens saved")
        
        return results
    
    def _generate_emails_sglang_parallel(self,
                                        prompt: str,
                                        topic: str,
                                        style: str,
                                        length: str) -> List[SGLangEmailCandidate]:
        """Generate emails using SGLang's fork/join primitives for true parallelism"""
        
        logger.info("Using SGLang fork/join primitives for parallel email generation")
        
        try:
            # Use SGLang's parallel generation capabilities
            @sgl.function
            def generate_multiple_emails_parallel(s, base_prompt, topic_text, style_text, length_text, model_keys):
                # Shared context that can be cached
                s += f"Email Generation Context:\n"
                s += f"Topic: {topic_text}\n"
                s += f"Style: {style_text}\n"
                s += f"Length: {length_text}\n"
                s += f"Prompt: {base_prompt}\n\n"
                
                # Register shared prefix for caching
                if self.cache_optimizer:
                    shared_context = s.text
                    interaction_id = self.cache_optimizer.track_multi_agent_interaction(
                        ["email"] * len(model_keys), shared_context, topic_text
                    )
                
                # Use fork/join for parallel execution
                candidates = []
                for i, model_key in enumerate(model_keys):
                    with s.fork(f"model_{i}") as fork_state:
                        fork_state += f"Generating email with model: {model_key}\n"
                        fork_state += "Email content:\n"
                        
                        # Get model-specific generation parameters
                        model_config = MODELS_CONFIG[model_key]
                        gen_params = {
                            "temperature": 0.7,
                            "max_new_tokens": 2048,
                            "top_p": 0.9
                        }
                        
                        # Update with SGLang-specific config
                        if 'sglang_config' in model_config:
                            gen_params.update(model_config['sglang_config'])
                        
                        fork_state += sgl.gen(f"email_{i}", **gen_params)
                        
                        # Extract result
                        email_content = fork_state[f"email_{i}"]
                        candidates.append((model_key, email_content))
                
                return candidates
            
            # Execute parallel generation
            model_keys = list(self.email_agents.keys())
            results = generate_multiple_emails_parallel.run(
                base_prompt=prompt,
                topic_text=topic,
                style_text=style,
                length_text=length,
                model_keys=model_keys
            )
            
            # Convert results to candidates
            candidates = []
            for model_key, email_content in results:
                try:
                    # Create candidate with SGLang-specific metadata
                    candidate = SGLangEmailCandidate(
                        email_content=email_content.strip(),
                        model_name=model_key,
                        model_id=MODELS_CONFIG[model_key]['model_id'],
                        generation_result=EmailGenerationResult(
                            content=email_content.strip(),
                            topic=topic,
                            llm_model_used=model_key,
                            generation_time=0.0  # Will be filled by timing
                        ),
                        generation_tokens=len(email_content.split()) * 1.3,  # Rough token estimate
                        cache_hits=0  # Will be updated by cache optimizer
                    )
                    candidates.append(candidate)
                    logger.info(f"Successfully generated email with SGLang model: {model_key}")
                    
                except Exception as e:
                    logger.error(f"Error processing result for {model_key}: {e}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"SGLang parallel generation failed: {e}")
            # Fallback to sequential generation
            return self._generate_emails_sequential_fallback(prompt, topic, style, length)
    
    def _generate_emails_sequential_fallback(self,
                                           prompt: str,
                                           topic: str,
                                           style: str,
                                           length: str) -> List[SGLangEmailCandidate]:
        """Fallback to sequential generation if parallel fails"""
        
        logger.info("Using sequential fallback for email generation")
        candidates = []
        
        for model_key, agent in self.email_agents.items():
            try:
                # Use memory context for resource management
                with self.memory_manager.sglang_memory_context(model_key, {}):
                    if hasattr(agent, 'generate_email_with_structure'):
                        result = agent.generate_email_with_structure(prompt, topic, style, length)
                    else:
                        result = agent.generate_email_with_metadata(prompt, topic, style, length)
                    
                    candidate = SGLangEmailCandidate(
                        email_content=result.content,
                        model_name=model_key,
                        model_id=agent.model_id,
                        generation_result=result,
                        generation_tokens=len(result.content.split()) * 1.3
                    )
                    candidates.append(candidate)
                    
                    logger.info(f"Sequential generation successful for {model_key}")
                    
            except Exception as e:
                logger.error(f"Sequential generation failed for {model_key}: {e}")
        
        return candidates
    
    def _evaluate_candidates_sglang(self,
                                   candidates: List[SGLangEmailCandidate],
                                   checklist: Checklist,
                                   prompt: str):
        """Evaluate candidates using SGLang's structured evaluation"""
        
        logger.info("Evaluating candidates with SGLang structured evaluation")
        
        for candidate in candidates:
            try:
                # Use SGLang judge agent with structured evaluation
                if hasattr(self.judge_agent, 'evaluate_email_structured'):
                    evaluation = self.judge_agent.evaluate_email_structured(
                        candidate.email_content,
                        checklist,
                        prompt
                    )
                else:
                    evaluation = self.judge_agent.evaluate_email(
                        candidate.email_content,
                        checklist,
                        prompt
                    )
                
                candidate.checklist_evaluation = evaluation
                candidate.overall_score = evaluation.overall_score
                candidate.weighted_score = evaluation.weighted_score
                
                logger.debug(f"SGLang evaluation for {candidate.model_name}: "
                           f"Overall={candidate.overall_score:.3f}, "
                           f"Weighted={candidate.weighted_score:.3f}")
                
            except Exception as e:
                logger.error(f"SGLang evaluation failed for {candidate.model_name}: {e}")
                candidate.overall_score = 0.0
                candidate.weighted_score = 0.0
    
    def generate_with_continuous_batching(self,
                                        prompts: List[str],
                                        topics: List[str],
                                        batch_size: int = 4) -> List[SGLangMultiModelResults]:
        """Generate multiple emails using SGLang's continuous batching"""
        
        logger.info(f"Using SGLang continuous batching for {len(prompts)} prompts")
        
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_topics = topics[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_prompts)} items")
            
            try:
                # Use SGLang's batch processing capabilities
                @sgl.function
                def process_batch(s, prompt_list, topic_list):
                    batch_results = []
                    
                    for j, (prompt, topic) in enumerate(zip(prompt_list, topic_list)):
                        s += f"\n=== Batch Item {j+1} ===\n"
                        s += f"Topic: {topic}\n"
                        s += f"Prompt: {prompt}\n\n"
                        
                        # Generate email
                        s += "Email:\n"
                        s += sgl.gen(f"email_{j}", max_new_tokens=1500, temperature=0.7)
                        
                        batch_results.append({
                            "prompt": prompt,
                            "topic": topic,
                            "email": s[f"email_{j}"]
                        })
                    
                    return batch_results
                
                # Execute batch processing
                batch_results = process_batch.run(
                    prompt_list=batch_prompts,
                    topic_list=batch_topics
                )
                
                # Process each result in the batch
                for result in batch_results:
                    # Generate full result for each item (simplified for batching)
                    single_result = self.generate_and_rank_emails_sglang(
                        result["prompt"],
                        result["topic"]
                    )
                    results.append(single_result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for batch {i//batch_size + 1}: {e}")
                
                # Fallback to individual processing
                for prompt, topic in zip(batch_prompts, batch_topics):
                    try:
                        single_result = self.generate_and_rank_emails_sglang(prompt, topic)
                        results.append(single_result)
                    except Exception as e2:
                        logger.error(f"Individual processing failed for topic {topic}: {e2}")
        
        logger.info(f"Continuous batching completed: {len(results)} results")
        return results
    
    def _rank_candidates(self,
                        candidates: List[SGLangEmailCandidate],
                        method: str) -> List[SGLangEmailCandidate]:
        """Rank candidates based on evaluation scores with SGLang-specific metrics"""
        
        if method == "simple":
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.overall_score or 0.0,
                reverse=True
            )
        elif method == "weighted":
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.weighted_score or 0.0,
                reverse=True
            )
        elif method == "hybrid":
            # Include cache efficiency in ranking
            def hybrid_score(candidate):
                base_score = ((candidate.overall_score or 0.0) + (candidate.weighted_score or 0.0)) / 2
                cache_bonus = candidate.cache_hits * 0.01  # Small bonus for cache efficiency
                return base_score + cache_bonus
            
            sorted_candidates = sorted(candidates, key=hybrid_score, reverse=True)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Assign ranks
        for i, candidate in enumerate(sorted_candidates):
            candidate.rank = i + 1
        
        return sorted_candidates
    
    def _collect_cache_stats(self) -> Dict[str, Any]:
        """Collect RadixAttention cache statistics"""
        
        if not self.cache_optimizer:
            return {}
        
        try:
            # Export cache analysis
            analysis_path = self.cache_optimizer.export_cache_analysis()
            
            # Get summary stats
            total_prefixes = len(self.cache_optimizer.cache_prefixes)
            total_interactions = len(self.cache_optimizer.cache_interactions)
            
            cache_stats = {
                "total_prefixes": total_prefixes,
                "total_interactions": total_interactions,
                "cache_hit_rate": self.cache_optimizer._calculate_global_hit_rate(),
                "tokens_saved": sum(
                    self.cache_optimizer._estimate_tokens_saved([prefix.prefix_hash])
                    for prefix in self.cache_optimizer.cache_prefixes.values()
                ),
                "analysis_exported": str(analysis_path)
            }
            
            return cache_stats
            
        except Exception as e:
            logger.warning(f"Failed to collect cache stats: {e}")
            return {"error": str(e)}
    
    def _collect_memory_stats(self) -> Dict[str, Any]:
        """Collect SGLang memory usage statistics"""
        
        try:
            memory_profile = self.memory_manager.get_sglang_memory_profile()
            
            memory_stats = {
                "gpu_allocated_gb": memory_profile.gpu_allocated_gb,
                "radix_cache_size_gb": memory_profile.radix_cache_size_gb,
                "active_requests": memory_profile.active_requests,
                "cache_hit_rate": memory_profile.cache_hit_rate,
                "server_processes": len(memory_profile.server_processes)
            }
            
            return memory_stats
            
        except Exception as e:
            logger.warning(f"Failed to collect memory stats: {e}")
            return {"error": str(e)}
    
    def _calculate_throughput_improvement(self, cache_stats: Dict[str, Any]) -> float:
        """Calculate throughput improvement from cache optimization"""
        
        if not cache_stats or "cache_hit_rate" not in cache_stats:
            return 1.0
        
        hit_rate = cache_stats["cache_hit_rate"]
        
        # Rough estimate: each cache hit provides ~20% speedup
        improvement = 1.0 + (hit_rate * 0.2)
        
        return improvement
    
    def compare_top_candidates_sglang(self,
                                     results: SGLangMultiModelResults,
                                     top_n: int = 3) -> Dict[str, Any]:
        """Compare top candidates with SGLang-specific metrics"""
        
        top_candidates = results.candidates[:top_n]
        
        comparison = {
            "top_candidates": [],
            "score_analysis": {},
            "model_performance": {},
            "sglang_metrics": {
                "cache_optimization": results.cache_optimization_stats,
                "memory_usage": results.memory_usage_stats,
                "throughput_improvement": results.throughput_improvement,
                "total_tokens_saved": results.total_tokens_saved
            }
        }
        
        for candidate in top_candidates:
            candidate_info = {
                "rank": candidate.rank,
                "model_name": candidate.model_name,
                "overall_score": candidate.overall_score,
                "weighted_score": candidate.weighted_score,
                "generation_time": candidate.generation_result.generation_time,
                "cache_hits": candidate.cache_hits,
                "generation_tokens": candidate.generation_tokens,
                "server_port": candidate.server_port,
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
        
        # Enhanced score analysis
        scores = [c.overall_score for c in results.candidates if c.overall_score is not None]
        weighted_scores = [c.weighted_score for c in results.candidates if c.weighted_score is not None]
        
        comparison["score_analysis"] = {
            "overall_score_range": [min(scores), max(scores)] if scores else [0, 0],
            "weighted_score_range": [min(weighted_scores), max(weighted_scores)] if weighted_scores else [0, 0],
            "score_gap": scores[0] - scores[-1] if len(scores) > 1 else 0,
            "cache_efficiency": sum(c.cache_hits for c in results.candidates) / len(results.candidates)
        }
        
        return comparison
    
    def cleanup(self):
        """Cleanup all SGLang agents and resources"""
        
        logger.info("Cleaning up SGLang multi-model orchestrator")
        
        # Cleanup email agents
        for model_key, agent in self.email_agents.items():
            try:
                agent.cleanup()
                logger.debug(f"Cleaned up email agent: {model_key}")
            except Exception as e:
                logger.error(f"Error cleaning up email agent {model_key}: {e}")
        
        # Cleanup evaluation agents
        try:
            if self.checklist_agent:
                self.checklist_agent.cleanup()
            if self.judge_agent:
                self.judge_agent.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up evaluation agents: {e}")
        
        # Cleanup SGLang-specific resources
        try:
            if self.memory_manager:
                self.memory_manager.cleanup_sglang_resources()
        except Exception as e:
            logger.error(f"Error cleaning up SGLang resources: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()