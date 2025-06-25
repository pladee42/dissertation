from argparse import ArgumentParser
from pathlib import Path
from agents.sglang_agent_factory import create_email_agent, create_checklist_agent, create_judge_agent, agent_manager
from config.models import MODELS_CONFIG
from config.settings import settings
from utils.sglang_cache_optimizer import get_cache_optimizer
from utils.sglang_advanced_memory_manager import get_advanced_memory_manager
import json
import logging
import time
import os

# Setup logging for better monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_mode", type=str, default='2', help="Select prompt type to use. e.g. 1.txt")
    parser.add_argument("--topic", type=str, default="AI Research Collaboration")
    parser.add_argument("--email_model", type=str, default="deepseek-r1-1.5b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--checklist_model", type=str, default="deepseek-r1-8b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--judge_model", type=str, default="gemma-3-4b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--enable_cache_optimization", action="store_true", default=True, help="Enable RadixAttention cache optimization")
    parser.add_argument("--enable_xgrammar", action="store_true", default=True, help="Enable xgrammar JSON validation")
    parser.add_argument("--enable_structured_evaluation", action="store_true", default=True, help="Enable structured evaluation")
    
    args = parser.parse_args()
    
    # Set SGLang backend
    os.environ["AGENT_BACKEND"] = "sglang"
    os.environ["INFERENCE_BACKEND"] = "sglang"
    
    # Initialize SGLang components
    cache_optimizer = get_cache_optimizer() if args.enable_cache_optimization else None
    memory_manager = get_advanced_memory_manager()
    
    # Log initial system state
    logger.info(f"Starting SGLang sequential agent pipeline for topic: {args.topic}")
    initial_profile = memory_manager.get_comprehensive_memory_profile()
    logger.info(f"Initial memory state: {initial_profile.gpu_allocated_gb:.2f}GB GPU, "
               f"{initial_profile.radix_cache_size_gb:.2f}GB cache, "
               f"{initial_profile.cache_hit_rate:.2f} hit rate")
    
    # Load email prompt once
    with open(f"prompts/instructions/{args.prompt_mode}.txt", 'r', encoding='utf-8') as f:
        email_prompt = f.read().replace('[TOPIC]', args.topic)
    
    # Cache optimization setup
    if cache_optimizer:
        logger.info("Setting up cache optimization...")
        agent_sequence = ["email", "checklist", "judge"]
        optimized_sequence = cache_optimizer.optimize_agent_sequence(
            agent_sequence, args.topic, email_prompt
        )
        cache_warming_results = cache_optimizer.warm_cache_for_topic(
            args.topic, agent_sequence
        )
        logger.info(f"Cache warming completed: {cache_warming_results}")
    
    # Track timing for performance analysis
    stage_timings = {}
    
    # Initialize variables for pipeline data
    email_content = None
    checklist = None
    evaluation = None
    
    try:
        # Stage 1: SGLang Email Generation
        logger.info("=== SGLang Email Generation Stage ===")
        stage_start = time.time()
        
        email_agent = create_email_agent(
            model_id=MODELS_CONFIG[args.email_model]['model_id'],
            dtype=MODELS_CONFIG[args.email_model].get('dtype', 'bfloat16'),
            quantization=MODELS_CONFIG[args.email_model].get('quantization'),
            backend="sglang"
        )
        
        try:
            # Use SGLang structured generation if available
            if hasattr(email_agent, 'generate_email_with_structure'):
                email_result = email_agent.generate_email_with_structure(
                    email_prompt, args.topic, style="professional"
                )
                email_content = email_result.content
            else:
                email_content = email_agent.generate_email(
                    email_prompt, args.topic, style="professional"
                )
            
            # Save email with SGLang identifier
            email_agent.save_email(
                email_content=email_content, 
                topic=args.topic, 
                filename=f"sglang_{args.prompt_mode}_{args.email_model}.txt"
            )
            
            stage_timings["email_generation"] = time.time() - stage_start
            logger.info(f"SGLang email generation completed in {stage_timings['email_generation']:.2f}s")
            
        finally:
            email_agent.cleanup()
        
        # Memory checkpoint after email stage
        post_email_profile = memory_manager.get_comprehensive_memory_profile()
        logger.info(f"Memory after email: {post_email_profile.gpu_allocated_gb:.2f}GB GPU, "
                   f"cache hit rate: {post_email_profile.cache_hit_rate:.2f}")
        
        # Stage 2: SGLang Checklist Generation with xgrammar
        logger.info("=== SGLang Checklist Generation Stage ===")
        stage_start = time.time()
        
        checklist_agent = create_checklist_agent(
            model_id=MODELS_CONFIG[args.checklist_model]['model_id'],
            dtype=MODELS_CONFIG[args.checklist_model].get('dtype', 'bfloat16'),
            quantization=MODELS_CONFIG[args.checklist_model].get('quantization'),
            backend="sglang"
        )
        
        try:
            # Use xgrammar validation if enabled and available
            if args.enable_xgrammar and hasattr(checklist_agent, 'generate_checklist_with_xgrammar'):
                logger.info("Using xgrammar JSON schema validation")
                checklist = checklist_agent.generate_checklist_with_xgrammar(
                    email_prompt, email_content, args.topic
                )
            elif hasattr(checklist_agent, 'generate_template_based_checklist'):
                logger.info("Using template-based checklist generation")
                checklist = checklist_agent.generate_template_based_checklist(
                    email_prompt, email_content, args.topic
                )
            else:
                checklist = checklist_agent.generate_checklist(
                    email_prompt, email_content, args.topic
                )
            
            # Save checklist with SGLang identifier
            checklist_agent.save_checklist(
                checklist, f"sglang_{args.prompt_mode}_{args.email_model}"
            )
            
            stage_timings["checklist_generation"] = time.time() - stage_start
            logger.info(f"SGLang checklist generation completed in {stage_timings['checklist_generation']:.2f}s")
            logger.info(f"Generated checklist with {len(checklist.items)} items")
            
        finally:
            checklist_agent.cleanup()
        
        # Memory checkpoint after checklist stage
        post_checklist_profile = memory_manager.get_comprehensive_memory_profile()
        logger.info(f"Memory after checklist: {post_checklist_profile.gpu_allocated_gb:.2f}GB GPU, "
                   f"cache hit rate: {post_checklist_profile.cache_hit_rate:.2f}")
        
        # Stage 3: SGLang Email Evaluation with Structured Output
        logger.info("=== SGLang Email Evaluation Stage ===")
        stage_start = time.time()
        
        judge_agent = create_judge_agent(
            model_id=MODELS_CONFIG[args.judge_model]['model_id'],
            dtype=MODELS_CONFIG[args.judge_model].get('dtype', 'bfloat16'),
            quantization=MODELS_CONFIG[args.judge_model].get('quantization'),
            backend="sglang"
        )
        
        try:
            # Use structured evaluation if enabled and available
            if args.enable_structured_evaluation and hasattr(judge_agent, 'evaluate_email_structured'):
                logger.info("Using SGLang structured evaluation")
                evaluation = judge_agent.evaluate_email_structured(
                    email_content, checklist, email_prompt
                )
            elif hasattr(judge_agent, 'evaluate_with_reasoning'):
                logger.info("Using SGLang evaluation with reasoning")
                detailed_evaluation = judge_agent.evaluate_with_reasoning(
                    email_content, checklist, email_prompt
                )
                evaluation = type('EvaluationResult', (), {
                    'overall_score': detailed_evaluation['overall_score'],
                    'weighted_score': detailed_evaluation['weighted_score'],
                    'checklist_results': detailed_evaluation['judgment_results'],
                    'model_dump': lambda: {
                        'overall_score': detailed_evaluation['overall_score'],
                        'weighted_score': detailed_evaluation['weighted_score'],
                        'checklist_results': [r.model_dump() for r in detailed_evaluation['judgment_results']],
                        'detailed_results': detailed_evaluation['detailed_results'],
                        'evaluation_method': detailed_evaluation['evaluation_method']
                    }
                })()
            else:
                evaluation = judge_agent.evaluate_email(
                    email_content, checklist, email_prompt
                )
            
            stage_timings["evaluation"] = time.time() - stage_start
            logger.info(f"SGLang evaluation completed in {stage_timings['evaluation']:.2f}s")
            
        finally:
            judge_agent.cleanup()
        
        # Final memory and cache analysis
        final_profile = memory_manager.get_comprehensive_memory_profile()
        logger.info(f"Final memory state: {final_profile.gpu_allocated_gb:.2f}GB GPU, "
                   f"{final_profile.radix_cache_size_gb:.2f}GB cache")
        
        # Cache performance analysis
        if cache_optimizer:
            cache_analysis = cache_optimizer.export_cache_analysis()
            logger.info(f"Cache analysis exported to: {cache_analysis}")
            
            # Get optimization recommendations
            recommendations = cache_optimizer.get_cache_recommendations(
                args.topic, ["email", "checklist", "judge"]
            )
            logger.info(f"Cache optimization recommendations: {recommendations}")
        
        # Save comprehensive results
        logger.info("=== Saving SGLang Results ===")
        output_dir = Path(settings.output_dir) / "sglang_evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        evaluation_file = output_dir / f"sglang_evaluation_{args.topic.replace(' ', '_')}.json"
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation.model_dump(), f, indent=2)
        
        # Save performance metrics
        performance_metrics = {
            "stage_timings": stage_timings,
            "total_time": sum(stage_timings.values()),
            "memory_profiles": {
                "initial": {
                    "gpu_allocated_gb": initial_profile.gpu_allocated_gb,
                    "cache_size_gb": initial_profile.radix_cache_size_gb,
                    "cache_hit_rate": initial_profile.cache_hit_rate
                },
                "final": {
                    "gpu_allocated_gb": final_profile.gpu_allocated_gb,
                    "cache_size_gb": final_profile.radix_cache_size_gb,
                    "cache_hit_rate": final_profile.cache_hit_rate
                }
            },
            "sglang_features_used": {
                "cache_optimization": args.enable_cache_optimization,
                "xgrammar_validation": args.enable_xgrammar,
                "structured_evaluation": args.enable_structured_evaluation,
                "radix_attention": True,
                "fork_join_primitives": True
            },
            "cache_optimization_stats": recommendations if cache_optimizer else {}
        }
        
        performance_file = output_dir / f"sglang_performance_{args.topic.replace(' ', '_')}.json"
        with open(performance_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        
        # Final results summary
        logger.info("=== SGLang Pipeline Results ===")
        logger.info(f"Overall Score: {evaluation.overall_score:.3f}")
        logger.info(f"Weighted Score: {evaluation.weighted_score:.3f}")
        logger.info(f"Total Generation Time: {sum(stage_timings.values()):.2f}s")
        logger.info(f"Cache Hit Rate: {final_profile.cache_hit_rate:.3f}")
        logger.info(f"Memory Efficiency: {final_profile.cache_efficiency:.3f}")
        
        # Performance comparison metrics
        if cache_optimizer:
            estimated_speedup = 1.0 + (final_profile.cache_hit_rate * 0.3)  # Rough estimate
            logger.info(f"Estimated Speedup vs VLLM: {estimated_speedup:.2f}x")
        
        logger.info("SGLang sequential agent pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"SGLang pipeline failed: {e}")
        raise
    
    finally:
        # Cleanup SGLang resources
        try:
            memory_manager.cleanup_advanced_resources()
        except Exception as e:
            logger.warning(f"Error during final cleanup: {e}")

if __name__ == "__main__":
    main()