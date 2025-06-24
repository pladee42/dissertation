from argparse import ArgumentParser
from pathlib import Path
from models.multi_email_generator import MultiModelEmailGenerator
from config.models import MODELS_CONFIG
from utils.cleanup import get_gpu_memory_info, check_memory_availability, _aggressive_memory_cleanup
import json
import logging

# Setup enhanced logging for multi-model operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(description="Multi-model email generation and ranking with enhanced memory management")
    parser.add_argument("--topic", type=str, 
                       default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_models", nargs='+', 
                       default=["deepseek-r1-1.5b", "deepseek-r1-7b", "gemma-3-12b"],
                       choices=list(MODELS_CONFIG.keys()),
                       help="List of models for email generation")
    parser.add_argument("--checklist_model", type=str, 
                       default="deepseek-r1-7b",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--judge_model", type=str, 
                       default="gemma-3-12b",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--style", type=str, default="professional",
                       choices=["professional", "friendly", "casual", "persuasive"])
    parser.add_argument("--length", type=str, default="medium",
                       choices=["short", "medium", "long"])
    parser.add_argument("--ranking_method", type=str, default="weighted",
                       choices=["simple", "weighted", "hybrid"])
    parser.add_argument("--max_concurrent", type=int, default=1,
                       help="Maximum concurrent model executions (reduced default for memory safety)")
    parser.add_argument("--output_dir", type=str, default="./output/multi_model")
    parser.add_argument("--sequential_mode", action="store_true", default=True,
                       help="Use sequential processing for better memory management")
    
    args = parser.parse_args()
    
    # Initial memory assessment
    logger.info("=== Multi-Model Pipeline Starting ===")
    initial_memory = get_gpu_memory_info()
    if initial_memory["available"]:
        logger.info(f"Initial GPU memory: {initial_memory['allocated_gb']:.2f}GB allocated, "
                   f"{initial_memory['free_gb']:.2f}GB free")
        
        # Estimate memory requirements
        estimated_memory_per_model = 4.0  # GB per model
        total_estimated = len(args.email_models) * estimated_memory_per_model + 8.0  # Extra for checklist/judge
        
        if args.sequential_mode:
            logger.info(f"Sequential mode: Will process {len(args.email_models)} models one at a time")
            required_memory = estimated_memory_per_model + 2.0  # Safety margin
        else:
            logger.info(f"Concurrent mode: Processing up to {args.max_concurrent} models simultaneously")
            required_memory = total_estimated
        
        if not check_memory_availability(required_memory, safety_margin=2.0):
            logger.warning("Insufficient memory detected, forcing sequential mode")
            args.sequential_mode = True
            args.max_concurrent = 1
    
    # Load email prompt
    with open("prompts/instructions/01.txt", 'r', encoding='utf-8') as f:
        email_prompt = f.read().replace('[TOPIC]', args.topic)
    
    # Perform pre-initialization cleanup
    logger.info("Performing pre-initialization cleanup...")
    _aggressive_memory_cleanup()
    
    # Initialize multi-model generator with memory-aware settings
    logger.info(f"Initializing memory-aware multi-model generator...")
    logger.info(f"Email models: {args.email_models}")
    logger.info(f"Checklist model: {args.checklist_model}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Sequential mode: {args.sequential_mode}")
    
    try:
        generator = MultiModelEmailGenerator(
            email_models=args.email_models,
            checklist_model=args.checklist_model,
            judge_model=args.judge_model,
            max_concurrent=args.max_concurrent if not args.sequential_mode else 1
        )
        
        # Memory checkpoint after initialization
        post_init_memory = get_gpu_memory_info()
        if post_init_memory["available"]:
            memory_used = post_init_memory['allocated_gb'] - initial_memory['allocated_gb']
            logger.info(f"Memory after generator init: {post_init_memory['allocated_gb']:.2f}GB "
                       f"(+{memory_used:.2f}GB)")
        
        # Generate and rank emails with memory monitoring
        logger.info("=== Starting Multi-Model Generation and Evaluation ===")
        results = generator.generate_and_rank_emails(
            prompt=email_prompt,
            topic=args.topic,
            style=args.style,
            length=args.length,
            ranking_method=args.ranking_method
        )
        
    except RuntimeError as e:
        if "memory" in str(e).lower():
            logger.error(f"Memory error during multi-model processing: {e}")
            logger.info("Attempting recovery with aggressive cleanup...")
            _aggressive_memory_cleanup()
            raise
        else:
            raise
    
    # Memory checkpoint after generation
    generation_memory = get_gpu_memory_info()
    if generation_memory["available"]:
        logger.info(f"Memory after generation: {generation_memory['allocated_gb']:.2f}GB allocated")
    
    # Display results with enhanced information
    logger.info("=== Displaying Results ===")
    print("\n" + "="*60)
    print("MULTI-MODEL EMAIL GENERATION RESULTS")
    print("="*60)
    
    print(f"Topic: {results.topic}")
    print(f"Models used: {', '.join(args.email_models)}")
    print(f"Generation time: {results.generation_time:.2f}s")
    print(f"Evaluation time: {results.evaluation_time:.2f}s")
    print(f"Ranking method: {args.ranking_method}")
    print(f"Sequential mode: {args.sequential_mode}")
    
    # Memory efficiency metrics
    if initial_memory["available"] and generation_memory["available"]:
        peak_memory_used = generation_memory['allocated_gb'] - initial_memory['allocated_gb']
        print(f"Peak additional memory used: {peak_memory_used:.2f}GB")
    
    print(f"\nRANKING RESULTS:")
    print("-" * 40)
    for candidate in results.candidates:
        print(f"#{candidate.rank}: {candidate.model_name}")
        print(f"   Overall Score: {candidate.overall_score:.3f}")
        print(f"   Weighted Score: {candidate.weighted_score:.3f}")
        if candidate.generation_result.generation_time:
            print(f"   Generation Time: {candidate.generation_result.generation_time:.2f}s")
        print()
    
    print(f"BEST EMAIL (from {results.best_candidate.model_name}):")
    print("-" * 40)
    print(results.best_candidate.email_content[:200] + "...")
    
    # Save results with memory information
    try:
        generator.save_results(results, args.output_dir)
        logger.info(f"Results saved to: {args.output_dir}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Show comparison of top 3
    try:
        comparison = generator.compare_top_candidates(results, top_n=3)
        print(f"\nTOP 3 COMPARISON:")
        print("-" * 40)
        for candidate_info in comparison["top_candidates"]:
            print(f"#{candidate_info['rank']}: {candidate_info['model_name']} "
                  f"(Overall: {candidate_info['overall_score']:.3f}, "
                  f"Weighted: {candidate_info['weighted_score']:.3f})")
    except Exception as e:
        logger.error(f"Error in top 3 comparison: {e}")
    
    # Final cleanup and memory summary
    logger.info("=== Final Cleanup ===")
    try:
        if hasattr(generator, 'cleanup'):
            generator.cleanup()
    except Exception as e:
        logger.warning(f"Generator cleanup failed: {e}")
    
    _aggressive_memory_cleanup()
    
    final_memory = get_gpu_memory_info()
    if final_memory["available"] and initial_memory["available"]:
        memory_delta = final_memory['allocated_gb'] - initial_memory['allocated_gb']
        logger.info(f"Final memory: {final_memory['allocated_gb']:.2f}GB (Δ{memory_delta:+.2f}GB from start)")
        
        if abs(memory_delta) < 0.1:
            logger.info("✓ Excellent memory cleanup - minimal residual memory")
        elif abs(memory_delta) < 1.0:
            logger.info("✓ Good memory cleanup - small residual memory")
        else:
            logger.warning(f"⚠ Significant residual memory: {memory_delta:.2f}GB")
    
    logger.info("=== Multi-Model Pipeline Completed ===")

if __name__ == "__main__":
    main()
