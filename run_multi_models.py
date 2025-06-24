from argparse import ArgumentParser
from pathlib import Path
from models.multi_email_generator import MultiModelEmailGenerator
from config.models import MODELS_CONFIG
import json

def main():
    parser = ArgumentParser(description="Multi-model email generation and ranking")
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
    parser.add_argument("--max_concurrent", type=int, default=3,
                       help="Maximum concurrent model executions")
    parser.add_argument("--output_dir", type=str, default="./output/multi_model")
    
    args = parser.parse_args()
    
    # Load email prompt
    with open("prompts/instructions/01.txt", 'r', encoding='utf-8') as f:
        email_prompt = f.read().replace('[TOPIC]', args.topic)
    
    # Initialize multi-model generator
    print(f"Initializing multi-model generator with {len(args.email_models)} models...")
    generator = MultiModelEmailGenerator(
        email_models=args.email_models,
        checklist_model=args.checklist_model,
        judge_model=args.judge_model,
        max_concurrent=args.max_concurrent
    )
    
    # Generate and rank emails
    print("Starting multi-model email generation and evaluation...")
    results = generator.generate_and_rank_emails(
        prompt=email_prompt,
        topic=args.topic,
        style=args.style,
        length=args.length,
        ranking_method=args.ranking_method
    )
    
    # Display results
    print("\n" + "="*60)
    print("MULTI-MODEL EMAIL GENERATION RESULTS")
    print("="*60)
    
    print(f"Topic: {results.topic}")
    print(f"Models used: {', '.join(args.email_models)}")
    print(f"Generation time: {results.generation_time:.2f}s")
    print(f"Evaluation time: {results.evaluation_time:.2f}s")
    print(f"Ranking method: {args.ranking_method}")
    
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
    
    # Save results
    generator.save_results(results, args.output_dir)
    print(f"\nDetailed results saved to: {args.output_dir}")
    
    # Show comparison of top 3
    comparison = generator.compare_top_candidates(results, top_n=3)
    print(f"\nTOP 3 COMPARISON:")
    print("-" * 40)
    for candidate_info in comparison["top_candidates"]:
        print(f"#{candidate_info['rank']}: {candidate_info['model_name']} "
              f"(Overall: {candidate_info['overall_score']:.3f}, "
              f"Weighted: {candidate_info['weighted_score']:.3f})")

if __name__ == "__main__":
    main()
