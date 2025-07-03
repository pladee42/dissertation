"""
Simplified Multi-Model Runner

This provides simple multi-model email generation with:
- Basic argument parsing
- Simple orchestrator usage
- Minimal complexity
"""

import logging
import os
from argparse import ArgumentParser
from models.orchestrator import ModelOrchestrator
from config.config import MODELS_CONFIG, get_setting, MODELS
from models.vllm_backend import VLLMBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_models_by_size(size_categories):
    """Get models by size categories (small, medium, large)"""
    if isinstance(size_categories, str):
        size_categories = [size_categories]
    
    models = []
    for model_name, config in MODELS.items():
        if config.get('size') in size_categories:
            models.append(model_name)
    return models

def main():
    parser = ArgumentParser(description="Multi-model email generation")
    parser.add_argument("--topic", type=str, 
                       default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_generation", type=str, 
                       choices=['small', 'medium', 'large', 'all'],
                       help="Size category for email models (small, medium uses small+medium, large, all)")
    parser.add_argument("--email_models", nargs='+', 
                       default=["deepseek-r1-1.5b", "llama-3-3b"],
                       choices=list(MODELS_CONFIG.keys()),
                       help="List of models for email generation (overrides --email_generation)")
    parser.add_argument("--checklist_model", type=str, 
                       default="deepseek-r1-8b",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--judge_model", type=str, 
                       default="gemma-3-4b",
                       choices=list(MODELS_CONFIG.keys()))
    
    args = parser.parse_args()
    
    # Handle email_generation mode
    if args.email_generation and not hasattr(args, 'email_models_specified'):
        if args.email_generation == 'small':
            args.email_models = get_models_by_size('small')
        elif args.email_generation == 'medium':
            args.email_models = get_models_by_size(['small', 'medium'])
        elif args.email_generation == 'large':
            args.email_models = get_models_by_size('large')
        elif args.email_generation == 'all':
            args.email_models = get_models_by_size(['small', 'medium', 'large'])
        
        logger.info(f"Email generation mode '{args.email_generation}' selected models: {args.email_models}")
    
    # Check vLLM server connectivity first
    server_url = get_setting('server_url', 'http://localhost:30000')
    backend = VLLMBackend(base_url=server_url)
    
    if not backend.is_available():
        logger.warning(f"vLLM server not available at {server_url}")
        logger.info("Running in fallback mode without vLLM")
    else:
        logger.info(f"vLLM server: {server_url} (available)")
    
    logger.info("=== Starting Simplified Multi-Model Pipeline ===")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Email models: {args.email_models}")
    logger.info(f"Checklist model: {args.checklist_model}")
    logger.info(f"Judge model: {args.judge_model}")
    
    # Load email prompt
    try:
        with open("config/prompts/instructions/2.md", 'r', encoding='utf-8') as f:
            email_prompt = f.read()
    except FileNotFoundError:
        email_prompt = "Write a professional email about [TOPIC]"
        logger.warning("Using default prompt")
    
    # Create orchestrator
    try:
        orchestrator = ModelOrchestrator(
            email_models=args.email_models,
            checklist_model=args.checklist_model,
            judge_model=args.judge_model,
            max_concurrent=1
        )
        
        # Run the pipeline
        logger.info("=== Running Pipeline ===")
        results = orchestrator.generate_and_rank_emails(
            prompt=email_prompt,
            topic=args.topic,
            user_query=f"Email about {args.topic}"
        )
        
        # Display results
        if results.get("success"):
            logger.info("✅ Pipeline completed successfully")
            
            print("\n" + "="*60)
            print("MULTI-MODEL EMAIL GENERATION RESULTS")
            print("="*60)
            print(f"Topic: {results['topic']}")
            print(f"Total time: {results['total_time']:.2f}s")
            print(f"Generated {len(results['emails'])} emails")
            
            print(f"\nRANKING RESULTS:")
            print("-" * 40)
            for i, email_result in enumerate(results['emails'], 1):
                print(f"#{i}: {email_result['model_name']}")
                if 'overall_score' in email_result:
                    print(f"   Score: {email_result['overall_score']:.3f}")
                print(f"   Success: {email_result.get('success', False)}")
                print()
            
            # Show best email
            best_email = results.get('best_email')
            if best_email:
                print(f"BEST EMAIL (from {best_email['model_name']}):")
                print("-" * 40)
                content = best_email.get('email_content', '')
                print(content[:300] + "..." if len(content) > 300 else content)
        else:
            logger.error("❌ Pipeline failed")
            print("Pipeline failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Error in multi-model pipeline: {e}")
        print(f"Error: {e}")
        return 1
    
    logger.info("=== Pipeline Completed ===")
    return 0

if __name__ == "__main__":
    exit(main())