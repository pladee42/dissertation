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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from models.orchestrator import ModelOrchestrator
from config.config import MODELS_CONFIG, get_setting, MODELS, CHECKLIST_MODES, list_models_by_size_group
from models.vllm_backend import VLLMBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_models_by_size(size_categories):
    """Get models by size categories (small, medium, large) - legacy function"""
    if isinstance(size_categories, str):
        size_categories = [size_categories]
    
    models = []
    for model_name, config in MODELS.items():
        if config.get('size') in size_categories:
            models.append(model_name)
    return models

def get_models_by_generation_type(generation_type):
    """Get models by generation type with DPO support"""
    if generation_type == 'small':
        return list_models_by_size_group('small')
    elif generation_type == 'medium':
        return get_models_by_size(['small', 'medium'])  # Legacy behavior
    elif generation_type == 'large':
        return list_models_by_size_group('large')
    elif generation_type == 'all':
        models = []
        for size in ['small', 'medium', 'large']:
            models.extend(list_models_by_size_group(size))
        return models
    elif generation_type == 'small-dpo':
        return list_models_by_size_group('small-dpo')
    elif generation_type == 'medium-dpo':
        return list_models_by_size_group('medium-dpo')
    elif generation_type == 'all-dpo':
        return list_models_by_size_group('all-dpo')
    elif generation_type == 'base-only':
        return list_models_by_size_group('base-only')
    else:
        return []

def main():
    parser = ArgumentParser(description="Multi-model email generation")
    parser.add_argument("--topic", type=str, 
                       default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_generation", type=str, 
                       choices=['small', 'medium', 'large', 'all', 'small-dpo', 'medium-dpo', 'all-dpo', 'base-only'],
                       help="Size category for email models (supports DPO: small-dpo, medium-dpo, all-dpo, base-only)")
    parser.add_argument("--email_models", nargs='+', 
                       default=["tinyllama-1.1b", "phi-3-mini"],
                       choices=list(MODELS_CONFIG.keys()),
                       help="List of models for email generation (overrides --email_generation)")
    parser.add_argument("--checklist_model", type=str, 
                       default="gpt-4.1-nano",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--judge_model", type=str, 
                       default="gemini-2.5-flash",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--example_email", type=str, 
                       default="1",
                       help="Example email file to use (e.g., '1' for example_email/1.md)")
    parser.add_argument("--checklist_mode", type=str,
                       default="enhanced",
                       choices=list(CHECKLIST_MODES.values()),
                       help="Checklist generation mode: enhanced (full context), extract_only (minimal), or preprocess (two-step)")
    
    args = parser.parse_args()
    
    # Handle email_generation mode (including DPO support)
    if args.email_generation and not hasattr(args, 'email_models_specified'):
        args.email_models = get_models_by_generation_type(args.email_generation)
        
        if not args.email_models:
            logger.error(f"No models found for generation type: {args.email_generation}")
            return
        
        logger.info(f"Email generation mode '{args.email_generation}' selected models: {args.email_models}")
        
        # Log DPO model info
        from config.config import is_dpo_model, get_base_model_for_dpo
        dpo_count = sum(1 for m in args.email_models if is_dpo_model(m))
        if dpo_count > 0:
            logger.info(f"Selected {dpo_count} DPO models and {len(args.email_models) - dpo_count} base models")
    
    # Check vLLM library availability
    backend = VLLMBackend()
    
    if not backend.is_available():
        logger.warning("vLLM library not available")
        logger.info("Running in fallback mode")
    else:
        logger.info("vLLM library available")
    
    logger.info("=== Starting Simplified Multi-Model Pipeline ===")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Email models: {args.email_models}")
    logger.info(f"Checklist model: {args.checklist_model}")
    logger.info(f"Judge model: {args.judge_model}")
    
    # Load email prompt
    try:
        with open("config/prompts/email.md", 'r', encoding='utf-8') as f:
            email_prompt = f.read()
    except FileNotFoundError:
        email_prompt = "Write a professional email about [TOPIC]"
        logger.warning("Using default prompt")
    
    # Load example email and replace placeholder
    try:
        with open(f"config/prompts/example_email/{args.example_email}.md", 'r', encoding='utf-8') as f:
            example_email_content = f.read()
        email_prompt = email_prompt.replace("[EXAMPLE_EMAIL]", example_email_content)
    except FileNotFoundError:
        logger.warning(f"Example email file not found: {args.example_email}.md")
        email_prompt = email_prompt.replace("[EXAMPLE_EMAIL]", "No example available")
    
    # Create orchestrator
    try:
        orchestrator = ModelOrchestrator(
            email_models=args.email_models,
            checklist_model=args.checklist_model,
            judge_model=args.judge_model,
            max_concurrent=1,
            checklist_mode=args.checklist_mode
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