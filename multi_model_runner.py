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
from models.orchestrator import SimpleModelOrchestrator
from config.config import MODELS_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(description="Simplified multi-model email generation")
    parser.add_argument("--topic", type=str, 
                       default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_models", nargs='+', 
                       default=["deepseek-r1-1.5b", "llama-3-3b"],
                       choices=list(MODELS_CONFIG.keys()),
                       help="List of models for email generation")
    parser.add_argument("--checklist_model", type=str, 
                       default="deepseek-r1-8b",
                       choices=list(MODELS_CONFIG.keys()))
    parser.add_argument("--judge_model", type=str, 
                       default="gemma-3-4b",
                       choices=list(MODELS_CONFIG.keys()))
    
    args = parser.parse_args()
    
    logger.info("=== Starting Simplified Multi-Model Pipeline ===")
    logger.info(f"Topic: {args.topic}")
    logger.info(f"Email models: {args.email_models}")
    logger.info(f"Checklist model: {args.checklist_model}")
    logger.info(f"Judge model: {args.judge_model}")
    
    # Load email prompt
    try:
        with open("prompts/instructions/2.txt", 'r', encoding='utf-8') as f:
            email_prompt = f.read()
    except FileNotFoundError:
        email_prompt = "Write a professional email about [TOPIC]"
        logger.warning("Using default prompt")
    
    # Create orchestrator
    try:
        orchestrator = SimpleModelOrchestrator(
            email_models=args.email_models,
            checklist_model=args.checklist_model,
            judge_model=args.judge_model,
            max_concurrent=1  # Keep it simple
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