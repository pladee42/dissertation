"""
Multi-Topic Runner

Multi-topic email generation and evaluation system with UID tracking
"""

import logging
import os
import json
import time
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path

from models.multi_topic_orchestrator import MultiTopicOrchestrator
from config.config import MODELS_CONFIG, get_setting, MODELS
from config.topic_manager import get_topic_manager
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

def save_results(results: dict, output_dir: str):
    """Save results with UID tracking"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / "multi_topic_results" / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as JSON
    results_file = output_path / "complete_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary CSV
    if results.get('successful_results'):
        import csv
        csv_file = output_path / "topic_summary.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['topic_uid', 'topic_name', 'best_model', 'best_score', 'processing_time'])
            writer.writeheader()
            
            for result in results['successful_results']:
                best_email = result.get('best_email', {})
                writer.writerow({
                    'topic_uid': result.get('topic_uid', ''),
                    'topic_name': result.get('topic_name', ''),
                    'best_model': best_email.get('model_name', ''),
                    'best_score': best_email.get('overall_score', 0),
                    'processing_time': result.get('processing_time', 0)
                })
    
    logger.info(f"Results saved to: {output_path}")
    return str(output_path)

def main():
    parser = ArgumentParser(description="Multi-topic email generation and evaluation")
    
    # Topic selection arguments
    parser.add_argument("--topics", nargs='+', 
                       help="Specify topic UIDs (e.g., T0001 T0002) or topic names")
    parser.add_argument("--topics_file", type=str,
                       help="Load topics from custom JSON file")
    parser.add_argument("--random_topics", type=int,
                       help="Select N random topics")
    parser.add_argument("--all_topics", action='store_true',
                       help="Process all available topics")
    
    # Model selection arguments
    parser.add_argument("--email_generation", type=str, 
                       choices=['small', 'medium', 'large', 'all'],
                       help="Size category for email models")
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
    
    # Processing arguments
    parser.add_argument("--max_concurrent_topics", type=int, default=1,
                       help="Maximum concurrent topics to process")
    parser.add_argument("--max_concurrent", type=int, default=1,
                       help="Maximum concurrent models per topic")
    parser.add_argument("--user_query_template", type=str, 
                       default="Email about {topic}",
                       help="Template for user query (use {topic} placeholder)")
    
    args = parser.parse_args()
    
    # Handle email_generation mode
    if args.email_generation:
        if args.email_generation == 'small':
            args.email_models = get_models_by_size('small')
        elif args.email_generation == 'medium':
            args.email_models = get_models_by_size(['small', 'medium'])
        elif args.email_generation == 'large':
            args.email_models = get_models_by_size('large')
        elif args.email_generation == 'all':
            args.email_models = get_models_by_size(['small', 'medium', 'large'])
        
        logger.info(f"Email generation mode '{args.email_generation}' selected models: {args.email_models}")
    
    # Check vLLM library availability
    backend = VLLMBackend()
    
    if not backend.is_available():
        logger.warning("vLLM library not available")
        logger.info("Running in fallback mode")
    else:
        logger.info("vLLM library available")
    
    # Load email prompt
    try:
        with open("config/prompts/instructions/instruction.md", 'r', encoding='utf-8') as f:
            email_prompt = f.read()
    except FileNotFoundError:
        email_prompt = "Write a professional email about [TOPIC]"
        logger.warning("Using default prompt")
    
    # Initialize topic manager
    topic_manager = get_topic_manager()
    
    # Determine topics to process
    topics_to_process = []
    
    if args.all_topics:
        topics_to_process = topic_manager.list_all_topics()
        logger.info("Processing all available topics")
    elif args.random_topics:
        topics_to_process = topic_manager.get_random_topics(args.random_topics)
        logger.info(f"Processing {args.random_topics} random topics")
    elif args.topics_file:
        try:
            with open(args.topics_file, 'r', encoding='utf-8') as f:
                custom_topics = json.load(f)
            topics_to_process = custom_topics if isinstance(custom_topics, list) else []
            logger.info(f"Loaded {len(topics_to_process)} topics from file")
        except Exception as e:
            logger.error(f"Error loading topics file: {e}")
            return 1
    elif args.topics:
        # Handle UIDs or topic names
        for topic_input in args.topics:
            if topic_input.startswith('T') and len(topic_input) == 5:
                # It's a UID
                topic_data = topic_manager.get_topic_by_uid(topic_input)
                if topic_data:
                    topics_to_process.append(topic_data)
                else:
                    logger.warning(f"Topic not found for UID: {topic_input}")
            else:
                # It's a topic name, search for it
                all_topics = topic_manager.list_all_topics()
                found = False
                for topic in all_topics:
                    if topic.get('topic_name', '').lower() == topic_input.lower():
                        topics_to_process.append(topic)
                        found = True
                        break
                if not found:
                    logger.warning(f"Topic not found for name: {topic_input}")
    else:
        # Default: process first 3 topics
        all_topics = topic_manager.list_all_topics()
        topics_to_process = all_topics[:3] if len(all_topics) >= 3 else all_topics
        logger.info("No specific topics selected, processing first 3 topics")
    
    if not topics_to_process:
        logger.error("No topics to process")
        return 1
    
    logger.info("=== Starting Multi-Topic Pipeline ===")
    logger.info(f"Topics to process: {len(topics_to_process)}")
    logger.info(f"Email models: {args.email_models}")
    logger.info(f"Checklist model: {args.checklist_model}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Max concurrent topics: {args.max_concurrent_topics}")
    
    # Create multi-topic orchestrator
    try:
        orchestrator = MultiTopicOrchestrator(
            email_models=args.email_models,
            checklist_model=args.checklist_model,
            judge_model=args.judge_model,
            max_concurrent=args.max_concurrent,
            max_concurrent_topics=args.max_concurrent_topics
        )
        
        # Run the pipeline
        logger.info("=== Running Multi-Topic Pipeline ===")
        results = orchestrator.process_multiple_topics(
            topics=topics_to_process,
            prompt=email_prompt,
            user_query_template=args.user_query_template
        )
        
        # Save results
        output_dir = get_setting('output_dir', './output')
        saved_path = save_results(results, output_dir)
        
        # Display results
        if results.get("success"):
            logger.info("✅ Multi-topic pipeline completed successfully")
            
            print("\n" + "="*60)
            print("MULTI-TOPIC EMAIL GENERATION RESULTS")
            print("="*60)
            print(f"Total topics processed: {results['total_topics']}")
            print(f"Successful: {results['successful_topics']}")
            print(f"Failed: {results['failed_topics']}")
            print(f"Total time: {results['total_time']:.2f}s")
            print(f"Results saved to: {saved_path}")
            
            # Show summary
            summary = results.get('summary', {})
            if summary:
                print(f"\nSUMMARY:")
                print("-" * 40)
                print(f"Average processing time per topic: {summary.get('avg_processing_time', 0):.2f}s")
                print(f"Total emails generated: {summary.get('total_emails_generated', 0)}")
                print(f"Best overall model: {summary.get('best_overall_model', 'N/A')}")
            
            # Show top results
            successful_results = results.get('successful_results', [])
            if successful_results:
                print(f"\nTOP RESULTS:")
                print("-" * 40)
                for i, result in enumerate(successful_results[:5], 1):
                    best_email = result.get('best_email', {})
                    print(f"#{i}: {result['topic_uid']} - {result['topic_name'][:50]}...")
                    print(f"   Best model: {best_email.get('model_name', 'N/A')}")
                    print(f"   Score: {best_email.get('overall_score', 0):.3f}")
                    print(f"   Time: {result.get('processing_time', 0):.2f}s")
                    print()
        else:
            logger.error("❌ Multi-topic pipeline failed")
            print("Pipeline failed. Check logs for details.")
            if results.get('failed_results'):
                print(f"Failed topics: {len(results['failed_results'])}")
            
    except Exception as e:
        logger.error(f"Error in multi-topic pipeline: {e}")
        print(f"Error: {e}")
        return 1
    
    logger.info("=== Multi-Topic Pipeline Completed ===")
    return 0

if __name__ == "__main__":
    exit(main())