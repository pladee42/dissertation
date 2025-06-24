from argparse import ArgumentParser
from pathlib import Path
from agents.email_agent import EmailAgent
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent
from config.models import MODELS_CONFIG
from config.settings import settings
import json
import logging
from utils.cleanup import sequential_agent_stage, sequential_pipeline, get_gpu_memory_info

# Setup logging for better monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_mode", type=str, default='2', help="Select prompt type to use. e.g. 1.txt")
    parser.add_argument("--topic", type=str, default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_model", type=str, default="deepseek-r1-1.5b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--checklist_model", type=str, default="deepseek-r1-8b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--judge_model", type=str, default="gemma-3-12b", choices=MODELS_CONFIG.keys())
    
    args = parser.parse_args()
    
    # Log initial system state
    logger.info(f"Starting sequential agent pipeline for topic: {args.topic}")
    initial_memory = get_gpu_memory_info()
    if initial_memory["available"]:
        logger.info(f"Initial GPU memory: {initial_memory['allocated_gb']:.2f}GB allocated, "
                   f"{initial_memory['free_gb']:.2f}GB free")
    
    # Load email prompt once
    with open(f"prompts/instructions/{args.prompt_mode}.txt", 'r', encoding='utf-8') as f:
        email_prompt = f.read().replace('[TOPIC]', args.topic)
    
    # Initialize variables for pipeline data
    email_content = None
    checklist = None
    evaluation = None
    
    # Sequential processing pipeline
    with sequential_pipeline():
        
        # Stage 1: Email Generation
        with sequential_agent_stage(EmailAgent, MODELS_CONFIG[args.email_model], 
                                  "Email Generation", required_memory_gb=4.0) as email_agent:
            
            logger.info("=== Email Generation Stage ===")
            email_content = email_agent.generate_email(email_prompt, args.topic)
            email_agent.save_email(
                email_content=email_content, 
                topic=args.topic, 
                filename=f"{args.prompt_mode}|{args.email_model}.txt"
            )
            logger.info("Email generation completed and saved")
        
        # Memory checkpoint between stages
        memory_checkpoint = get_gpu_memory_info()
        if memory_checkpoint["available"]:
            logger.info(f"Memory after email stage: {memory_checkpoint['allocated_gb']:.2f}GB allocated")
        
        # Stage 2: Checklist Generation
        with sequential_agent_stage(ChecklistAgent, MODELS_CONFIG[args.checklist_model], 
                                  "Checklist Generation", required_memory_gb=4.0) as checklist_agent:
            
            logger.info("=== Checklist Generation Stage ===")
            checklist = checklist_agent.generate_checklist(email_prompt, email_content, args.topic)
            checklist_agent.save_checklist(checklist, f"{args.prompt_mode}|{args.email_model}.txt")
            logger.info("Checklist generation completed and saved")
        
        # Memory checkpoint between stages  
        memory_checkpoint = get_gpu_memory_info()
        if memory_checkpoint["available"]:
            logger.info(f"Memory after checklist stage: {memory_checkpoint['allocated_gb']:.2f}GB allocated")
        
        # Stage 3: Email Evaluation
        with sequential_agent_stage(JudgeAgent, MODELS_CONFIG[args.judge_model], 
                                  "Email Evaluation", required_memory_gb=4.0) as judge_agent:
            
            logger.info("=== Email Evaluation Stage ===")
            evaluation = judge_agent.evaluate_email(email_content, checklist, email_prompt)
            logger.info("Email evaluation completed")
    
    # Save final results
    logger.info("=== Saving Results ===")
    output_dir = Path(settings.output_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"evaluation_{args.topic.replace(' ', '_')}.json", 'w') as f:
        json.dump(evaluation.model_dump(), f, indent=2)
    
    # Final results and memory summary
    logger.info("=== Pipeline Results ===")
    logger.info(f"Overall Score: {evaluation.overall_score:.2f}")
    logger.info(f"Weighted Score: {evaluation.weighted_score:.2f}")
    
    final_memory = get_gpu_memory_info()
    if final_memory["available"] and initial_memory["available"]:
        memory_delta = final_memory['allocated_gb'] - initial_memory['allocated_gb']
        logger.info(f"Final GPU memory: {final_memory['allocated_gb']:.2f}GB allocated "
                   f"(Δ{memory_delta:+.2f}GB from start)")
    
    logger.info("Sequential agent pipeline completed successfully!")

if __name__ == "__main__":
    main()
