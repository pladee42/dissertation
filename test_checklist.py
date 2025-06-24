from argparse import ArgumentParser
from pathlib import Path
from agents.email_agent import EmailAgent
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent
from config.models import MODELS_CONFIG
from config.settings import settings
import json
from utils.cleanup import agent_session

def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_mode", type=str, default='2', help="Select prompt type to use. e.g. 1.txt")
    parser.add_argument("--topic", type=str, default="Polar Bears Rescue by University of Sheffield")
    parser.add_argument("--email_model", type=str, default="deepseek-r1-1.5b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--checklist_model", type=str, default="deepseek-r1-8b", choices=MODELS_CONFIG.keys())
    parser.add_argument("--judge_model", type=str, default="gemma-3-12b", choices=MODELS_CONFIG.keys())
    
    args = parser.parse_args()
    
    # Initialize agents
    print("Initializing agents...")
    email_agent = EmailAgent(MODELS_CONFIG[args.email_model]['model_id'], MODELS_CONFIG[args.email_model]['dtype'], MODELS_CONFIG[args.email_model]['quantization'])
    checklist_agent = ChecklistAgent(MODELS_CONFIG[args.checklist_model]['model_id'], MODELS_CONFIG[args.checklist_model]['dtype'], MODELS_CONFIG[args.checklist_model]['quantization'])
    judge_agent = JudgeAgent(MODELS_CONFIG[args.judge_model]['model_id'], MODELS_CONFIG[args.judge_model]['dtype'], MODELS_CONFIG[args.judge_model]['quantization'])
    
    # Load email prompt
    with open(f"prompts/instructions/{args.prompt_mode}.txt", 'r', encoding='utf-8') as f:
        email_prompt = f.read().replace('[TOPIC]', args.topic)
    
    # Generate email
    print("Generating email...")
    with agent_session(email_agent):
        email_content = email_agent.generate_email(email_prompt, args.topic)
        email_agent.save_email(email_content=email_content, topic=args.topic, filename=f"{args.prompt_mode}|{args.email_model}.txt")
    
    # Generate checklist
    print("Generating checklist...")
    checklist = checklist_agent.generate_checklist(email_prompt, email_content, args.topic)
    checklist_agent.save_checklist(checklist, f"{args.prompt_mode}|{args.email_model}.txt")
    
    # Evaluate email
    print("Evaluating email...")
    evaluation = judge_agent.evaluate_email(email_content, checklist, email_prompt)
    
    # Save results
    output_dir = Path(settings.output_dir) / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"evaluation_{args.topic.replace(' ', '_')}.json", 'w') as f:
        json.dump(evaluation.model_dump(), f, indent=2)
    
    print(f"Overall Score: {evaluation.overall_score:.2f}")
    print(f"Weighted Score: {evaluation.weighted_score:.2f}")

if __name__ == "__main__":
    main()
