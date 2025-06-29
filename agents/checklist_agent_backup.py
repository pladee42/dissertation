from models.llm import ModelInference
from models.schemas import Checklist, ChecklistItem
from utils.retry import retry_with_backoff
from utils.output import extract_and_validate_json
from config.settings import settings
import json
from pathlib import Path

class ChecklistAgent:
    def __init__(self, model_id: str, dtype: str, quantization: str):
        self.llm = ModelInference(model_id=model_id, dtype=dtype, quantization=quantization)
        self.model_id = model_id
    
    @retry_with_backoff(max_retries=settings.max_retries)
    def generate_checklist(self, user_query: str, reference_response: str, topic: str) -> Checklist:
        """Generate structured checklist with validation and retry logic"""
        
        # Load prompt template
        prompt_path = Path("config/prompts/checklist/checklist.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # Format prompt
        prompt = prompt_template.replace('{user_query}', user_query)
        prompt = prompt.replace('{reference_response}', reference_response)
        prompt += "\n\nIMPORTANT: Return your response in valid JSON format matching this schema:\n"
        prompt += Checklist.model_json_schema(indent=2)
        
        # Generate response
        response = self.llm.generate(
            query=prompt, 
            model_name=self.model_id.split('/')[-1],
            remove_cot=True
        )
        
        # Extract and validate JSON
        checklist_data = extract_and_validate_json(response, Checklist)
        
        # Ensure topic is set
        checklist_data.topic = topic
        
        return checklist_data
    
    def save_checklist(self, checklist: Checklist, filename: str):
        """Save checklist to both JSON and text formats"""
        output_dir = Path(settings.output_dir) / "checklist"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(checklist.model_dump(), f, indent=2)
        
        # Save readable text
        txt_path = output_dir / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Checklist for: {checklist.topic}\n\n")
            for i, item in enumerate(checklist.items, 1):
                f.write(f"{i}. {item.question}\n")
                f.write(f"   Expected: {item.correct_answer}\n")
                f.write(f"   Priority: {item.priority.value}\n\n")
    
    def cleanup(self):
        """Cleanup the checklist agent and release resources"""
        logger = logging.getLogger(__name__)
        logger.info(f"Cleaning up ChecklistAgent with model: {self.model_id.split('/')[-1]}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("ChecklistAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during ChecklistAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
