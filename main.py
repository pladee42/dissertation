"""
Simplified Main Script

This provides simple email generation with:
- Basic model selection
- Simple argument handling
- Minimal complexity
"""

import logging
import os
from config.config import MODELS, get_model_config, get_setting
from agents.email_agent import EmailAgent
from models.sglang_backend import SGLangBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt(prompt_file: str = "1.txt") -> str:
    """Load a simple prompt from file"""
    prompt_path = f"./prompts/instructions/{prompt_file}"
    
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {prompt_path}")
        return "Write a professional email about [TOPIC]"

def generate_simple_email(model_name: str, topic: str, prompt: str) -> str:
    """Generate email using specified model"""
    logger.info(f"Generating email with {model_name} for topic: {topic}")
    
    try:
        # Get model config
        model_config = get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create agent
        agent = EmailAgent(
            model_id=model_config['model_id'],
            dtype="bfloat16",  # Simple default
            quantization="experts_int8"  # Simple default
        )
        
        # Generate email with template_id
        template_id = os.environ.get("TEMPLATE_ID", "1")
        email_content = agent.generate_email(prompt, topic, template_id)
        
        # Save output
        output_file = f"./output/emails/{model_name}_{topic.replace(' ', '_')}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(email_content)
        
        logger.info(f"Email saved to: {output_file}")
        return email_content
        
    except Exception as e:
        logger.error(f"Error generating email with {model_name}: {e}")
        return f"Error: {str(e)}"

def main():
    """Simple main function"""
    logger.info("Starting simple email generation")
    
    # Check SGLang server connectivity first
    sglang_url = get_setting('sglang_server_url', 'http://localhost:30000')
    backend = SGLangBackend(base_url=sglang_url)
    
    if not backend.is_available():
        logger.warning(f"SGLang server not available at {sglang_url}")
        logger.info("Running in fallback mode without SGLang")
    else:
        logger.info(f"SGLang server is available at {sglang_url}")
    
    # Simple configuration - no complex argument parsing
    model_name = "deepseek-r1-1.5b"  # Default model
    topic = "AI Research Collaboration"  # Default topic
    prompt_file = "1.txt"  # Default prompt
    
    # Allow simple environment variable overrides
    model_name = os.environ.get("EMAIL_MODEL", model_name)
    topic = os.environ.get("EMAIL_TOPIC", topic)
    prompt_file = os.environ.get("PROMPT_FILE", prompt_file)
    
    logger.info(f"Configuration: model={model_name}, topic={topic}, prompt={prompt_file}")
    
    # Load prompt
    prompt = load_prompt(prompt_file)
    
    # Generate email
    email_content = generate_simple_email(model_name, topic, prompt)
    
    # Print result
    print("\n" + "="*50)
    print("GENERATED EMAIL:")
    print("="*50)
    print(email_content)
    print("="*50 + "\n")
    
    logger.info("Email generation completed")

if __name__ == "__main__":
    main()