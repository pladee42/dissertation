from models.llm import ModelInference
from models.schemas import BaseModel
from utils.retry import retry_with_backoff
from config.settings import settings
from pathlib import Path
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class EmailGenerationResult(BaseModel):
    """Schema for email generation results"""
    content: str
    topic: str
    llm_model_used: str
    generation_time: Optional[float] = None
    prompt_used: Optional[str] = None

class EmailAgent:
    """Enhanced Email Generation Agent with robust error handling"""
    
    def __init__(self, model_id: str, dtype: str, quantization: str, custom_config: Optional[Dict[str, Any]] = None):
        """Initialize the email agent with specified model"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"Initializing EmailAgent with model: {model_id}")
        
        # Initialize the LLM with email-optimized configuration
        email_config = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        if custom_config:
            email_config.update(custom_config)
        
        self.llm = ModelInference(
            model_id=model_id, 
            dtype=dtype,
            quantization=quantization,
            custom_config=custom_config
        )
        
        self.email_config = email_config
        
    @retry_with_backoff(max_retries=settings.max_retries)
    def generate_email(self, 
                      prompt: str, 
                      topic: str,
                      style: str = "professional",
                      length: str = "medium",
                      custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate email content with enhanced prompt engineering"""
        
        # Enhance the prompt with email-specific instructions
        enhanced_prompt = self._enhance_prompt(prompt, topic, style, length)
        
        # Use custom parameters if provided, otherwise use email-optimized config
        generation_params = custom_params if custom_params else self.email_config
        
        logger.info(f"Generating email for topic: {topic}")
        
        try:
            result = self.llm.generate(
                query=enhanced_prompt,
                model_name=self.model_name,
                custom_params=generation_params,
                remove_cot=True,
                return_full_output=True
            )
            
            email_content = result["text"]
            
            # Post-process the email content
            email_content = self._post_process_email(email_content)
            
            logger.info(f"Email generated successfully in {result['generation_time']:.2f}s")
            
            return email_content
            
        except Exception as e:
            logger.error(f"Email generation failed: {e}")
            raise
    
    def generate_email_with_metadata(self, 
                                   prompt: str, 
                                   topic: str,
                                   style: str = "professional",
                                   length: str = "medium") -> EmailGenerationResult:
        """Generate email with full metadata"""
        
        enhanced_prompt = self._enhance_prompt(prompt, topic, style, length)
        
        result = self.llm.generate(
            query=enhanced_prompt,
            model_name=self.model_name,
            custom_params=self.email_config,
            remove_cot=True,
            return_full_output=True
        )
        
        email_content = self._post_process_email(result["text"])
        
        return EmailGenerationResult(
            content=email_content,
            topic=topic,
            llm_model_used=self.model_name,
            generation_time=result["generation_time"],
            prompt_used=enhanced_prompt
        )
    
    def _enhance_prompt(self, 
                       base_prompt: str, 
                       topic: str, 
                       style: str, 
                       length: str) -> str:
        """Enhance the base prompt with email-specific instructions"""
        
        style_instructions = {
            "professional": "Write in a professional, formal tone suitable for business communication.",
            "friendly": "Write in a warm, friendly tone while maintaining professionalism.",
            "casual": "Write in a casual, conversational tone.",
            "persuasive": "Write in a persuasive tone to encourage action or engagement."
        }
        
        length_instructions = {
            "short": "Keep the email concise and to the point (150-250 words).",
            "medium": "Write a well-structured email with adequate detail (250-400 words).",
            "long": "Provide comprehensive information in a detailed email (400+ words)."
        }
        
        # Replace topic placeholder
        enhanced_prompt = base_prompt.replace('[TOPIC]', topic)
        
        # Add style and length instructions
        enhanced_prompt += f"\n\nStyle: {style_instructions.get(style, style_instructions['professional'])}"
        enhanced_prompt += f"\nLength: {length_instructions.get(length, length_instructions['medium'])}"
        
        # Add email formatting instructions
        enhanced_prompt += """

Email Requirements:
1. Include an appropriate subject line
2. Use proper email structure (greeting, body, closing)
3. Ensure the content is relevant to the topic
4. Use clear and engaging language
5. Include a call-to-action if appropriate
6. Maintain consistency in tone throughout

Please generate a complete email response:"""
        
        return enhanced_prompt
    
    def _post_process_email(self, email_content: str) -> str:
        """Post-process the generated email content"""
        
        # Remove any unwanted prefixes or suffixes
        email_content = email_content.strip()
        
        # Remove common unwanted phrases
        unwanted_phrases = [
            "Here's the email:",
            "Here is the email:",
            "Email:",
            "Subject line:",
            "Email content:"
        ]
        
        for phrase in unwanted_phrases:
            if email_content.lower().startswith(phrase.lower()):
                email_content = email_content[len(phrase):].strip()
        
        # Ensure proper email structure
        if not self._has_proper_structure(email_content):
            email_content = self._add_basic_structure(email_content)
        
        return email_content
    
    def _has_proper_structure(self, email_content: str) -> bool:
        """Check if email has proper structure"""
        content_lower = email_content.lower()
        
        # Check for basic email elements
        has_greeting = any(greeting in content_lower for greeting in [
            "dear", "hello", "hi", "greetings", "good morning", "good afternoon"
        ])
        
        has_closing = any(closing in content_lower for closing in [
            "sincerely", "regards", "best", "thank you", "yours"
        ])
        
        return has_greeting and has_closing
    
    def _add_basic_structure(self, email_content: str) -> str:
        """Add basic email structure if missing"""
        
        # If content doesn't start with a greeting, add one
        content_lower = email_content.lower()
        if not any(greeting in content_lower[:50] for greeting in [
            "dear", "hello", "hi", "greetings"
        ]):
            email_content = "Dear Recipient,\n\n" + email_content
        
        # If content doesn't end with a closing, add one
        if not any(closing in content_lower[-100:] for closing in [
            "sincerely", "regards", "best", "thank you"
        ]):
            email_content += "\n\nBest regards"
        
        return email_content
    
    def save_email(self, 
                   email_content: str, 
                   topic: str, 
                   filename: Optional[str] = None) -> Path:
        """Save generated email to file"""
        
        output_dir = Path(settings.output_dir) / "emails"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            # Create filename from topic and model
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_')
            filename = f"{safe_topic}_{self.model_name}.txt"
        
        file_path = output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("-" * 50 + "\n\n")
            f.write(email_content)
        
        logger.info(f"Email saved to: {file_path}")
        return file_path
    
    def cleanup(self):
        """Cleanup the email agent and release resources"""
        logger.info(f"Cleaning up EmailAgent with model: {self.model_name}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("EmailAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during EmailAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the email agent"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "email_config": self.email_config,
            "model_info": self.llm.get_model_info()
        }
