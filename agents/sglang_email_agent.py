from models.model_factory import create_model_instance
from models.schemas import BaseModel
from utils.retry import retry_with_backoff
from config.settings import settings
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import sglang as sgl

logger = logging.getLogger(__name__)

class EmailGenerationResult(BaseModel):
    """Schema for email generation results"""
    content: str
    topic: str
    llm_model_used: str
    generation_time: Optional[float] = None
    prompt_used: Optional[str] = None

class SGLangEmailAgent:
    """SGLang-optimized Email Generation Agent with structured generation primitives"""
    
    def __init__(self, model_id: str, dtype: str, quantization: str, custom_config: Optional[Dict[str, Any]] = None):
        """Initialize the SGLang email agent with specified model"""
        self.model_id = model_id
        self.model_name = model_id.split('/')[-1]
        
        logger.info(f"Initializing SGLangEmailAgent with model: {model_id}")
        
        # Initialize the SGLang model
        self.llm = create_model_instance(
            model_name=self._get_model_name_from_id(model_id),
            backend="sglang",
            custom_config=custom_config
        )
        
        # Email-optimized generation parameters
        self.email_config = {
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "top_p": 0.9,
            "stop": None
        }
        
        if custom_config:
            self.email_config.update(custom_config)
    
    def _get_model_name_from_id(self, model_id: str) -> str:
        """Extract model name from model_id for config lookup"""
        # Map model IDs to config names
        id_to_name = {
            'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 'deepseek-r1-1.5b',
            'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 'deepseek-r1-8b',
            'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 'deepseek-r1-70b',
            'unsloth/Llama-3.2-3B-Instruct': 'llama-3-3b',
            'casperhansen/llama-3-8b-instruct-awq': 'llama-3-8b',
            'gaunernst/gemma-3-4b-it-qat-autoawq': 'gemma-3-4b',
            'Qwen/Qwen3-8B-AWQ': 'qwen-3-8b',
            'kishizaki-sci/Llama-4-Scout-17B-16E-Instruct-AWQ': 'llama-4-109b'
        }
        return id_to_name.get(model_id, model_id.split('/')[-1])
    
    @retry_with_backoff(max_retries=settings.max_retries)
    def generate_email(self, 
                      prompt: str, 
                      topic: str,
                      style: str = "professional",
                      length: str = "medium",
                      custom_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate email content using SGLang's generation primitives"""
        
        # Use custom parameters if provided, otherwise use email-optimized config
        generation_params = custom_params if custom_params else self.email_config
        
        logger.info(f"Generating email for topic: {topic} using SGLang")
        
        try:
            # Use SGLang's structured generation
            @sgl.function
            def generate_structured_email(s, base_prompt, topic_text, style_text, length_text):
                # Email generation with SGLang primitives
                s += base_prompt.replace('[TOPIC]', topic_text)
                s += f"\n\nStyle: {self._get_style_instruction(style_text)}"
                s += f"\nLength: {self._get_length_instruction(length_text)}"
                s += "\n\nEmail Requirements:"
                s += "\n1. Include an appropriate subject line"
                s += "\n2. Use proper email structure (greeting, body, closing)"
                s += "\n3. Ensure the content is relevant to the topic"
                s += "\n4. Use clear and engaging language"
                s += "\n5. Include a call-to-action if appropriate"
                s += "\n6. Maintain consistency in tone throughout"
                s += "\n\nPlease generate a complete email response:\n"
                
                # Generate the email content
                s += sgl.gen("email_content", **generation_params)
            
            # Execute the SGLang function
            state = generate_structured_email.run(
                base_prompt=prompt,
                topic_text=topic,
                style_text=style,
                length_text=length
            )
            
            email_content = state["email_content"]
            
            # Post-process the email content
            email_content = self._post_process_email(email_content)
            
            logger.info(f"SGLang email generated successfully")
            
            return email_content
            
        except Exception as e:
            logger.error(f"SGLang email generation failed: {e}")
            raise
    
    def generate_email_with_structure(self, 
                                    prompt: str, 
                                    topic: str,
                                    style: str = "professional",
                                    length: str = "medium") -> EmailGenerationResult:
        """Generate email with structured output using SGLang primitives"""
        
        logger.info(f"Generating structured email for topic: {topic}")
        
        try:
            @sgl.function
            def generate_structured_email_with_metadata(s, base_prompt, topic_text, style_text, length_text):
                # Enhanced structured generation with multiple components
                s += base_prompt.replace('[TOPIC]', topic_text)
                s += f"\n\nStyle: {self._get_style_instruction(style_text)}"
                s += f"\nLength: {self._get_length_instruction(length_text)}"
                
                # Generate subject line first
                s += "\n\nFirst, generate an appropriate subject line:\nSubject: "
                s += sgl.gen("subject_line", max_new_tokens=50, stop=["\n"])
                
                # Generate email body
                s += "\n\nNow generate the complete email body:\n"
                s += sgl.gen("email_body", **self.email_config)
                
                # Generate metadata
                s += "\n\nEmail generation complete."
            
            # Execute the SGLang function
            import time
            start_time = time.time()
            
            state = generate_structured_email_with_metadata.run(
                base_prompt=prompt,
                topic_text=topic,
                style_text=style,
                length_text=length
            )
            
            generation_time = time.time() - start_time
            
            # Combine subject and body
            subject_line = state["subject_line"].strip()
            email_body = state["email_body"].strip()
            
            # Format complete email
            complete_email = f"Subject: {subject_line}\n\n{email_body}"
            complete_email = self._post_process_email(complete_email)
            
            enhanced_prompt = self._enhance_prompt(prompt, topic, style, length)
            
            return EmailGenerationResult(
                content=complete_email,
                topic=topic,
                llm_model_used=self.model_name,
                generation_time=generation_time,
                prompt_used=enhanced_prompt
            )
            
        except Exception as e:
            logger.error(f"SGLang structured email generation failed: {e}")
            raise
    
    def generate_email_with_branching(self, 
                                    prompt: str, 
                                    topic: str,
                                    num_variants: int = 3) -> List[str]:
        """Generate multiple email variants using SGLang's fork/join primitives"""
        
        logger.info(f"Generating {num_variants} email variants for topic: {topic}")
        
        try:
            @sgl.function
            def generate_email_variants(s, base_prompt, topic_text, num_variants):
                s += base_prompt.replace('[TOPIC]', topic_text)
                s += "\n\nGenerate multiple professional email variants:\n"
                
                # Use SGLang's fork to generate multiple variants
                variants = []
                for i in range(num_variants):
                    with s.fork(f"variant_{i}") as fork_state:
                        fork_state += f"\n\nVariant {i+1}:\n"
                        fork_state += sgl.gen(f"email_variant_{i}", **self.email_config)
                        variants.append(fork_state[f"email_variant_{i}"])
                
                return variants
            
            # Execute the SGLang function
            variants = generate_email_variants.run(
                base_prompt=prompt,
                topic_text=topic,
                num_variants=num_variants
            )
            
            # Post-process all variants
            processed_variants = [self._post_process_email(variant) for variant in variants]
            
            logger.info(f"Generated {len(processed_variants)} email variants successfully")
            
            return processed_variants
            
        except Exception as e:
            logger.error(f"SGLang email variant generation failed: {e}")
            # Fallback to single email generation
            single_email = self.generate_email(prompt, topic)
            return [single_email]
    
    def _get_style_instruction(self, style: str) -> str:
        """Get style instruction for email generation"""
        style_instructions = {
            "professional": "Write in a professional, formal tone suitable for business communication.",
            "friendly": "Write in a warm, friendly tone while maintaining professionalism.",
            "casual": "Write in a casual, conversational tone.",
            "persuasive": "Write in a persuasive tone to encourage action or engagement."
        }
        return style_instructions.get(style, style_instructions['professional'])
    
    def _get_length_instruction(self, length: str) -> str:
        """Get length instruction for email generation"""
        length_instructions = {
            "short": "Keep the email concise and to the point (150-250 words).",
            "medium": "Write a well-structured email with adequate detail (250-400 words).",
            "long": "Provide comprehensive information in a detailed email (400+ words)."
        }
        return length_instructions.get(length, length_instructions['medium'])
    
    def _enhance_prompt(self, 
                       base_prompt: str, 
                       topic: str, 
                       style: str, 
                       length: str) -> str:
        """Enhance the base prompt with email-specific instructions"""
        
        enhanced_prompt = base_prompt.replace('[TOPIC]', topic)
        enhanced_prompt += f"\n\nStyle: {self._get_style_instruction(style)}"
        enhanced_prompt += f"\nLength: {self._get_length_instruction(length)}"
        
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
            "Email content:",
            "Variant 1:",
            "Variant 2:",
            "Variant 3:"
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
            filename = f"{safe_topic}_{self.model_name}_sglang.txt"
        
        file_path = output_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Model: {self.model_name} (SGLang)\n")
            f.write(f"Backend: SGLang with RadixAttention\n")
            f.write("-" * 50 + "\n\n")
            f.write(email_content)
        
        logger.info(f"SGLang email saved to: {file_path}")
        return file_path
    
    def cleanup(self):
        """Cleanup the SGLang email agent and release resources"""
        logger.info(f"Cleaning up SGLangEmailAgent with model: {self.model_name}")
        
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm.cleanup()
                logger.info("SGLangEmailAgent cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during SGLangEmailAgent cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the SGLang email agent"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "backend": "sglang",
            "email_config": self.email_config,
            "model_info": self.llm.get_model_info(),
            "features": [
                "structured_generation",
                "radix_attention",
                "fork_join_primitives",
                "constrained_output"
            ]
        }