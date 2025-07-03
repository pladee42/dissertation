"""
Template Manager

Simple template manager to load and format prompts from prompts/ folders.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class TemplateManager:
    """Simple template manager for loading prompt templates"""
    
    def __init__(self, base_path: str = "./config/prompts"):
        """
        Initialize template manager
        
        Args:
            base_path: Base path to prompts directory
        """
        self.base_path = base_path
        self.templates = {}
        
        # Load all templates at initialization
        self._load_templates()
        
        logger.info(f"TemplateManager initialized with {len(self.templates)} templates")
    
    def _load_templates(self):
        """Load all template files from prompts directory"""
        try:
            # Load email instruction templates
            instructions_path = os.path.join(self.base_path, "instructions")
            if os.path.exists(instructions_path):
                self.templates["email"] = {}
                for file in os.listdir(instructions_path):
                    if file.endswith(".md"):
                        template_name = file.replace(".md", "")
                        file_path = os.path.join(instructions_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            self.templates["email"][template_name] = f.read().strip()
            
            # Load checklist template
            checklist_path = os.path.join(self.base_path, "checklist", "checklist.md")
            if os.path.exists(checklist_path):
                with open(checklist_path, 'r', encoding='utf-8') as f:
                    self.templates["checklist"] = f.read().strip()
            
            # Load judge template
            judge_path = os.path.join(self.base_path, "judge", "judge.md")
            if os.path.exists(judge_path):
                with open(judge_path, 'r', encoding='utf-8') as f:
                    self.templates["judge"] = f.read().strip()
            
            logger.info("Templates loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = {}
    
    def get_email_template(self, template_id: str = "1") -> str:
        """
        Get email instruction template
        
        Args:
            template_id: Template ID (1, 2, 3, 4)
            
        Returns:
            Email template string
        """
        email_templates = self.templates.get("email", {})
        template = email_templates.get(template_id, "")
        
        if not template:
            logger.warning(f"Email template '{template_id}' not found")
            return "Write a professional email about [TOPIC]"
        
        return template
    
    def get_checklist_template(self) -> str:
        """
        Get checklist template
        
        Returns:
            Checklist template string
        """
        template = self.templates.get("checklist", "")
        
        if not template:
            logger.warning("Checklist template not found")
            return "Create a checklist for evaluating the email"
        
        return template
    
    def get_judge_template(self) -> str:
        """
        Get judge template
        
        Returns:
            Judge template string
        """
        template = self.templates.get("judge", "")
        
        if not template:
            logger.warning("Judge template not found")
            return "Evaluate the email against the checklist"
        
        return template
    
    def format_template(self, template: str, **kwargs) -> str:
        """
        Format template with provided variables
        
        Args:
            template: Template string
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted template string
        """
        try:
            # Simple string replacement for [TOPIC] and other placeholders
            formatted = template
            for key, value in kwargs.items():
                placeholder = f"[{key.upper()}]"
                formatted = formatted.replace(placeholder, str(value))
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting template: {e}")
            return template
    
    def list_available_templates(self) -> Dict[str, list]:
        """
        List all available templates
        
        Returns:
            Dictionary with template types and their available IDs
        """
        available = {}
        
        if "email" in self.templates:
            available["email"] = list(self.templates["email"].keys())
        
        if "checklist" in self.templates:
            available["checklist"] = ["checklist"]
        
        if "judge" in self.templates:
            available["judge"] = ["judge"]
        
        return available

# Global template manager instance
_template_manager = None

def get_template_manager() -> TemplateManager:
    """Get global template manager instance"""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager