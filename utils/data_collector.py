"""
Simple Training Data Collector

Collects training data from the email generation pipeline for future fine-tuning.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataCollector:
    """Simple data collector for training data"""
    
    def __init__(self, base_dir: str = "./output/training_data"):
        """
        Initialize data collector
        
        Args:
            base_dir: Base directory for training data storage
        """
        self.base_dir = Path(base_dir)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = self.base_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        logger.info(f"Training data directories created at: {self.base_dir}")
    
    def _get_session_id(self) -> str:
        """Generate unique session ID"""
        return str(uuid.uuid4())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def _get_daily_folder(self) -> Path:
        """Get today's folder path"""
        today = datetime.now().strftime("%Y-%m-%d")
        daily_folder = self.base_dir / today
        daily_folder.mkdir(exist_ok=True)
        return daily_folder
    
    def _get_filename(self) -> str:
        """Generate filename for session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}.json"
    
    def _validate_session_data(self, session_data: Dict[str, Any]) -> bool:
        """Validate session data structure"""
        required_fields = ["session_id", "timestamp", "topic", "input", "models_used", "outputs", "rankings", "pipeline_metadata"]
        
        for field in required_fields:
            if field not in session_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate key sub-structures
        if not isinstance(session_data["outputs"], dict):
            logger.error("Invalid outputs structure")
            return False
            
        if not isinstance(session_data["rankings"], list):
            logger.error("Invalid rankings structure")
            return False
        
        return True
    
    def save_session_data(self, 
                         results: Dict[str, Any],
                         topic_data: Dict[str, Any] = None,
                         input_data: Dict[str, Any] = None) -> str:
        """
        Save session data to JSON file
        
        Args:
            results: Results from ModelOrchestrator
            topic_data: Topic information (uid, name)
            input_data: Input prompt and query data
            
        Returns:
            Path to saved file
        """
        try:
            # Create session data structure
            session_data = {
                "session_id": self._get_session_id(),
                "timestamp": self._get_timestamp(),
                "topic": topic_data or {},
                "input": input_data or {},
                "models_used": self._extract_models_used(results),
                "outputs": self._extract_outputs(results),
                "rankings": self._extract_rankings(results),
                "pipeline_metadata": self._extract_metadata(results)
            }
            
            # Validate session data
            if not self._validate_session_data(session_data):
                logger.error("Session data validation failed")
                return ""
            
            # Get file path
            daily_folder = self._get_daily_folder()
            filename = self._get_filename()
            file_path = daily_folder / filename
            
            # Save to JSON with error handling
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Training data saved to: {file_path}")
                return str(file_path)
            except Exception as write_error:
                logger.error(f"Failed to write session data: {write_error}")
                return ""
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return ""
    
    def _extract_models_used(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model information from results"""
        models_used = {
            "email_models": [],
            "checklist_model": "",
            "judge_model": ""
        }
        
        # Extract email models
        emails = results.get("emails", [])
        for email in emails:
            model_name = email.get("model_name", "")
            if model_name and model_name not in models_used["email_models"]:
                models_used["email_models"].append(model_name)
        
        # Extract checklist model
        checklist = results.get("checklist", {})
        models_used["checklist_model"] = checklist.get("model_name", "")
        
        # Extract judge model (from first evaluation)
        emails = results.get("emails", [])
        if emails and len(emails) > 0:
            evaluation = emails[0].get("evaluation", {})
            models_used["judge_model"] = evaluation.get("evaluated_by", "")
        
        return models_used
    
    def _extract_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract outputs from results"""
        outputs = {
            "emails": [],
            "checklist": {},
            "evaluations": []
        }
        
        # Extract emails
        emails = results.get("emails", [])
        for email in emails:
            email_data = {
                "model": email.get("model_name", ""),
                "content": email.get("email_content", ""),
                "generation_time": email.get("generation_time", 0),
                "success": email.get("success", False)
            }
            outputs["emails"].append(email_data)
        
        # Extract checklist
        checklist = results.get("checklist", {})
        if checklist:
            outputs["checklist"] = {
                "model": checklist.get("model_name", ""),
                "content": checklist.get("checklist", {}),
                "generation_time": checklist.get("generation_time", 0),
                "success": checklist.get("success", False)
            }
        
        # Extract evaluations
        for email in emails:
            evaluation = email.get("evaluation", {})
            if evaluation:
                eval_data = {
                    "model": evaluation.get("evaluated_by", ""),
                    "email_model": email.get("model_name", ""),
                    "score": email.get("overall_score", 0),
                    "detailed_scores": evaluation
                }
                outputs["evaluations"].append(eval_data)
        
        return outputs
    
    def _extract_rankings(self, results: Dict[str, Any]) -> list:
        """Extract rankings from results"""
        rankings = []
        
        emails = results.get("emails", [])
        for email in emails:
            ranking_data = {
                "model": email.get("model_name", ""),
                "rank": email.get("rank", 0),
                "score": email.get("overall_score", 0)
            }
            rankings.append(ranking_data)
        
        return rankings
    
    def _extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pipeline metadata"""
        return {
            "total_time": results.get("total_time", 0),
            "success": results.get("success", False),
            "errors": []  # Will be populated in later stages
        }