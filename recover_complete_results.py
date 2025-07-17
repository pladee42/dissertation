#!/usr/bin/env python3
"""
Manual Recovery Script for complete_results.json

This script reconstructs complete_results.json from individual session data
when the main HPC pipeline fails to create the aggregated results file.
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteResultsRecovery:
    """Recovers complete_results.json from session data"""
    
    def __init__(self, base_dir: str = "./output"):
        self.base_dir = Path(base_dir)
        self.training_data_dir = self.base_dir / "training_data"
        self.multi_topic_dir = self.base_dir / "multi_topic_results"
        
    def find_session_files(self, date: str, mode: str = None) -> List[Path]:
        """Find all session files for a given date and optional mode"""
        date_dir = self.training_data_dir / date
        
        if not date_dir.exists():
            logger.error(f"Date directory not found: {date_dir}")
            return []
        
        session_files = []
        
        if mode:
            # Specific mode directory
            mode_dir = date_dir / mode
            if mode_dir.exists():
                pattern = f"session_*_{mode}.json"
                session_files.extend(mode_dir.glob(pattern))
        else:
            # All modes
            for mode_dir in date_dir.iterdir():
                if mode_dir.is_dir():
                    pattern = f"session_*.json"
                    session_files.extend(mode_dir.glob(pattern))
        
        return sorted(session_files)
    
    def load_session_data(self, session_file: Path) -> Optional[Dict[str, Any]]:
        """Load and validate session data"""
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            required_fields = ["session_id", "timestamp", "topic", "outputs"]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing field {field} in {session_file}")
                    return None
            
            return data
        except Exception as e:
            logger.error(f"Failed to load session data from {session_file}: {e}")
            return None
    
    def convert_session_to_topic_result(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert session data to topic result format"""
        try:
            # Extract basic info
            topic_info = session_data.get("topic", {})
            outputs = session_data.get("outputs", {})
            
            # Build emails array
            emails = []
            email_outputs = outputs.get("emails", [])
            evaluation_outputs = outputs.get("evaluations", [])
            
            # Create evaluation lookup by model
            eval_lookup = {}
            for eval_data in evaluation_outputs:
                model = eval_data.get("email_model", "")
                eval_lookup[model] = eval_data
            
            # Build email entries with evaluations
            for email_data in email_outputs:
                model_name = email_data.get("model", "")
                
                email_entry = {
                    "model_name": model_name,
                    "model_id": model_name,  # Approximate
                    "email_content": email_data.get("content", ""),
                    "generation_time": email_data.get("generation_time", 0),
                    "success": email_data.get("success", False)
                }
                
                # Add evaluation if available
                if model_name in eval_lookup:
                    eval_data = eval_lookup[model_name]
                    email_entry["evaluation"] = {
                        "checklist_scores": eval_data.get("binary_results", []),
                        "strengths": eval_data.get("strengths", ""),
                        "weaknesses": eval_data.get("weaknesses", ""),
                        "consistency_confidence": eval_data.get("consistency_confidence", 0.0),
                        "average_response_time": eval_data.get("average_response_time", 0.0),
                        "generation_attempts": eval_data.get("generation_attempts", 1),
                        "total_criteria": eval_data.get("total_criteria", 0),
                        "weighted_score": eval_data.get("weighted_score", 0),
                        "priority_breakdown": eval_data.get("priority_breakdown", {}),
                        "evaluated_by": eval_data.get("model", "")
                    }
                    
                    email_entry["overall_score"] = eval_data.get("ranking_score", 0)
                    email_entry["rank"] = eval_data.get("rank", 0)
                
                emails.append(email_entry)
            
            # Sort emails by rank if available
            emails.sort(key=lambda x: x.get("rank", 999))
            
            # Find best email (rank 1 or highest score)
            best_email = None
            if emails:
                # Try to find rank 1 first
                for email in emails:
                    if email.get("rank") == 1:
                        best_email = email
                        break
                
                # If no rank 1, take highest scoring
                if not best_email:
                    best_email = max(emails, key=lambda x: x.get("overall_score", 0))
            
            # Build checklist info
            checklist_output = outputs.get("checklist", {})
            checklist = {
                "model_name": checklist_output.get("model", ""),
                "checklist": checklist_output.get("content", {}),
                "generation_time": checklist_output.get("generation_time", 0),
                "success": checklist_output.get("success", False)
            }
            
            # Calculate processing time (approximate)
            processing_time = 0
            for email in emails:
                processing_time += email.get("generation_time", 0)
            processing_time += checklist.get("generation_time", 0)
            
            # Build result
            result = {
                "emails": emails,
                "checklist": checklist,
                "topic_uid": topic_info.get("uid", ""),
                "topic_name": topic_info.get("name", ""),
                "processing_time": processing_time,
                "success": len(emails) > 0 and checklist.get("success", False)
            }
            
            if best_email:
                result["best_email"] = best_email
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to convert session data: {e}")
            return None
    
    def generate_summary(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not successful_results:
            return {}
        
        try:
            # Calculate average processing time
            processing_times = [r.get('processing_time', 0) for r in successful_results]
            avg_processing_time = sum(processing_times) / len(processing_times)
            
            # Count total emails generated
            total_emails = sum(len(r.get('emails', [])) for r in successful_results)
            
            # Find best performing models across topics
            model_scores = {}
            for result in successful_results:
                best_email = result.get('best_email', {})
                model_name = best_email.get('model_name', 'unknown')
                score = best_email.get('overall_score', 0)
                
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(score)
            
            # Calculate average scores per model
            model_avg_scores = {}
            for model, scores in model_scores.items():
                model_avg_scores[model] = sum(scores) / len(scores) if scores else 0
            
            return {
                'avg_processing_time': avg_processing_time,
                'total_emails_generated': total_emails,
                'model_performance': model_avg_scores,
                'best_overall_model': max(model_avg_scores.items(), key=lambda x: x[1])[0] if model_avg_scores else None
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    def recover_complete_results(self, date: str, mode: str = None, 
                               output_timestamp: str = None) -> Optional[str]:
        """Recover complete_results.json from session data"""
        
        logger.info(f"Starting recovery for date: {date}, mode: {mode}")
        
        # Find session files
        session_files = self.find_session_files(date, mode)
        if not session_files:
            logger.error("No session files found")
            return None
        
        logger.info(f"Found {len(session_files)} session files")
        
        # Process session files
        all_results = []
        successful_results = []
        failed_results = []
        total_time = 0
        
        for session_file in session_files:
            logger.info(f"Processing: {session_file}")
            
            session_data = self.load_session_data(session_file)
            if not session_data:
                continue
            
            topic_result = self.convert_session_to_topic_result(session_data)
            if not topic_result:
                continue
            
            all_results.append(topic_result)
            total_time += topic_result.get('processing_time', 0)
            
            if topic_result.get('success', False):
                successful_results.append(topic_result)
            else:
                failed_results.append(topic_result)
        
        if not all_results:
            logger.error("No valid results processed")
            return None
        
        # Build complete results structure
        complete_results = {
            'success': len(failed_results) == 0,
            'total_topics': len(all_results),
            'successful_topics': len(successful_results),
            'failed_topics': len(failed_results),
            'total_time': total_time,
            'results': all_results,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'summary': self.generate_summary(successful_results)
        }
        
        # Save to output directory
        if not output_timestamp:
            output_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_dir = self.multi_topic_dir / output_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "complete_results.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully recovered complete_results.json to: {results_file}")
            
            # Also create CSV summary
            self.create_csv_summary(complete_results, output_dir)
            
            return str(results_file)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def create_csv_summary(self, results: Dict[str, Any], output_dir: Path):
        """Create CSV summary file"""
        try:
            import csv
            
            csv_file = output_dir / "topic_summary.csv"
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['topic_uid', 'topic_name', 'best_model', 'best_score', 'processing_time'])
                writer.writeheader()
                
                for result in results.get('successful_results', []):
                    best_email = result.get('best_email', {})
                    writer.writerow({
                        'topic_uid': result.get('topic_uid', ''),
                        'topic_name': result.get('topic_name', ''),
                        'best_model': best_email.get('model_name', ''),
                        'best_score': best_email.get('overall_score', 0),
                        'processing_time': result.get('processing_time', 0)
                    })
            
            logger.info(f"CSV summary saved to: {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to create CSV summary: {e}")


def main():
    parser = argparse.ArgumentParser(description="Recover complete_results.json from session data")
    
    parser.add_argument("--date", type=str, required=True,
                       help="Date to recover from (YYYY-MM-DD format)")
    parser.add_argument("--mode", type=str, choices=['enhanced', 'extract_only', 'preprocess'],
                       help="Specific checklist mode to recover (optional)")
    parser.add_argument("--output_timestamp", type=str,
                       help="Custom timestamp for output directory (optional)")
    parser.add_argument("--base_dir", type=str, default="./output",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    # Create recovery instance
    recovery = CompleteResultsRecovery(args.base_dir)
    
    # Perform recovery
    result_file = recovery.recover_complete_results(
        date=args.date,
        mode=args.mode,
        output_timestamp=args.output_timestamp
    )
    
    if result_file:
        print(f"✅ Recovery successful! File saved to: {result_file}")
        return 0
    else:
        print("❌ Recovery failed!")
        return 1


if __name__ == "__main__":
    exit(main())