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
from datetime import datetime, time, timedelta
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
        
    def find_session_files(self, start_date: str, mode: str = None, from_time: str = None) -> List[Path]:
        """Find all session files from start_date onward, optionally filtered by mode"""
        session_files = []
        
        # If from_time is specified, we need to search across multiple dates
        if from_time:
            # Find all available dates from start_date onward
            max_days = getattr(self, '_max_days', 30)  # Use instance variable if set
            available_dates = self.get_available_dates_from(start_date, max_days)
            logger.info(f"Searching across dates: {available_dates}")
            
            for date_str in available_dates:
                date_files = self.find_session_files_for_date(date_str, mode)
                session_files.extend(date_files)
        else:
            # Original behavior: only search specified date
            session_files = self.find_session_files_for_date(start_date, mode)
        
        return sorted(session_files)
    
    def find_session_files_for_date(self, date: str, mode: str = None) -> List[Path]:
        """Find session files for a specific date"""
        date_dir = self.training_data_dir / date
        
        if not date_dir.exists():
            logger.debug(f"Date directory not found: {date_dir}")
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
        
        return session_files
    
    def get_available_dates_from(self, start_date: str, max_days: int = 30) -> List[str]:
        """Get list of available date directories from start_date onward"""
        available_dates = []
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
        
        # Search for up to max_days from start_date
        for i in range(max_days):
            check_date = start_date_obj + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            date_dir = self.training_data_dir / date_str
            
            if date_dir.exists():
                available_dates.append(date_str)
                logger.debug(f"Found data for date: {date_str}")
            else:
                # If we find 3 consecutive missing dates, stop searching
                if i > 0 and all(
                    not (self.training_data_dir / (start_date_obj + timedelta(days=j)).strftime("%Y-%m-%d")).exists()
                    for j in range(max(0, i-2), i+1)
                ):
                    logger.debug(f"Stopping search at {date_str} (consecutive missing dates)")
                    break
        
        return available_dates
    
    def parse_time_string(self, time_str: str) -> time:
        """Parse time string in HH:MM or HH:MM:SS format"""
        try:
            if len(time_str) == 5:  # HH:MM format
                return datetime.strptime(time_str, "%H:%M").time()
            elif len(time_str) == 8:  # HH:MM:SS format
                return datetime.strptime(time_str, "%H:%M:%S").time()
            else:
                raise ValueError(f"Invalid time format: {time_str}. Use HH:MM or HH:MM:SS")
        except ValueError as e:
            logger.error(f"Failed to parse time string '{time_str}': {e}")
            raise
    
    def filter_by_time(self, session_files: List[Path], start_date: str, 
                      from_time: str = None, to_time: str = None) -> List[Path]:
        """Filter session files by time range, supporting cross-date filtering"""
        if not from_time:
            return session_files
        
        logger.info(f"Filtering sessions from {start_date} {from_time}" + (f" to {to_time}" if to_time else " onward"))
        
        try:
            # Parse target times
            from_time_obj = self.parse_time_string(from_time)
            to_time_obj = self.parse_time_string(to_time) if to_time else None
            
            # Create target datetimes
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d").date()
            from_datetime = datetime.combine(start_date_obj, from_time_obj)
            
            # For to_time, if specified, it should be on the same start date unless we add date support
            to_datetime = datetime.combine(start_date_obj, to_time_obj) if to_time_obj else None
            
            filtered_files = []
            for session_file in session_files:
                # Try to get timestamp from session content first
                session_data = self.load_session_data(session_file)
                session_datetime = None
                
                if session_data and "timestamp" in session_data:
                    try:
                        # Parse ISO timestamp from session data
                        timestamp_str = session_data["timestamp"]
                        # Handle different ISO formats
                        if timestamp_str.endswith('Z'):
                            timestamp_str = timestamp_str[:-1] + '+00:00'
                        session_datetime = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Convert to local time (remove timezone info for comparison)
                        session_datetime = session_datetime.replace(tzinfo=None)
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamp from {session_file}: {e}")
                
                # Fallback: extract time from filename
                if not session_datetime:
                    try:
                        # Extract timestamp from filename: session_20250715_033000_preprocess.json
                        filename = session_file.name
                        if filename.startswith("session_"):
                            parts = filename.split("_")
                            if len(parts) >= 3:
                                date_part = parts[1]  # 20250715
                                time_part = parts[2]  # 033000
                                datetime_str = f"{date_part}_{time_part}"
                                session_datetime = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamp from filename {session_file}: {e}")
                        continue
                
                if not session_datetime:
                    logger.warning(f"Could not determine timestamp for {session_file}, skipping")
                    continue
                
                # Apply time filtering
                if session_datetime >= from_datetime:
                    if not to_datetime or session_datetime <= to_datetime:
                        filtered_files.append(session_file)
                        logger.debug(f"Including session from {session_datetime}")
                    else:
                        logger.debug(f"Excluding session from {session_datetime} (after end time)")
                else:
                    logger.debug(f"Excluding session from {session_datetime} (before start time)")
            
            logger.info(f"Filtered to {len(filtered_files)} sessions within time range")
            return filtered_files
            
        except Exception as e:
            logger.error(f"Error filtering by time: {e}")
            return session_files  # Return all files if filtering fails
    
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
                               output_timestamp: str = None, from_time: str = None,
                               to_time: str = None, max_days: int = 30) -> Optional[str]:
        """Recover complete_results.json from session data"""
        
        logger.info(f"Starting recovery for date: {date}, mode: {mode}")
        if from_time:
            logger.info(f"Time filtering: from {from_time}" + (f" to {to_time}" if to_time else " onward"))
            logger.info(f"Will search up to {max_days} days from {date}")
        
        # Set max_days for use in find_session_files
        self._max_days = max_days
        
        # Find session files (now supports cross-date search when from_time is specified)
        session_files = self.find_session_files(date, mode, from_time)
        if not session_files:
            logger.error("No session files found")
            return None
        
        logger.info(f"Found {len(session_files)} session files")
        
        # Apply time filtering if specified
        if from_time:
            session_files = self.filter_by_time(session_files, date, from_time, to_time)
            if not session_files:
                logger.error("No session files found within specified time range")
                return None
        
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
            'summary': self.generate_summary(successful_results)
        }
        
        # Only add separate arrays if there are actually failed results
        if failed_results:
            complete_results['successful_results'] = successful_results
            complete_results['failed_results'] = failed_results
        
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
    parser.add_argument("--from_time", type=str,
                       help="Time to start recovery from (HH:MM or HH:MM:SS format). When specified, searches across multiple dates from --date onward")
    parser.add_argument("--to_time", type=str,
                       help="Time to end recovery at (HH:MM or HH:MM:SS format, optional). Currently only supports same-day filtering")
    parser.add_argument("--max_days", type=int, default=30,
                       help="Maximum number of days to search when using --from_time (default: 30)")
    parser.add_argument("--output_timestamp", type=str,
                       help="Custom timestamp for output directory (optional)")
    parser.add_argument("--base_dir", type=str, default="./output",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    # Validate time arguments
    if args.to_time and not args.from_time:
        parser.error("--to_time requires --from_time to be specified")
    
    # Create recovery instance
    recovery = CompleteResultsRecovery(args.base_dir)
    
    # Perform recovery
    result_file = recovery.recover_complete_results(
        date=args.date,
        mode=args.mode,
        output_timestamp=args.output_timestamp,
        from_time=args.from_time,
        to_time=args.to_time,
        max_days=args.max_days
    )
    
    if result_file:
        print(f"✅ Recovery successful! File saved to: {result_file}")
        if args.from_time:
            print(f"📅 Recovered sessions from {args.date} {args.from_time}" + 
                  (f" to {args.to_time}" if args.to_time else " onward"))
        return 0
    else:
        print("❌ Recovery failed!")
        return 1


if __name__ == "__main__":
    exit(main())