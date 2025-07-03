"""
Merge Daily Files Utility

Simple utility to combine a day's training data sessions into a single file.
"""

import json
import os
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_session_files(daily_folder: str) -> List[Dict[str, Any]]:
    """Load all session files from a daily folder"""
    session_files = glob.glob(f"{daily_folder}/session_*.json")
    sessions = []
    
    for file_path in sorted(session_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                sessions.append(session_data)
                logger.info(f"Loaded session: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    return sessions

def create_merged_file(sessions: List[Dict[str, Any]], output_path: str):
    """Create merged file with all sessions"""
    merged_data = {
        "merged_timestamp": datetime.now().isoformat(),
        "total_sessions": len(sessions),
        "date_range": {
            "first_session": sessions[0]["timestamp"] if sessions else "",
            "last_session": sessions[-1]["timestamp"] if sessions else ""
        },
        "summary": {
            "total_emails_generated": sum(len(s.get("outputs", {}).get("emails", [])) for s in sessions),
            "total_topics_processed": len(set(s.get("topic", {}).get("uid", "") for s in sessions if s.get("topic", {}).get("uid"))),
            "models_used": list(set(
                model for s in sessions 
                for model in s.get("models_used", {}).get("email_models", [])
            )),
            "successful_sessions": len([s for s in sessions if s.get("pipeline_metadata", {}).get("success", False)])
        },
        "sessions": sessions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Merged {len(sessions)} sessions into: {output_path}")

def merge_daily_data(date: str = None, base_dir: str = "./output/training_data"):
    """Merge daily training data files"""
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    daily_folder = Path(base_dir) / date
    
    if not daily_folder.exists():
        logger.error(f"Daily folder not found: {daily_folder}")
        return False
    
    logger.info(f"Merging data for date: {date}")
    
    # Load all session files
    sessions = load_session_files(str(daily_folder))
    
    if not sessions:
        logger.warning(f"No session files found in {daily_folder}")
        return False
    
    # Create merged file
    output_filename = f"merged_{date}.json"
    output_path = daily_folder / output_filename
    
    create_merged_file(sessions, str(output_path))
    
    logger.info(f"Summary:")
    logger.info(f"- Date: {date}")
    logger.info(f"- Sessions merged: {len(sessions)}")
    logger.info(f"- Output file: {output_path}")
    
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Merge daily training data files")
    parser.add_argument("--date", type=str, 
                       help="Date to merge (YYYY-MM-DD format). Default: today")
    parser.add_argument("--base_dir", type=str, 
                       default="./output/training_data",
                       help="Base directory for training data")
    
    args = parser.parse_args()
    
    logger.info("=== Daily Files Merger ===")
    
    success = merge_daily_data(args.date, args.base_dir)
    
    if success:
        print("✅ Daily files merged successfully!")
        return 0
    else:
        print("❌ Failed to merge daily files")
        return 1

if __name__ == "__main__":
    exit(main())