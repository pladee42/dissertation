"""
Export for Training Utility

Simple utility to export training data in common formats for ML training.
"""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import argparse
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(input_path: str) -> List[Dict[str, Any]]:
    """Load training data from file or directory"""
    sessions = []
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if "sessions" in data:  # Merged file
            sessions = data["sessions"]
        else:  # Single session file
            sessions = [data]
            
    elif input_path.is_dir():
        # Directory - load all session files
        session_files = glob.glob(str(input_path / "**" / "session_*.json"), recursive=True)
        for file_path in sorted(session_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    logger.info(f"Loaded {len(sessions)} sessions for export")
    return sessions

def export_csv_format(sessions: List[Dict[str, Any]], output_path: str):
    """Export as CSV format"""
    csv_data = []
    
    for session in sessions:
        session_id = session.get("session_id", "")
        topic_uid = session.get("topic", {}).get("uid", "")
        topic_name = session.get("topic", {}).get("name", "")
        prompt = session.get("input", {}).get("prompt", "")
        user_query = session.get("input", {}).get("user_query", "")
        
        # Extract each email output
        emails = session.get("outputs", {}).get("emails", [])
        for email in emails:
            csv_data.append({
                "session_id": session_id,
                "topic_uid": topic_uid, 
                "topic_name": topic_name,
                "prompt": prompt,
                "user_query": user_query,
                "model": email.get("model", ""),
                "email_content": email.get("content", ""),
                "generation_time": email.get("generation_time", 0),
                "success": email.get("success", False)
            })
    
    # Write CSV
    if csv_data:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        
        logger.info(f"Exported {len(csv_data)} email records to CSV: {output_path}")
    else:
        logger.warning("No data to export to CSV")

def export_huggingface_format(sessions: List[Dict[str, Any]], output_path: str):
    """Export in HuggingFace datasets format"""
    hf_data = []
    
    for session in sessions:
        prompt = session.get("input", {}).get("prompt", "")
        topic_name = session.get("topic", {}).get("name", "")
        
        # Create input text
        input_text = f"Topic: {topic_name}\nPrompt: {prompt}"
        
        # Extract each email output
        emails = session.get("outputs", {}).get("emails", [])
        for email in emails:
            if email.get("success", False):
                hf_data.append({
                    "input": input_text,
                    "output": email.get("content", ""),
                    "model": email.get("model", ""),
                    "topic_uid": session.get("topic", {}).get("uid", ""),
                    "generation_time": email.get("generation_time", 0)
                })
    
    # Write JSON format compatible with HuggingFace
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hf_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(hf_data)} input-output pairs for HuggingFace: {output_path}")

def export_comparison_format(sessions: List[Dict[str, Any]], output_path: str):
    """Export in model comparison format"""
    comparison_data = []
    
    for session in sessions:
        if len(session.get("outputs", {}).get("emails", [])) < 2:
            continue  # Skip sessions with less than 2 models
            
        emails = session.get("outputs", {}).get("emails", [])
        rankings = session.get("rankings", [])
        
        # Create ranking map
        rank_map = {r.get("model", ""): r.get("rank", 0) for r in rankings}
        
        session_comparison = {
            "session_id": session.get("session_id", ""),
            "topic": session.get("topic", {}),
            "input": session.get("input", {}),
            "emails": []
        }
        
        for email in emails:
            model = email.get("model", "")
            session_comparison["emails"].append({
                "model": model,
                "content": email.get("content", ""),
                "rank": rank_map.get(model, 0),
                "score": email.get("overall_score", 0),
                "success": email.get("success", False)
            })
        
        comparison_data.append(session_comparison)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported {len(comparison_data)} comparison sessions: {output_path}")

def export_training_data(input_path: str, output_dir: str, formats: List[str] = None):
    """Export training data in specified formats"""
    if formats is None:
        formats = ["csv", "huggingface", "comparison"]
    
    # Load data
    sessions = load_training_data(input_path)
    if not sessions:
        logger.error("No training data found to export")
        return False
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export in each format
    for format_type in formats:
        if format_type == "csv":
            output_path = output_dir / f"training_data_{timestamp}.csv"
            export_csv_format(sessions, str(output_path))
            
        elif format_type == "huggingface":
            output_path = output_dir / f"training_data_hf_{timestamp}.json"
            export_huggingface_format(sessions, str(output_path))
            
        elif format_type == "comparison":
            output_path = output_dir / f"training_data_comparison_{timestamp}.json"
            export_comparison_format(sessions, str(output_path))
            
        else:
            logger.warning(f"Unknown format: {format_type}")
    
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Export training data for ML training")
    parser.add_argument("input_path", type=str, 
                       help="Input path (file or directory with training data)")
    parser.add_argument("--output_dir", type=str, 
                       default="./output/exported_training_data",
                       help="Output directory for exported files")
    parser.add_argument("--formats", nargs='+', 
                       choices=["csv", "huggingface", "comparison"],
                       default=["csv", "huggingface", "comparison"],
                       help="Export formats")
    
    args = parser.parse_args()
    
    logger.info("=== Training Data Exporter ===")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Formats: {args.formats}")
    
    success = export_training_data(args.input_path, args.output_dir, args.formats)
    
    if success:
        print("✅ Training data exported successfully!")
        print(f"Check output directory: {args.output_dir}")
        return 0
    else:
        print("❌ Failed to export training data")
        return 1

if __name__ == "__main__":
    exit(main())