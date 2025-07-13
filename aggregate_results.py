#!/usr/bin/env python3
"""
Results Aggregation Script for Checklist Mode Comparison

This script aggregates results from batch mode comparison jobs and creates
comprehensive reports for dissertation analysis.

Usage:
    python aggregate_results.py --job_id <SLURM_JOB_ID>
    python aggregate_results.py --date 2025-07-13
"""

import json
import csv
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsAggregator:
    """Aggregates results from multiple checklist mode comparison runs"""
    
    def __init__(self, output_dir: str = "./log/aggregated_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def aggregate_batch_results(self, job_id: str = None, date: str = None) -> Dict[str, Any]:
        """Aggregate results from batch comparison job"""
        if job_id:
            results_pattern = f"log/batch_results_{job_id}/summary_task*.csv"
        elif date:
            # Look for training data by date
            results_pattern = f"output/training_data/{date}/*/*.json"
        else:
            # Look for most recent results
            today = datetime.now().strftime("%Y-%m-%d")
            results_pattern = f"output/training_data/{today}/*/*.json"
        
        logger.info(f"Looking for results with pattern: {results_pattern}")
        
        # Aggregate data
        aggregated_data = {
            "summary": {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "modes_tested": set(),
                "topics_tested": set()
            },
            "mode_performance": {},
            "topic_performance": {},
            "detailed_results": []
        }
        
        if job_id:
            # Aggregate from batch job CSV files
            csv_files = glob.glob(results_pattern)
            for csv_file in csv_files:
                self._process_csv_file(csv_file, aggregated_data)
        else:
            # Aggregate from training data JSON files
            json_files = glob.glob(results_pattern)
            for json_file in json_files:
                self._process_json_file(json_file, aggregated_data)
        
        # Convert sets to lists for JSON serialization
        aggregated_data["summary"]["modes_tested"] = list(aggregated_data["summary"]["modes_tested"])
        aggregated_data["summary"]["topics_tested"] = list(aggregated_data["summary"]["topics_tested"])
        
        return aggregated_data
    
    def _process_csv_file(self, csv_file: str, aggregated_data: Dict[str, Any]) -> None:
        """Process a single CSV summary file"""
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    topic_uid = row['topic_uid']
                    mode = row['mode']
                    status = row['status']
                    runtime = float(row['runtime_seconds'])
                    
                    # Update summary
                    aggregated_data["summary"]["total_runs"] += 1
                    if status == "SUCCESS":
                        aggregated_data["summary"]["successful_runs"] += 1
                    else:
                        aggregated_data["summary"]["failed_runs"] += 1
                    
                    aggregated_data["summary"]["modes_tested"].add(mode)
                    aggregated_data["summary"]["topics_tested"].add(topic_uid)
                    
                    # Update mode performance
                    if mode not in aggregated_data["mode_performance"]:
                        aggregated_data["mode_performance"][mode] = {
                            "total_runs": 0,
                            "successful_runs": 0,
                            "failed_runs": 0,
                            "avg_runtime": 0,
                            "runtimes": []
                        }
                    
                    mode_data = aggregated_data["mode_performance"][mode]
                    mode_data["total_runs"] += 1
                    mode_data["runtimes"].append(runtime)
                    
                    if status == "SUCCESS":
                        mode_data["successful_runs"] += 1
                    else:
                        mode_data["failed_runs"] += 1
                    
                    # Update topic performance
                    if topic_uid not in aggregated_data["topic_performance"]:
                        aggregated_data["topic_performance"][topic_uid] = {
                            "modes_tested": {},
                            "success_rate": 0
                        }
                    
                    aggregated_data["topic_performance"][topic_uid]["modes_tested"][mode] = {
                        "status": status,
                        "runtime": runtime
                    }
                    
                    # Add to detailed results
                    aggregated_data["detailed_results"].append({
                        "topic_uid": topic_uid,
                        "mode": mode,
                        "status": status,
                        "runtime_seconds": runtime,
                        "source": "batch_job"
                    })
                    
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_file}: {e}")
    
    def _process_json_file(self, json_file: str, aggregated_data: Dict[str, Any]) -> None:
        """Process a single training data JSON file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract information
            mode = data.get("checklist_mode", "unknown")
            topic_name = data.get("topic", {}).get("name", "unknown")
            topic_uid = data.get("topic", {}).get("uid", "unknown")
            success = data.get("pipeline_metadata", {}).get("success", False)
            total_time = data.get("pipeline_metadata", {}).get("total_time", 0)
            
            # Update summary
            aggregated_data["summary"]["total_runs"] += 1
            if success:
                aggregated_data["summary"]["successful_runs"] += 1
            else:
                aggregated_data["summary"]["failed_runs"] += 1
            
            aggregated_data["summary"]["modes_tested"].add(mode)
            if topic_uid != "unknown":
                aggregated_data["summary"]["topics_tested"].add(topic_uid)
            
            # Update mode performance
            if mode not in aggregated_data["mode_performance"]:
                aggregated_data["mode_performance"][mode] = {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "avg_runtime": 0,
                    "runtimes": []
                }
            
            mode_data = aggregated_data["mode_performance"][mode]
            mode_data["total_runs"] += 1
            mode_data["runtimes"].append(total_time)
            
            if success:
                mode_data["successful_runs"] += 1
            else:
                mode_data["failed_runs"] += 1
            
            # Add to detailed results
            aggregated_data["detailed_results"].append({
                "topic_uid": topic_uid,
                "topic_name": topic_name,
                "mode": mode,
                "status": "SUCCESS" if success else "FAILED",
                "runtime_seconds": total_time,
                "source": "training_data",
                "file_path": json_file
            })
            
        except Exception as e:
            logger.error(f"Error processing JSON file {json_file}: {e}")
    
    def calculate_statistics(self, aggregated_data: Dict[str, Any]) -> None:
        """Calculate additional statistics"""
        # Calculate average runtimes for each mode
        for mode, data in aggregated_data["mode_performance"].items():
            if data["runtimes"]:
                data["avg_runtime"] = sum(data["runtimes"]) / len(data["runtimes"])
                data["min_runtime"] = min(data["runtimes"])
                data["max_runtime"] = max(data["runtimes"])
                data["success_rate"] = data["successful_runs"] / data["total_runs"] if data["total_runs"] > 0 else 0
        
        # Calculate success rates for topics
        for topic_uid, data in aggregated_data["topic_performance"].items():
            total_modes = len(data["modes_tested"])
            successful_modes = sum(1 for mode_data in data["modes_tested"].values() 
                                 if mode_data["status"] == "SUCCESS")
            data["success_rate"] = successful_modes / total_modes if total_modes > 0 else 0
    
    def save_results(self, aggregated_data: Dict[str, Any], suffix: str = "") -> str:
        """Save aggregated results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"aggregated_results_{timestamp}{suffix}"
        
        # Calculate statistics
        self.calculate_statistics(aggregated_data)
        
        # Save JSON file
        json_file = self.output_dir / f"{base_filename}.json"
        with open(json_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        
        # Save CSV summary
        csv_file = self.output_dir / f"{base_filename}_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_runs", aggregated_data["summary"]["total_runs"]])
            writer.writerow(["successful_runs", aggregated_data["summary"]["successful_runs"]])
            writer.writerow(["failed_runs", aggregated_data["summary"]["failed_runs"]])
            writer.writerow(["success_rate", 
                           aggregated_data["summary"]["successful_runs"] / 
                           aggregated_data["summary"]["total_runs"] 
                           if aggregated_data["summary"]["total_runs"] > 0 else 0])
        
        # Save detailed results CSV
        detailed_csv = self.output_dir / f"{base_filename}_detailed.csv"
        if aggregated_data["detailed_results"]:
            with open(detailed_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=aggregated_data["detailed_results"][0].keys())
                writer.writeheader()
                writer.writerows(aggregated_data["detailed_results"])
        
        logger.info(f"Results saved to:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV Summary: {csv_file}")
        logger.info(f"  Detailed CSV: {detailed_csv}")
        
        return str(json_file)

def main():
    parser = argparse.ArgumentParser(description="Aggregate checklist mode comparison results")
    parser.add_argument("--job_id", help="SLURM job ID to aggregate results from")
    parser.add_argument("--date", help="Date to aggregate training data from (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default="./log/aggregated_results", 
                       help="Output directory for aggregated results")
    
    args = parser.parse_args()
    
    aggregator = ResultsAggregator(args.output_dir)
    
    logger.info("Starting results aggregation...")
    aggregated_data = aggregator.aggregate_batch_results(args.job_id, args.date)
    
    suffix = f"_job{args.job_id}" if args.job_id else f"_date{args.date}" if args.date else ""
    result_file = aggregator.save_results(aggregated_data, suffix)
    
    # Print summary
    summary = aggregated_data["summary"]
    print(f"\n=== Aggregation Summary ===")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Successful runs: {summary['successful_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    print(f"Success rate: {summary['successful_runs']/summary['total_runs']*100:.1f}%" if summary['total_runs'] > 0 else "N/A")
    print(f"Modes tested: {', '.join(summary['modes_tested'])}")
    print(f"Topics tested: {len(summary['topics_tested'])}")
    
    print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    main()