#!/usr/bin/env python3
"""
Automated File Existence and Quality Checks
Context preservation verification system for Results section implementation
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

def check_file_existence(context_dir: str = "/Users/tan_waris/Library/CloudStorage/GoogleDrive-wratthapoom1@sheffield.ac.uk/My Drive/Dissertation/dissertation/analysis/results_context") -> Dict:
    """
    Check existence of all expected files based on master context tracker
    
    Args:
        context_dir: Path to results_context directory
    
    Returns:
        Dictionary with file existence status
    """
    context_path = Path(context_dir)
    
    # Load master context tracker
    tracker_file = context_path / "registries" / "master_context_tracker.json"
    if not tracker_file.exists():
        return {"error": "Master context tracker not found"}
    
    with open(tracker_file, 'r') as f:
        tracker = json.load(f)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "CHECKING",
        "stage_results": {},
        "missing_files": [],
        "existing_files": [],
        "total_expected": 0,
        "total_existing": 0
    }
    
    # Check each stage
    for stage_key, stage_info in tracker.items():
        if stage_key == "overall_progress":
            continue
            
        if stage_info["status"] != "not_started":
            stage_results = {
                "stage": stage_key,
                "expected_files": stage_info.get("expected_files", []),
                "missing": [],
                "existing": [],
                "status": "PASS"
            }
            
            for expected_file in stage_info.get("expected_files", []):
                file_path = context_path / expected_file
                results["total_expected"] += 1
                
                if file_path.exists():
                    stage_results["existing"].append(expected_file)
                    results["existing_files"].append(expected_file)
                    results["total_existing"] += 1
                else:
                    stage_results["missing"].append(expected_file)
                    results["missing_files"].append(expected_file)
                    stage_results["status"] = "FAIL"
            
            results["stage_results"][stage_key] = stage_results
    
    # Overall status
    if results["missing_files"]:
        results["overall_status"] = "FAIL"
    else:
        results["overall_status"] = "PASS"
    
    results["completion_rate"] = (results["total_existing"] / results["total_expected"]) * 100 if results["total_expected"] > 0 else 0
    
    return results

def check_file_sizes(context_dir: str) -> Dict:
    """
    Check file sizes to identify potentially corrupted or empty files
    
    Args:
        context_dir: Path to results_context directory
    
    Returns:
        Dictionary with file size analysis
    """
    context_path = Path(context_dir)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "file_sizes": {},
        "suspicious_files": [],
        "status": "PASS"
    }
    
    # Expected minimum sizes (in bytes)
    min_sizes = {
        ".json": 50,     # JSON files should have some content
        ".md": 100,      # Markdown files should have meaningful content
        ".png": 1000,    # PNG files should be reasonably sized
        ".tex": 50       # LaTeX files should have some content
    }
    
    # Check all files in context directory
    for file_path in context_path.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            extension = file_path.suffix.lower()
            
            relative_path = str(file_path.relative_to(context_path))
            results["file_sizes"][relative_path] = size
            
            # Check if file is suspiciously small
            min_expected = min_sizes.get(extension, 0)
            if size < min_expected:
                results["suspicious_files"].append({
                    "file": relative_path,
                    "size": size,
                    "expected_min": min_expected,
                    "issue": "File too small"
                })
                results["status"] = "WARNING"
    
    return results

def check_json_validity(context_dir: str) -> Dict:
    """
    Verify that all JSON files are valid and parseable
    
    Args:
        context_dir: Path to results_context directory
    
    Returns:
        Dictionary with JSON validation results
    """
    context_path = Path(context_dir)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "json_files_checked": 0,
        "valid_json_files": 0,
        "invalid_json_files": [],
        "status": "PASS"
    }
    
    # Find all JSON files
    for json_file in context_path.rglob("*.json"):
        results["json_files_checked"] += 1
        
        try:
            with open(json_file, 'r') as f:
                json.load(f)
            results["valid_json_files"] += 1
        except json.JSONDecodeError as e:
            results["invalid_json_files"].append({
                "file": str(json_file.relative_to(context_path)),
                "error": str(e)
            })
            results["status"] = "FAIL"
        except Exception as e:
            results["invalid_json_files"].append({
                "file": str(json_file.relative_to(context_path)),
                "error": f"Unexpected error: {str(e)}"
            })
            results["status"] = "FAIL"
    
    return results

def generate_verification_report(context_dir: str) -> Dict:
    """
    Generate comprehensive verification report
    
    Args:
        context_dir: Path to results_context directory
    
    Returns:
        Complete verification report
    """
    print("Running comprehensive file verification...")
    
    # Run all checks
    existence_check = check_file_existence(context_dir)
    size_check = check_file_sizes(context_dir)
    json_check = check_json_validity(context_dir)
    
    # Combine results
    report = {
        "verification_timestamp": datetime.now().isoformat(),
        "overall_status": "PASS",
        "checks": {
            "file_existence": existence_check,
            "file_sizes": size_check,
            "json_validity": json_check
        },
        "summary": {
            "total_files_expected": existence_check["total_expected"],
            "total_files_existing": existence_check["total_existing"],
            "completion_rate": existence_check["completion_rate"],
            "issues_found": []
        }
    }
    
    # Determine overall status
    if existence_check["overall_status"] == "FAIL":
        report["overall_status"] = "FAIL"
        report["summary"]["issues_found"].append("Missing required files")
    
    if size_check["status"] == "WARNING":
        if report["overall_status"] == "PASS":
            report["overall_status"] = "WARNING"
        report["summary"]["issues_found"].append("Suspicious file sizes")
    
    if json_check["status"] == "FAIL":
        report["overall_status"] = "FAIL"
        report["summary"]["issues_found"].append("Invalid JSON files")
    
    # Save report
    report_path = Path(context_dir) / "verification" / "automated_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Verification report saved to: {report_path}")
    
    return report

def main():
    """Run automated file checks"""
    context_dir = "/Users/tan_waris/Library/CloudStorage/GoogleDrive-wratthapoom1@sheffield.ac.uk/My Drive/Dissertation/dissertation/analysis/results_context"
    
    print("=" * 60)
    print("AUTOMATED FILE VERIFICATION")
    print("=" * 60)
    
    report = generate_verification_report(context_dir)
    
    print(f"\nOverall Status: {report['overall_status']}")
    print(f"Files Expected: {report['summary']['total_files_expected']}")
    print(f"Files Existing: {report['summary']['total_files_existing']}")
    print(f"Completion Rate: {report['summary']['completion_rate']:.1f}%")
    
    if report['summary']['issues_found']:
        print("\nIssues Found:")
        for issue in report['summary']['issues_found']:
            print(f"  - {issue}")
    else:
        print("\nNo issues found - all checks passed!")

if __name__ == "__main__":
    main()