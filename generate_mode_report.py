#!/usr/bin/env python3
"""
Mode Comparison Report Generator

Generates comprehensive reports comparing checklist generation modes for dissertation analysis.

Usage:
    python generate_mode_report.py --input aggregated_results.json
    python generate_mode_report.py --date 2025-07-13
"""

import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModeComparisonReport:
    """Generates comprehensive mode comparison reports"""
    
    def __init__(self, output_dir: str = "./log/mode_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, data_source: str, is_file: bool = True) -> str:
        """Generate comprehensive mode comparison report"""
        if is_file:
            with open(data_source, 'r') as f:
                data = json.load(f)
        else:
            # Load from aggregated results by date
            from aggregate_results import ResultsAggregator
            aggregator = ResultsAggregator()
            data = aggregator.aggregate_batch_results(date=data_source)
            aggregator.calculate_statistics(data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"mode_comparison_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            self._write_report_header(f, data)
            self._write_executive_summary(f, data)
            self._write_mode_analysis(f, data)
            self._write_performance_metrics(f, data)
            self._write_detailed_findings(f, data)
            self._write_recommendations(f, data)
        
        # Generate supplementary files
        self._generate_csv_exports(data, timestamp)
        
        logger.info(f"Mode comparison report generated: {report_file}")
        return str(report_file)
    
    def _write_report_header(self, f, data: Dict[str, Any]) -> None:
        """Write report header"""
        f.write("# Checklist Generation Mode Comparison Report\\n\\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write("## Overview\\n\\n")
        f.write("This report compares the performance and effectiveness of three checklist generation modes:\\n")
        f.write("- **Enhanced Mode**: Full context with comprehensive analysis instructions\\n")
        f.write("- **Extract-Only Mode**: Minimal context focusing on example comparison\\n")
        f.write("- **Preprocess Mode**: Two-step structured analysis approach\\n\\n")
    
    def _write_executive_summary(self, f, data: Dict[str, Any]) -> None:
        """Write executive summary"""
        summary = data["summary"]
        f.write("## Executive Summary\\n\\n")
        f.write(f"- **Total Runs**: {summary['total_runs']}\\n")
        f.write(f"- **Success Rate**: {summary['successful_runs']/summary['total_runs']*100:.1f}%\\n" if summary['total_runs'] > 0 else "- **Success Rate**: N/A\\n")
        f.write(f"- **Modes Tested**: {', '.join(summary['modes_tested'])}\\n")
        f.write(f"- **Topics Covered**: {len(summary['topics_tested'])}\\n\\n")
        
        # Best performing mode
        if data["mode_performance"]:
            best_mode = max(data["mode_performance"].items(), 
                          key=lambda x: x[1].get("success_rate", 0))
            f.write(f"**Best Performing Mode**: {best_mode[0]} ({best_mode[1].get('success_rate', 0)*100:.1f}% success rate)\\n\\n")
    
    def _write_mode_analysis(self, f, data: Dict[str, Any]) -> None:
        """Write detailed mode analysis"""
        f.write("## Mode Performance Analysis\\n\\n")
        
        for mode, metrics in data["mode_performance"].items():
            f.write(f"### {mode.replace('_', ' ').title()} Mode\\n\\n")
            f.write(f"- **Total Runs**: {metrics['total_runs']}\\n")
            f.write(f"- **Success Rate**: {metrics.get('success_rate', 0)*100:.1f}%\\n")
            f.write(f"- **Average Runtime**: {metrics.get('avg_runtime', 0):.2f}s\\n")
            
            if 'min_runtime' in metrics:
                f.write(f"- **Runtime Range**: {metrics['min_runtime']:.2f}s - {metrics['max_runtime']:.2f}s\\n")
            
            f.write("\\n")
            
            # Mode-specific insights
            if mode == "enhanced":
                f.write("**Characteristics**: Full context approach with comprehensive analysis instructions.\\n")
                f.write("**Use Case**: Baseline comparison and comprehensive validation.\\n\\n")
            elif mode == "extract_only":
                f.write("**Characteristics**: Minimal context focusing on direct example comparison.\\n")
                f.write("**Use Case**: Efficient validation with reduced prompt complexity.\\n\\n")
            elif mode == "preprocess":
                f.write("**Characteristics**: Two-step structured analysis with characteristic extraction.\\n")
                f.write("**Use Case**: Systematic validation based on extracted email features.\\n\\n")
    
    def _write_performance_metrics(self, f, data: Dict[str, Any]) -> None:
        """Write performance metrics comparison"""
        f.write("## Performance Metrics Comparison\\n\\n")
        f.write("| Mode | Success Rate | Avg Runtime | Total Runs |\\n")
        f.write("|------|-------------|-------------|------------|\\n")
        
        for mode, metrics in data["mode_performance"].items():
            success_rate = metrics.get('success_rate', 0) * 100
            avg_runtime = metrics.get('avg_runtime', 0)
            total_runs = metrics['total_runs']
            f.write(f"| {mode.replace('_', ' ').title()} | {success_rate:.1f}% | {avg_runtime:.2f}s | {total_runs} |\\n")
        
        f.write("\\n")
    
    def _write_detailed_findings(self, f, data: Dict[str, Any]) -> None:
        """Write detailed findings"""
        f.write("## Detailed Findings\\n\\n")
        
        # Runtime analysis
        f.write("### Runtime Analysis\\n\\n")
        mode_runtimes = []
        for mode, metrics in data["mode_performance"].items():
            if 'avg_runtime' in metrics:
                mode_runtimes.append((mode, metrics['avg_runtime']))
        
        if mode_runtimes:
            mode_runtimes.sort(key=lambda x: x[1])
            f.write(f"**Fastest Mode**: {mode_runtimes[0][0]} ({mode_runtimes[0][1]:.2f}s average)\\n")
            f.write(f"**Slowest Mode**: {mode_runtimes[-1][0]} ({mode_runtimes[-1][1]:.2f}s average)\\n\\n")
        
        # Success rate analysis
        f.write("### Success Rate Analysis\\n\\n")
        mode_success = []
        for mode, metrics in data["mode_performance"].items():
            if 'success_rate' in metrics:
                mode_success.append((mode, metrics['success_rate']))
        
        if mode_success:
            mode_success.sort(key=lambda x: x[1], reverse=True)
            f.write(f"**Most Reliable Mode**: {mode_success[0][0]} ({mode_success[0][1]*100:.1f}% success rate)\\n")
            if len(mode_success) > 1:
                f.write(f"**Least Reliable Mode**: {mode_success[-1][0]} ({mode_success[-1][1]*100:.1f}% success rate)\\n")
        f.write("\\n")
        
        # Topic performance
        if data["topic_performance"]:
            f.write("### Topic Performance\\n\\n")
            f.write("Performance varies by topic complexity and content type:\\n\\n")
            for topic_uid, topic_data in data["topic_performance"].items():
                if topic_data["modes_tested"]:
                    success_count = sum(1 for mode_data in topic_data["modes_tested"].values() 
                                      if mode_data["status"] == "SUCCESS")
                    total_modes = len(topic_data["modes_tested"])
                    f.write(f"- **{topic_uid}**: {success_count}/{total_modes} modes successful\\n")
            f.write("\\n")
    
    def _write_recommendations(self, f, data: Dict[str, Any]) -> None:
        """Write recommendations"""
        f.write("## Recommendations\\n\\n")
        
        # Determine best mode for different use cases
        modes = data["mode_performance"]
        
        if len(modes) >= 3:
            # Find fastest and most reliable modes
            fastest_mode = min(modes.items(), key=lambda x: x[1].get('avg_runtime', float('inf')))
            most_reliable = max(modes.items(), key=lambda x: x[1].get('success_rate', 0))
            
            f.write("### Use Case Recommendations\\n\\n")
            f.write(f"**For Production Use**: {most_reliable[0]} mode (highest success rate: {most_reliable[1].get('success_rate', 0)*100:.1f}%)\\n\\n")
            f.write(f"**For High-Throughput Processing**: {fastest_mode[0]} mode (fastest runtime: {fastest_mode[1].get('avg_runtime', 0):.2f}s)\\n\\n")
            
            f.write("### Implementation Strategy\\n\\n")
            f.write("1. **Primary Mode**: Use the most reliable mode for critical evaluations\\n")
            f.write("2. **Fallback Strategy**: Implement cascade from complex to simple modes on failure\\n")
            f.write("3. **Performance Monitoring**: Track success rates and runtime metrics in production\\n\\n")
        
        f.write("### Future Improvements\\n\\n")
        f.write("- **Error Analysis**: Investigate failure patterns for mode-specific optimizations\\n")
        f.write("- **Hybrid Approach**: Consider combining strengths of different modes\\n")
        f.write("- **Dynamic Selection**: Implement context-aware mode selection based on topic characteristics\\n\\n")
    
    def _generate_csv_exports(self, data: Dict[str, Any], timestamp: str) -> None:
        """Generate CSV exports for further analysis"""
        # Mode performance CSV
        mode_csv = self.output_dir / f"mode_performance_{timestamp}.csv"
        mode_rows = []
        for mode, metrics in data["mode_performance"].items():
            mode_rows.append({
                "mode": mode,
                "total_runs": metrics["total_runs"],
                "successful_runs": metrics["successful_runs"],
                "failed_runs": metrics["failed_runs"],
                "success_rate": metrics.get("success_rate", 0),
                "avg_runtime": metrics.get("avg_runtime", 0),
                "min_runtime": metrics.get("min_runtime", 0),
                "max_runtime": metrics.get("max_runtime", 0)
            })
        
        if mode_rows:
            if HAS_PANDAS:
                pd.DataFrame(mode_rows).to_csv(mode_csv, index=False)
            else:
                # Fallback CSV writing without pandas
                import csv
                with open(mode_csv, 'w', newline='') as f:
                    if mode_rows:
                        writer = csv.DictWriter(f, fieldnames=mode_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(mode_rows)
            logger.info(f"Mode performance CSV: {mode_csv}")
        
        # Detailed results CSV
        if data["detailed_results"]:
            detailed_csv = self.output_dir / f"detailed_results_{timestamp}.csv"
            if HAS_PANDAS:
                pd.DataFrame(data["detailed_results"]).to_csv(detailed_csv, index=False)
            else:
                # Fallback CSV writing without pandas
                import csv
                with open(detailed_csv, 'w', newline='') as f:
                    if data["detailed_results"]:
                        writer = csv.DictWriter(f, fieldnames=data["detailed_results"][0].keys())
                        writer.writeheader()
                        writer.writerows(data["detailed_results"])
            logger.info(f"Detailed results CSV: {detailed_csv}")

def main():
    parser = argparse.ArgumentParser(description="Generate mode comparison report")
    parser.add_argument("--input", help="Path to aggregated results JSON file")
    parser.add_argument("--date", help="Date to generate report from (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default="./log/mode_reports",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    if not args.input and not args.date:
        parser.error("Must specify either --input or --date")
    
    reporter = ModeComparisonReport(args.output_dir)
    
    if args.input:
        report_file = reporter.generate_report(args.input, is_file=True)
    else:
        report_file = reporter.generate_report(args.date, is_file=False)
    
    print(f"\\nMode comparison report generated: {report_file}")

if __name__ == "__main__":
    main()