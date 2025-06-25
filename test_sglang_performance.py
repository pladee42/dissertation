#!/usr/bin/env python3
"""
SGLang Performance Benchmark Script
Compares VLLM vs SGLang performance across multiple metrics
"""

import time
import json
import logging
import statistics
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Any
import os

from agents.email_agent import EmailAgent
from agents.checklist_agent import ChecklistAgent
from agents.judge_agent import JudgeAgent
from agents.sglang_agent_factory import create_email_agent, create_checklist_agent, create_judge_agent
from config.models import MODELS_CONFIG
from config.settings import settings
from utils.sglang_cache_optimizer import get_cache_optimizer
from utils.sglang_advanced_memory_manager import get_advanced_memory_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Comprehensive performance benchmark for VLLM vs SGLang"""
    
    def __init__(self, models_to_test: List[str], num_iterations: int = 3):
        self.models_to_test = models_to_test
        self.num_iterations = num_iterations
        self.results = {
            "vllm_results": {},
            "sglang_results": {},
            "comparison_metrics": {}
        }
        
        # Test prompts and topics
        self.test_cases = [
            {
                "topic": "AI Research Collaboration",
                "prompt": "Write a professional email requesting collaboration on AI research between universities.",
                "style": "professional"
            },
            {
                "topic": "Conference Invitation",
                "prompt": "Create an invitation email for an international AI conference.",
                "style": "formal"
            },
            {
                "topic": "Project Update",
                "prompt": "Write an update email about the progress of a machine learning project.",
                "style": "friendly"
            }
        ]
        
        logger.info(f"Initialized benchmark for {len(models_to_test)} models, {num_iterations} iterations per test")
    
    def run_vllm_benchmark(self) -> Dict[str, Any]:
        """Run VLLM baseline benchmark"""
        
        logger.info("=== Running VLLM Baseline Benchmark ===")
        
        vllm_results = {}
        
        for model_key in self.models_to_test:
            logger.info(f"Testing VLLM model: {model_key}")
            model_results = []
            
            for iteration in range(self.num_iterations):
                logger.info(f"VLLM {model_key} - Iteration {iteration + 1}/{self.num_iterations}")
                
                iteration_results = {}
                
                for test_case in self.test_cases:
                    try:
                        # Test email generation
                        start_time = time.time()
                        
                        model_config = MODELS_CONFIG[model_key]
                        email_agent = EmailAgent(
                            model_id=model_config['model_id'],
                            dtype=model_config.get('dtype', 'bfloat16'),
                            quantization=model_config.get('quantization')
                        )
                        
                        email_content = email_agent.generate_email(
                            test_case["prompt"], test_case["topic"], test_case["style"]
                        )
                        
                        email_time = time.time() - start_time
                        
                        # Test checklist generation (using a different model for fairness)
                        checklist_start = time.time()
                        checklist_model = "deepseek-r1-8b"  # Fixed checklist model
                        checklist_config = MODELS_CONFIG[checklist_model]
                        
                        checklist_agent = ChecklistAgent(
                            model_id=checklist_config['model_id'],
                            dtype=checklist_config.get('dtype', 'bfloat16'),
                            quantization=checklist_config.get('quantization')
                        )
                        
                        checklist = checklist_agent.generate_checklist(
                            test_case["prompt"], email_content, test_case["topic"]
                        )
                        
                        checklist_time = time.time() - checklist_start
                        
                        # Test evaluation
                        eval_start = time.time()
                        judge_model = "gemma-3-4b"  # Fixed judge model
                        judge_config = MODELS_CONFIG[judge_model]
                        
                        judge_agent = JudgeAgent(
                            model_id=judge_config['model_id'],
                            dtype=judge_config.get('dtype', 'bfloat16'),
                            quantization=judge_config.get('quantization')
                        )
                        
                        evaluation = judge_agent.evaluate_email(
                            email_content, checklist, test_case["prompt"]
                        )
                        
                        eval_time = time.time() - eval_start
                        
                        # Store results
                        iteration_results[test_case["topic"]] = {
                            "email_generation_time": email_time,
                            "checklist_generation_time": checklist_time,
                            "evaluation_time": eval_time,
                            "total_time": email_time + checklist_time + eval_time,
                            "overall_score": evaluation.overall_score,
                            "weighted_score": evaluation.weighted_score,
                            "email_length": len(email_content.split()),
                            "checklist_items": len(checklist.items)
                        }
                        
                        # Cleanup
                        email_agent.cleanup()
                        checklist_agent.cleanup()
                        judge_agent.cleanup()
                        
                    except Exception as e:
                        logger.error(f"VLLM test failed for {model_key} - {test_case['topic']}: {e}")
                        iteration_results[test_case["topic"]] = {"error": str(e)}
                
                model_results.append(iteration_results)
            
            vllm_results[model_key] = model_results
        
        self.results["vllm_results"] = vllm_results
        return vllm_results
    
    def run_sglang_benchmark(self) -> Dict[str, Any]:
        """Run SGLang performance benchmark"""
        
        logger.info("=== Running SGLang Performance Benchmark ===")
        
        # Set SGLang backend
        os.environ["AGENT_BACKEND"] = "sglang"
        os.environ["INFERENCE_BACKEND"] = "sglang"
        
        # Initialize SGLang components
        cache_optimizer = get_cache_optimizer()
        memory_manager = get_advanced_memory_manager()
        
        sglang_results = {}
        
        for model_key in self.models_to_test:
            logger.info(f"Testing SGLang model: {model_key}")
            model_results = []
            
            for iteration in range(self.num_iterations):
                logger.info(f"SGLang {model_key} - Iteration {iteration + 1}/{self.num_iterations}")
                
                iteration_results = {}
                
                for test_case in self.test_cases:
                    try:
                        # Warm cache for this topic
                        cache_warming_start = time.time()
                        cache_optimizer.warm_cache_for_topic(
                            test_case["topic"], ["email", "checklist", "judge"]
                        )
                        cache_warming_time = time.time() - cache_warming_start
                        
                        # Test SGLang email generation
                        start_time = time.time()
                        
                        model_config = MODELS_CONFIG[model_key]
                        email_agent = create_email_agent(
                            model_id=model_config['model_id'],
                            dtype=model_config.get('dtype', 'bfloat16'),
                            quantization=model_config.get('quantization'),
                            backend="sglang"
                        )
                        
                        # Use structured generation if available
                        if hasattr(email_agent, 'generate_email_with_structure'):
                            email_result = email_agent.generate_email_with_structure(
                                test_case["prompt"], test_case["topic"], test_case["style"]
                            )
                            email_content = email_result.content
                        else:
                            email_content = email_agent.generate_email(
                                test_case["prompt"], test_case["topic"], test_case["style"]
                            )
                        
                        email_time = time.time() - start_time
                        
                        # Test SGLang checklist generation with xgrammar
                        checklist_start = time.time()
                        checklist_model = "deepseek-r1-8b"
                        checklist_config = MODELS_CONFIG[checklist_model]
                        
                        checklist_agent = create_checklist_agent(
                            model_id=checklist_config['model_id'],
                            dtype=checklist_config.get('dtype', 'bfloat16'),
                            quantization=checklist_config.get('quantization'),
                            backend="sglang"
                        )
                        
                        # Use xgrammar if available
                        if hasattr(checklist_agent, 'generate_checklist_with_xgrammar'):
                            checklist = checklist_agent.generate_checklist_with_xgrammar(
                                test_case["prompt"], email_content, test_case["topic"]
                            )
                        else:
                            checklist = checklist_agent.generate_checklist(
                                test_case["prompt"], email_content, test_case["topic"]
                            )
                        
                        checklist_time = time.time() - checklist_start
                        
                        # Test SGLang structured evaluation
                        eval_start = time.time()
                        judge_model = "gemma-3-4b"
                        judge_config = MODELS_CONFIG[judge_model]
                        
                        judge_agent = create_judge_agent(
                            model_id=judge_config['model_id'],
                            dtype=judge_config.get('dtype', 'bfloat16'),
                            quantization=judge_config.get('quantization'),
                            backend="sglang"
                        )
                        
                        # Use structured evaluation if available
                        if hasattr(judge_agent, 'evaluate_email_structured'):
                            evaluation = judge_agent.evaluate_email_structured(
                                email_content, checklist, test_case["prompt"]
                            )
                        else:
                            evaluation = judge_agent.evaluate_email(
                                email_content, checklist, test_case["prompt"]
                            )
                        
                        eval_time = time.time() - eval_start
                        
                        # Get cache metrics
                        cache_hit_rate = cache_optimizer._calculate_global_hit_rate()
                        
                        # Get memory profile
                        memory_profile = memory_manager.get_comprehensive_memory_profile()
                        
                        # Store results with SGLang-specific metrics
                        iteration_results[test_case["topic"]] = {
                            "email_generation_time": email_time,
                            "checklist_generation_time": checklist_time,
                            "evaluation_time": eval_time,
                            "cache_warming_time": cache_warming_time,
                            "total_time": email_time + checklist_time + eval_time,
                            "overall_score": evaluation.overall_score,
                            "weighted_score": evaluation.weighted_score,
                            "email_length": len(email_content.split()),
                            "checklist_items": len(checklist.items),
                            # SGLang-specific metrics
                            "cache_hit_rate": cache_hit_rate,
                            "radix_cache_size_gb": memory_profile.radix_cache_size_gb,
                            "memory_efficiency": memory_profile.cache_efficiency,
                            "throughput_tokens_per_second": memory_profile.throughput_tokens_per_second
                        }
                        
                        # Cleanup
                        email_agent.cleanup()
                        checklist_agent.cleanup()
                        judge_agent.cleanup()
                        
                    except Exception as e:
                        logger.error(f"SGLang test failed for {model_key} - {test_case['topic']}: {e}")
                        iteration_results[test_case["topic"]] = {"error": str(e)}
                
                model_results.append(iteration_results)
            
            sglang_results[model_key] = model_results
        
        # Cleanup SGLang resources
        memory_manager.cleanup_advanced_resources()
        
        self.results["sglang_results"] = sglang_results
        return sglang_results
    
    def calculate_comparison_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive comparison metrics"""
        
        logger.info("=== Calculating Comparison Metrics ===")
        
        comparison_metrics = {
            "throughput_improvement": {},
            "latency_reduction": {},
            "memory_efficiency": {},
            "quality_comparison": {},
            "cache_benefits": {},
            "overall_performance": {}
        }
        
        for model_key in self.models_to_test:
            if model_key not in self.results["vllm_results"] or model_key not in self.results["sglang_results"]:
                continue
            
            vllm_data = self.results["vllm_results"][model_key]
            sglang_data = self.results["sglang_results"][model_key]
            
            model_comparison = {}
            
            for test_case in self.test_cases:
                topic = test_case["topic"]
                
                # Extract times for valid iterations
                vllm_times = []
                sglang_times = []
                vllm_scores = []
                sglang_scores = []
                
                for iteration in vllm_data:
                    if topic in iteration and "error" not in iteration[topic]:
                        vllm_times.append(iteration[topic]["total_time"])
                        vllm_scores.append(iteration[topic]["overall_score"])
                
                for iteration in sglang_data:
                    if topic in iteration and "error" not in iteration[topic]:
                        sglang_times.append(iteration[topic]["total_time"])
                        sglang_scores.append(iteration[topic]["overall_score"])
                
                if vllm_times and sglang_times:
                    # Calculate metrics
                    vllm_avg_time = statistics.mean(vllm_times)
                    sglang_avg_time = statistics.mean(sglang_times)
                    
                    vllm_avg_score = statistics.mean(vllm_scores)
                    sglang_avg_score = statistics.mean(sglang_scores)
                    
                    speedup = vllm_avg_time / sglang_avg_time if sglang_avg_time > 0 else 0
                    quality_ratio = sglang_avg_score / vllm_avg_score if vllm_avg_score > 0 else 0
                    
                    model_comparison[topic] = {
                        "vllm_avg_time": vllm_avg_time,
                        "sglang_avg_time": sglang_avg_time,
                        "speedup": speedup,
                        "latency_reduction_percent": (1 - sglang_avg_time / vllm_avg_time) * 100 if vllm_avg_time > 0 else 0,
                        "vllm_avg_score": vllm_avg_score,
                        "sglang_avg_score": sglang_avg_score,
                        "quality_ratio": quality_ratio,
                        "quality_improvement_percent": (quality_ratio - 1) * 100
                    }
                    
                    # Add SGLang-specific metrics if available
                    sglang_metrics = []
                    for iteration in sglang_data:
                        if topic in iteration and "error" not in iteration[topic]:
                            sglang_metrics.append(iteration[topic])
                    
                    if sglang_metrics:
                        avg_cache_hit_rate = statistics.mean([m.get("cache_hit_rate", 0) for m in sglang_metrics])
                        avg_memory_efficiency = statistics.mean([m.get("memory_efficiency", 0) for m in sglang_metrics])
                        
                        model_comparison[topic].update({
                            "avg_cache_hit_rate": avg_cache_hit_rate,
                            "avg_memory_efficiency": avg_memory_efficiency
                        })
            
            comparison_metrics["throughput_improvement"][model_key] = model_comparison
        
        # Calculate overall summary
        all_speedups = []
        all_quality_ratios = []
        all_cache_hit_rates = []
        
        for model_data in comparison_metrics["throughput_improvement"].values():
            for topic_data in model_data.values():
                if "speedup" in topic_data:
                    all_speedups.append(topic_data["speedup"])
                if "quality_ratio" in topic_data:
                    all_quality_ratios.append(topic_data["quality_ratio"])
                if "avg_cache_hit_rate" in topic_data:
                    all_cache_hit_rates.append(topic_data["avg_cache_hit_rate"])
        
        comparison_metrics["overall_performance"] = {
            "average_speedup": statistics.mean(all_speedups) if all_speedups else 0,
            "median_speedup": statistics.median(all_speedups) if all_speedups else 0,
            "max_speedup": max(all_speedups) if all_speedups else 0,
            "min_speedup": min(all_speedups) if all_speedups else 0,
            "average_quality_ratio": statistics.mean(all_quality_ratios) if all_quality_ratios else 0,
            "average_cache_hit_rate": statistics.mean(all_cache_hit_rates) if all_cache_hit_rates else 0,
            "performance_target_met": statistics.mean(all_speedups) >= 2.0 if all_speedups else False,  # 2x target
            "quality_maintained": statistics.mean(all_quality_ratios) >= 0.95 if all_quality_ratios else False  # 95% quality retention
        }
        
        self.results["comparison_metrics"] = comparison_metrics
        return comparison_metrics
    
    def save_results(self, output_dir: Path = None) -> Path:
        """Save comprehensive benchmark results"""
        
        if output_dir is None:
            output_dir = Path(settings.output_dir) / "benchmarks"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        results_file = output_dir / f"sglang_vs_vllm_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary report
        summary = {
            "benchmark_config": {
                "models_tested": self.models_to_test,
                "iterations_per_test": self.num_iterations,
                "test_cases": len(self.test_cases)
            },
            "key_findings": self.results.get("comparison_metrics", {}).get("overall_performance", {}),
            "detailed_results": self.results.get("comparison_metrics", {}).get("throughput_improvement", {})
        }
        
        summary_file = output_dir / f"benchmark_summary_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")
        
        return results_file

def main():
    parser = ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["deepseek-r1-1.5b", "llama-3-3b"], 
                       choices=MODELS_CONFIG.keys(), help="Models to benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per test")
    parser.add_argument("--skip_vllm", action="store_true", help="Skip VLLM baseline (SGLang only)")
    parser.add_argument("--skip_sglang", action="store_true", help="Skip SGLang benchmark (VLLM only)")
    
    args = parser.parse_args()
    
    logger.info("=== SGLang Performance Benchmark ===")
    logger.info(f"Models: {args.models}")
    logger.info(f"Iterations: {args.iterations}")
    
    benchmark = PerformanceBenchmark(args.models, args.iterations)
    
    try:
        # Run benchmarks
        if not args.skip_vllm:
            benchmark.run_vllm_benchmark()
        
        if not args.skip_sglang:
            benchmark.run_sglang_benchmark()
        
        # Calculate comparison metrics
        if not args.skip_vllm and not args.skip_sglang:
            benchmark.calculate_comparison_metrics()
        
        # Save results
        results_file = benchmark.save_results()
        
        # Print summary
        if benchmark.results.get("comparison_metrics", {}).get("overall_performance"):
            perf = benchmark.results["comparison_metrics"]["overall_performance"]
            
            logger.info("=== BENCHMARK RESULTS SUMMARY ===")
            logger.info(f"Average Speedup: {perf['average_speedup']:.2f}x")
            logger.info(f"Median Speedup: {perf['median_speedup']:.2f}x")
            logger.info(f"Max Speedup: {perf['max_speedup']:.2f}x")
            logger.info(f"Average Cache Hit Rate: {perf['average_cache_hit_rate']:.3f}")
            logger.info(f"Quality Maintained: {perf['quality_maintained']}")
            logger.info(f"Performance Target Met (2x): {perf['performance_target_met']}")
        
        logger.info("Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()