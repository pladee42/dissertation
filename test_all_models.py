#!/usr/bin/env python3
"""
Test script for all models in config/config.py
Tests model loading and basic inference for each model
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any
import torch
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import MODELS, get_model_config, get_memory_requirement
from models.vllm_backend import VLLMBackend

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    """Test all models in the configuration"""
    
    def __init__(self, output_dir: str = "./test_results"):
        self.output_dir = output_dir
        self.backend = VLLMBackend(max_parallel=1)  # Single model testing
        self.results = {}
        self.test_prompt = "Hello, how are you today?"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup environment
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup environment for testing"""
        # Set environment variables
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["HF_HOME"] = "./downloaded_models"
        
        # Create cache directory
        os.makedirs("./downloaded_models", exist_ok=True)
        
        logger.info("Environment setup complete")
    
    def _cleanup_gpu_memory(self):
        """Cleanup GPU memory between tests"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            logger.info("GPU memory cleanup completed")
        except Exception as e:
            logger.warning(f"GPU cleanup failed: {e}")
    
    def test_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single model"""
        logger.info(f"Testing model: {model_name} (UID: {model_config.get('uid', 'N/A')})")
        
        result = {
            'model_name': model_name,
            'uid': model_config.get('uid', 'N/A'),
            'model_id': model_config.get('model_id', 'N/A'),
            'size': model_config.get('size', 'N/A'),
            'quantization': model_config.get('quantization', 'N/A'),
            'dtype': model_config.get('dtype', 'N/A'),
            'recommended_for': model_config.get('recommended_for', []),
            'memory_requirement': get_memory_requirement(model_name),
            'status': 'pending',
            'error': None,
            'response': None,
            'response_length': 0,
            'load_time': 0,
            'inference_time': 0,
            'test_timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check available GPU memory before loading
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                logger.info(f"Available GPU memory: {free_memory / 1024**3:.2f} GB")
            
            # Test model loading and inference
            start_time = datetime.now()
            
            logger.info(f"Loading model: {model_config.get('model_id', model_name)}")
            
            # Generate response
            response = self.backend.generate(
                prompt=self.test_prompt,
                model=model_name,
                max_tokens=50,
                temperature=0.7
            )
            
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Update result
            result['status'] = 'success'
            result['response'] = response
            result['response_length'] = len(response) if response else 0
            result['inference_time'] = total_time
            
            logger.info(f"✅ Model {model_name} test successful - Response length: {len(response)} chars")
            
        except Exception as e:
            error_msg = str(e)
            result['status'] = 'failed'
            result['error'] = error_msg
            
            logger.error(f"❌ Model {model_name} test failed: {error_msg}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
        finally:
            # Cleanup memory after each test
            self._cleanup_gpu_memory()
        
        return result
    
    def test_all_models(self) -> Dict[str, Any]:
        """Test all models in the configuration"""
        logger.info(f"Starting tests for {len(MODELS)} models")
        
        summary = {
            'total_models': len(MODELS),
            'successful': 0,
            'failed': 0,
            'test_start_time': datetime.now().isoformat(),
            'test_end_time': None,
            'results': {}
        }
        
        for model_name, model_config in MODELS.items():
            try:
                result = self.test_model(model_name, model_config)
                self.results[model_name] = result
                summary['results'][model_name] = result
                
                if result['status'] == 'success':
                    summary['successful'] += 1
                else:
                    summary['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Critical error testing {model_name}: {e}")
                summary['failed'] += 1
                summary['results'][model_name] = {
                    'model_name': model_name,
                    'status': 'critical_error',
                    'error': str(e)
                }
        
        summary['test_end_time'] = datetime.now().isoformat()
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any]):
        """Generate test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"model_test_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Start Time: {summary['test_start_time']}\n")
            f.write(f"Test End Time: {summary['test_end_time']}\n")
            f.write(f"Total Models: {summary['total_models']}\n")
            f.write(f"Successful: {summary['successful']}\n")
            f.write(f"Failed: {summary['failed']}\n")
            f.write(f"Success Rate: {(summary['successful'] / summary['total_models'] * 100):.1f}%\n\n")
            
            # Model results by size
            for size in ['small', 'medium', 'large']:
                models_by_size = [name for name, result in summary['results'].items() 
                                if result.get('size') == size]
                if models_by_size:
                    f.write(f"\n{size.upper()} MODELS ({len(models_by_size)} models):\n")
                    f.write("-" * 50 + "\n")
                    
                    for model_name in models_by_size:
                        result = summary['results'][model_name]
                        status_icon = "✅" if result['status'] == 'success' else "❌"
                        f.write(f"{status_icon} {model_name} ({result.get('uid', 'N/A')})\n")
                        f.write(f"   Model ID: {result.get('model_id', 'N/A')}\n")
                        f.write(f"   Status: {result['status']}\n")
                        
                        if result['status'] == 'success':
                            f.write(f"   Response Length: {result.get('response_length', 0)} chars\n")
                            f.write(f"   Inference Time: {result.get('inference_time', 0):.2f}s\n")
                        else:
                            f.write(f"   Error: {result.get('error', 'Unknown error')}\n")
                        f.write("\n")
            
            # Failed models summary
            failed_models = [name for name, result in summary['results'].items() 
                           if result['status'] != 'success']
            if failed_models:
                f.write("\nFAILED MODELS SUMMARY:\n")
                f.write("-" * 30 + "\n")
                for model_name in failed_models:
                    result = summary['results'][model_name]
                    f.write(f"❌ {model_name}: {result.get('error', 'Unknown error')}\n")
        
        logger.info(f"Test report saved to: {report_file}")
        return report_file
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 80)
        print("MODEL TESTING SUMMARY")
        print("=" * 80)
        print(f"Total Models: {summary['total_models']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {(summary['successful'] / summary['total_models'] * 100):.1f}%")
        
        # Show successful models
        successful_models = [name for name, result in summary['results'].items() 
                           if result['status'] == 'success']
        if successful_models:
            print(f"\n✅ SUCCESSFUL MODELS ({len(successful_models)}):")
            for model_name in successful_models:
                result = summary['results'][model_name]
                print(f"   {model_name} ({result.get('uid', 'N/A')}) - {result.get('size', 'N/A')} - {result.get('inference_time', 0):.2f}s")
        
        # Show failed models
        failed_models = [name for name, result in summary['results'].items() 
                       if result['status'] != 'success']
        if failed_models:
            print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
            for model_name in failed_models:
                result = summary['results'][model_name]
                print(f"   {model_name} ({result.get('uid', 'N/A')}) - {result.get('error', 'Unknown error')}")
        
        print("=" * 80)

def main():
    """Main testing function"""
    logger.info("Starting model testing script")
    
    # Check if vLLM is available
    try:
        import vllm
        logger.info("vLLM is available")
    except ImportError:
        logger.error("vLLM is not available! Please install vLLM first.")
        sys.exit(1)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available - {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA is not available - testing may be limited")
    
    # Create tester and run tests
    tester = ModelTester()
    
    try:
        summary = tester.test_all_models()
        
        # Generate report and print summary
        report_file = tester.generate_report(summary)
        tester.print_summary(summary)
        
        logger.info(f"Testing completed. Report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if summary['failed'] == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error during testing: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()