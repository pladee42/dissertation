"""
Simplified Pipeline Testing

This module provides basic testing of the email generation pipeline:
- Simple test execution
- Basic functionality verification
- Minimal complexity
"""

import logging
import time
from config.config import MODELS, get_model_config, get_setting
from models.orchestrator import SimpleModelOrchestrator
from models.sglang_backend import SGLangBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_pipeline():
    """Test the simplified pipeline with basic models"""
    logger.info("Starting simplified pipeline test")
    
    # Check SGLang server connectivity first
    sglang_url = get_setting('sglang_server_url', 'http://localhost:30000')
    backend = SGLangBackend(base_url=sglang_url)
    
    if not backend.is_available():
        logger.error(f"SGLang server not available at {sglang_url}")
        return {"success": False, "error": "SGLang server not available"}
    
    logger.info(f"SGLang server is available at {sglang_url}")
    
    # Use simple models for testing
    email_models = ["deepseek-r1-1.5b", "llama-3-3b"]
    checklist_model = "deepseek-r1-8b"
    judge_model = "gemma-3-4b"
    
    # Create orchestrator
    orchestrator = SimpleModelOrchestrator(
        email_models=email_models,
        checklist_model=checklist_model,
        judge_model=judge_model,
        max_concurrent=1  # Keep it simple
    )
    
    # Test topic
    topic = "AI Research Collaboration"
    prompt = "Write a professional email about [TOPIC]"
    
    try:
        # Run complete pipeline
        logger.info("Running complete pipeline test...")
        start_time = time.time()
        
        result = orchestrator.generate_and_rank_emails(
            prompt=prompt,
            topic=topic,
            user_query=f"Email about {topic}"
        )
        
        total_time = time.time() - start_time
        
        # Simple validation
        if result.get("success"):
            logger.info("✅ Pipeline test PASSED")
            logger.info(f"Generated {len(result.get('emails', []))} emails")
            logger.info(f"Best email from: {result.get('best_email', {}).get('model_name', 'unknown')}")
            logger.info(f"Total time: {total_time:.2f}s")
        else:
            logger.error("❌ Pipeline test FAILED")
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED with error: {e}")
        return {"success": False, "error": str(e)}

def test_individual_components():
    """Test individual components"""
    logger.info("Testing individual components")
    
    try:
        # Test model config
        config = get_model_config("deepseek-r1-1.5b")
        assert config.get("model_id"), "Model config should have model_id"
        logger.info("✅ Config test passed")
        
        # Test models list
        assert len(MODELS) > 0, "Should have models configured"
        logger.info("✅ Models list test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Running simplified pipeline tests")
    
    # Run tests
    component_result = test_individual_components()
    pipeline_result = test_simple_pipeline()
    
    # Summary
    if component_result and pipeline_result.get("success"):
        logger.info("🎉 All tests PASSED")
    else:
        logger.error("💥 Some tests FAILED")
        
    logger.info("Test execution completed")