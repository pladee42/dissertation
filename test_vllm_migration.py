"""
vLLM Migration Test

Simple test to verify vLLM backend compatibility with existing system.
"""

import logging
from models.vllm_backend import VLLMBackend
from agents.email_agent import EmailAgent
from config.config import get_setting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_backend():
    """Test vLLM backend basic functionality"""
    backend = VLLMBackend()
    
    logger.info("Testing vLLM backend availability...")
    if backend.is_available():
        logger.info("✅ vLLM library is available")
        return True
    else:
        logger.warning("❌ vLLM library is not available")
        return False

def test_email_agent_with_vllm():
    """Test EmailAgent with vLLM backend"""
    try:
        agent = EmailAgent(
            model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            model_key="deepseek-r1-1.5b"
        )
        
        email = agent.generate_email(
            prompt="Write a professional email",
            topic="Test Topic"
        )
        
        if email and len(email) > 10:
            logger.info("✅ EmailAgent with vLLM backend works")
            return True
        else:
            logger.warning("❌ EmailAgent returned empty or short response")
            return False
            
    except Exception as e:
        logger.error(f"❌ EmailAgent test failed: {e}")
        return False

def main():
    """Run migration tests"""
    logger.info("=== vLLM Migration Test ===")
    
    # Test 1: Backend availability
    backend_test = test_vllm_backend()
    
    # Test 2: Agent functionality
    agent_test = test_email_agent_with_vllm()
    
    # Summary
    if backend_test and agent_test:
        logger.info("✅ All tests passed - Migration successful!")
        return 0
    else:
        logger.error("❌ Some tests failed - Check vLLM library and configuration")
        return 1

if __name__ == "__main__":
    exit(main())