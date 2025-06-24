import gc
import torch
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def agent_session(*agents):
    """Context manager for automatic agent cleanup with enhanced memory management"""
    
    # Log memory usage before
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        logger.info(f"GPU memory before agent session: {initial_memory / 1024**3:.2f} GB")
    
    try:
        yield agents
    finally:
        # Enhanced cleanup for all agents
        for agent in agents:
            try:
                # Correct attribute path: agent.llm (ModelInference) contains .llm (VLLM object)
                if hasattr(agent, 'llm') and hasattr(agent.llm, 'llm'):
                    logger.info(f"Cleaning up agent with model: {getattr(agent.llm, 'model_name', 'unknown')}")
                    
                    # Call the enhanced cleanup method
                    if hasattr(agent.llm, 'cleanup'):
                        agent.llm.cleanup()
                    else:
                        # Fallback cleanup
                        del agent.llm.llm
                        if hasattr(agent.llm, 'is_cleaned_up'):
                            agent.llm.is_cleaned_up = True
                            
            except Exception as e:
                logger.warning(f"Error during agent cleanup: {e}")
        
        # Aggressive system cleanup
        _aggressive_memory_cleanup()

def _aggressive_memory_cleanup():
    """Perform aggressive GPU and system memory cleanup"""
    
    # Multiple rounds of garbage collection
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        # Clear CUDA cache multiple times
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Log final memory usage
        final_memory = torch.cuda.memory_allocated()
        logger.info(f"GPU memory after cleanup: {final_memory / 1024**3:.2f} GB")
        
        # Additional memory clearing for problematic scenarios
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray cluster shutdown completed")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Ray shutdown failed: {e}")

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    memory_allocated = torch.cuda.memory_allocated()
    memory_reserved = torch.cuda.memory_reserved()
    memory_free = torch.cuda.get_device_properties(0).total_memory - memory_reserved
    
    return {
        "available": True,
        "allocated_gb": memory_allocated / 1024**3,
        "reserved_gb": memory_reserved / 1024**3,
        "free_gb": memory_free / 1024**3,
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "utilization_percent": (memory_allocated / torch.cuda.get_device_properties(0).total_memory) * 100
    }

def check_memory_availability(required_gb=4.0, safety_margin=1.0):
    """Check if sufficient GPU memory is available for model loading"""
    memory_info = get_gpu_memory_info()
    
    if not memory_info["available"]:
        logger.warning("CUDA not available, cannot check GPU memory")
        return False
    
    required_with_margin = required_gb + safety_margin
    available = memory_info["free_gb"]
    
    logger.info(f"Memory check: {available:.2f}GB available, {required_with_margin:.2f}GB required")
    
    if available < required_with_margin:
        logger.warning(f"Insufficient memory: {available:.2f}GB available < {required_with_margin:.2f}GB required")
        return False
    
    return True

@contextmanager
def sequential_agent_stage(agent_class, model_config, stage_name, required_memory_gb=4.0):
    """Context manager for sequential agent processing with memory monitoring"""
    
    logger.info(f"=== Starting {stage_name} Stage ===")
    
    # Pre-stage memory check
    memory_before = get_gpu_memory_info()
    if memory_before["available"]:
        logger.info(f"Memory before {stage_name}: {memory_before['allocated_gb']:.2f}GB allocated, "
                   f"{memory_before['free_gb']:.2f}GB free")
    
    # Check if we have enough memory
    if not check_memory_availability(required_memory_gb):
        logger.error(f"Insufficient memory for {stage_name} stage")
        # Force cleanup before failing
        _aggressive_memory_cleanup()
        raise RuntimeError(f"Insufficient GPU memory for {stage_name} stage")
    
    agent = None
    try:
        # Initialize agent
        logger.info(f"Initializing {stage_name} agent...")
        agent = agent_class(
            model_config['model_id'],
            model_config['dtype'], 
            model_config['quantization']
        )
        
        # Post-initialization memory check
        memory_after_init = get_gpu_memory_info()
        if memory_after_init["available"]:
            memory_used = memory_after_init['allocated_gb'] - memory_before['allocated_gb']
            logger.info(f"Memory after {stage_name} init: {memory_after_init['allocated_gb']:.2f}GB "
                       f"(+{memory_used:.2f}GB used)")
        
        yield agent
        
    except Exception as e:
        logger.error(f"Error in {stage_name} stage: {e}")
        raise
        
    finally:
        # Cleanup agent
        if agent is not None:
            try:
                logger.info(f"Cleaning up {stage_name} agent...")
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
                else:
                    # Fallback cleanup
                    if hasattr(agent, 'llm') and hasattr(agent.llm, 'cleanup'):
                        agent.llm.cleanup()
            except Exception as e:
                logger.warning(f"Error during {stage_name} cleanup: {e}")
        
        # Post-cleanup memory verification
        _aggressive_memory_cleanup()
        
        memory_after = get_gpu_memory_info()
        if memory_after["available"]:
            logger.info(f"Memory after {stage_name} cleanup: {memory_after['allocated_gb']:.2f}GB allocated")
        
        logger.info(f"=== Completed {stage_name} Stage ===")

@contextmanager  
def sequential_pipeline(*stage_configs):
    """Context manager for complete sequential agent pipeline"""
    
    logger.info("=== Starting Sequential Agent Pipeline ===")
    
    initial_memory = get_gpu_memory_info()
    if initial_memory["available"]:
        logger.info(f"Pipeline start memory: {initial_memory['allocated_gb']:.2f}GB allocated")
    
    try:
        yield stage_configs
        
    finally:
        # Final pipeline cleanup
        logger.info("=== Pipeline Cleanup ===")
        _aggressive_memory_cleanup()
        
        final_memory = get_gpu_memory_info()
        if final_memory["available"]:
            logger.info(f"Pipeline end memory: {final_memory['allocated_gb']:.2f}GB allocated")
            
        logger.info("=== Sequential Agent Pipeline Completed ===")

@contextmanager
def memory_monitored_agent(agent, stage_name="Unknown"):
    """Context manager for individual agent with detailed memory monitoring"""
    
    memory_before = get_gpu_memory_info()
    logger.info(f"Starting {stage_name} with {memory_before.get('free_gb', 'unknown'):.2f}GB free")
    
    try:
        yield agent
        
    finally:
        # Agent-specific cleanup with monitoring
        try:
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
            
            memory_after = get_gpu_memory_info()
            if memory_after["available"] and memory_before["available"]:
                memory_freed = memory_before['allocated_gb'] - memory_after['allocated_gb']
                logger.info(f"Completed {stage_name}, freed {memory_freed:.2f}GB memory")
                
        except Exception as e:
            logger.warning(f"Error in {stage_name} cleanup: {e}")