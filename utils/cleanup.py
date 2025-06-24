import gc
import torch
from contextlib import contextmanager

@contextmanager
def agent_session(*agents):
    """Context manager for automatic agent cleanup"""
    try:
        yield agents
    finally:
        # Cleanup all agents
        for agent in agents:
            if hasattr(agent, 'model_inference') and hasattr(agent.model_inference, 'llm'):
                del agent.model_inference.llm
        
        # System cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()