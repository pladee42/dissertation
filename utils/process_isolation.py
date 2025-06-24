"""
Process Isolation System for Complete Memory Separation

This module provides enterprise-grade process isolation capabilities:
- Complete memory isolation between model instances
- Fault tolerance and error recovery
- Resource monitoring and limits
- Inter-process communication for model operations
"""

import os
import sys
import time
import signal
import pickle
import logging
import multiprocessing as mp
import subprocess
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import queue
import threading
from pathlib import Path
import psutil

from config.settings import settings

logger = logging.getLogger(__name__)

class IsolationMethod(Enum):
    """Available isolation methods"""
    MULTIPROCESSING = "multiprocessing"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"  # Future implementation

@dataclass
class ProcessResult:
    """Result from isolated process execution"""
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    memory_peak_gb: float
    process_id: int
    exit_code: Optional[int] = None

@dataclass
class ProcessConfig:
    """Configuration for isolated process"""
    memory_limit_gb: Optional[float]
    timeout_seconds: int
    cpu_limit: Optional[int]
    isolation_method: IsolationMethod
    working_directory: Optional[str]
    environment_variables: Dict[str, str]

class IsolatedModelRunner:
    """Runs model operations in complete isolation"""
    
    def __init__(self, 
                 isolation_method: IsolationMethod = IsolationMethod.MULTIPROCESSING,
                 memory_limit_gb: Optional[float] = None,
                 timeout_seconds: int = 3600):
        
        self.isolation_method = isolation_method
        self.memory_limit_gb = memory_limit_gb or getattr(settings, 'isolation_memory_limit_gb', None)
        self.timeout_seconds = timeout_seconds
        self.active_processes: Dict[int, mp.Process] = {}
        self.process_monitor = ProcessMonitor()
        
        # Setup signal handlers for cleanup
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"IsolatedModelRunner initialized with {isolation_method.value} method")
    
    def run_isolated_model_operation(self, 
                                   operation: str,
                                   model_config: Dict[str, Any],
                                   operation_args: Dict[str, Any]) -> ProcessResult:
        """
        Run a model operation in complete isolation
        
        Args:
            operation: Type of operation ('load', 'generate', 'evaluate')
            model_config: Model configuration parameters
            operation_args: Operation-specific arguments
            
        Returns:
            ProcessResult with operation outcome
        """
        
        logger.info(f"Starting isolated {operation} operation")
        
        process_config = ProcessConfig(
            memory_limit_gb=self.memory_limit_gb,
            timeout_seconds=self.timeout_seconds,
            cpu_limit=None,
            isolation_method=self.isolation_method,
            working_directory=os.getcwd(),
            environment_variables=dict(os.environ)
        )
        
        if self.isolation_method == IsolationMethod.MULTIPROCESSING:
            return self._run_multiprocessing(operation, model_config, operation_args, process_config)
        elif self.isolation_method == IsolationMethod.SUBPROCESS:
            return self._run_subprocess(operation, model_config, operation_args, process_config)
        else:
            raise ValueError(f"Unsupported isolation method: {self.isolation_method}")
    
    def _run_multiprocessing(self, 
                           operation: str,
                           model_config: Dict[str, Any],
                           operation_args: Dict[str, Any],
                           process_config: ProcessConfig) -> ProcessResult:
        """Run operation using multiprocessing"""
        
        # Create communication queues
        result_queue = mp.Queue()
        error_queue = mp.Queue()
        
        # Create worker process
        worker_args = (operation, model_config, operation_args, result_queue, error_queue)
        process = mp.Process(target=self._isolated_worker, args=worker_args)
        
        start_time = time.time()
        memory_peak = 0.0
        
        try:
            # Start process with monitoring
            process.start()
            self.active_processes[process.pid] = process
            
            # Monitor process
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process.pid, process_config.memory_limit_gb)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Wait for completion with timeout
            process.join(timeout=process_config.timeout_seconds)
            
            execution_time = time.time() - start_time
            
            # Check if process completed successfully
            if process.is_alive():
                logger.warning(f"Process {process.pid} timed out, terminating")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                
                return ProcessResult(
                    success=False,
                    result=None,
                    error="Operation timed out",
                    execution_time=execution_time,
                    memory_peak_gb=memory_peak,
                    process_id=process.pid,
                    exit_code=-1
                )
            
            # Get results
            try:
                if not error_queue.empty():
                    error_msg = error_queue.get_nowait()
                    return ProcessResult(
                        success=False,
                        result=None,
                        error=error_msg,
                        execution_time=execution_time,
                        memory_peak_gb=memory_peak,
                        process_id=process.pid,
                        exit_code=process.exitcode
                    )
                
                if not result_queue.empty():
                    result = result_queue.get_nowait()
                    return ProcessResult(
                        success=True,
                        result=result,
                        error=None,
                        execution_time=execution_time,
                        memory_peak_gb=memory_peak,
                        process_id=process.pid,
                        exit_code=process.exitcode
                    )
                else:
                    return ProcessResult(
                        success=False,
                        result=None,
                        error="No result received from process",
                        execution_time=execution_time,
                        memory_peak_gb=memory_peak,
                        process_id=process.pid,
                        exit_code=process.exitcode
                    )
                    
            except queue.Empty:
                return ProcessResult(
                    success=False,
                    result=None,
                    error="Communication queue timeout",
                    execution_time=execution_time,
                    memory_peak_gb=memory_peak,
                    process_id=process.pid,
                    exit_code=process.exitcode
                )
                
        except Exception as e:
            logger.error(f"Error in isolated process execution: {e}")
            return ProcessResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time,
                memory_peak_gb=memory_peak,
                process_id=process.pid if 'process' in locals() else -1
            )
            
        finally:
            # Cleanup
            if 'process' in locals() and process.pid in self.active_processes:
                del self.active_processes[process.pid]
    
    def _run_subprocess(self,
                       operation: str,
                       model_config: Dict[str, Any], 
                       operation_args: Dict[str, Any],
                       process_config: ProcessConfig) -> ProcessResult:
        """Run operation using subprocess"""
        
        # Create temporary files for communication
        input_file = Path("/tmp") / f"model_input_{os.getpid()}_{int(time.time())}.pkl"
        output_file = Path("/tmp") / f"model_output_{os.getpid()}_{int(time.time())}.pkl"
        
        try:
            # Serialize input data
            input_data = {
                "operation": operation,
                "model_config": model_config,
                "operation_args": operation_args
            }
            
            with open(input_file, 'wb') as f:
                pickle.dump(input_data, f)
            
            # Create subprocess command
            cmd = [
                sys.executable, "-c",
                f"""
import pickle
import sys
sys.path.append('{os.getcwd()}')

from utils.process_isolation import subprocess_worker

with open('{input_file}', 'rb') as f:
    data = pickle.load(f)

result = subprocess_worker(
    data['operation'],
    data['model_config'], 
    data['operation_args']
)

with open('{output_file}', 'wb') as f:
    pickle.dump(result, f)
"""
            ]
            
            start_time = time.time()
            
            # Run subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=process_config.environment_variables,
                cwd=process_config.working_directory
            )
            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=process_config.timeout_seconds)
                execution_time = time.time() - start_time
                
                # Read result
                if output_file.exists():
                    with open(output_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    return ProcessResult(
                        success=True,
                        result=result,
                        error=None,
                        execution_time=execution_time,
                        memory_peak_gb=0.0,  # Not easily measurable in subprocess
                        process_id=process.pid,
                        exit_code=process.returncode
                    )
                else:
                    return ProcessResult(
                        success=False,
                        result=None,
                        error=stderr.decode() if stderr else "No output file created",
                        execution_time=execution_time,
                        memory_peak_gb=0.0,
                        process_id=process.pid,
                        exit_code=process.returncode
                    )
                    
            except subprocess.TimeoutExpired:
                process.kill()
                return ProcessResult(
                    success=False,
                    result=None,
                    error="Subprocess timed out",
                    execution_time=process_config.timeout_seconds,
                    memory_peak_gb=0.0,
                    process_id=process.pid,
                    exit_code=-1
                )
                
        finally:
            # Cleanup temporary files
            for temp_file in [input_file, output_file]:
                if temp_file.exists():
                    temp_file.unlink()
    
    def _isolated_worker(self,
                        operation: str,
                        model_config: Dict[str, Any],
                        operation_args: Dict[str, Any],
                        result_queue: mp.Queue,
                        error_queue: mp.Queue):
        """Worker function for isolated model operations"""
        
        try:
            # Import here to avoid issues with multiprocessing
            from agents.email_agent import EmailAgent
            from agents.checklist_agent import ChecklistAgent
            from agents.judge_agent import JudgeAgent
            
            if operation == "generate_email":
                agent = EmailAgent(
                    model_config['model_id'],
                    model_config['dtype'],
                    model_config['quantization']
                )
                
                result = agent.generate_email(
                    operation_args['prompt'],
                    operation_args['topic'],
                    operation_args.get('style', 'professional'),
                    operation_args.get('length', 'medium')
                )
                
                result_queue.put(result)
                
            elif operation == "generate_checklist":
                agent = ChecklistAgent(
                    model_config['model_id'],
                    model_config['dtype'],
                    model_config['quantization']
                )
                
                result = agent.generate_checklist(
                    operation_args['user_query'],
                    operation_args['reference_response'],
                    operation_args['topic']
                )
                
                result_queue.put(result.model_dump())
                
            elif operation == "evaluate_email":
                agent = JudgeAgent(
                    model_config['model_id'],
                    model_config['dtype'],
                    model_config['quantization']
                )
                
                # Reconstruct checklist from dict
                from models.schemas import Checklist
                checklist = Checklist(**operation_args['checklist'])
                
                result = agent.evaluate_email(
                    operation_args['email_content'],
                    checklist,
                    operation_args['user_query']
                )
                
                result_queue.put(result.model_dump())
                
            else:
                error_queue.put(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Error in isolated worker: {e}")
            error_queue.put(str(e))
    
    def _monitor_process(self, pid: int, memory_limit_gb: Optional[float]):
        """Monitor process resource usage"""
        
        if not memory_limit_gb:
            return
        
        try:
            process = psutil.Process(pid)
            
            while process.is_running():
                try:
                    memory_info = process.memory_info()
                    memory_gb = memory_info.rss / (1024**3)
                    
                    if memory_gb > memory_limit_gb:
                        logger.warning(f"Process {pid} exceeded memory limit: {memory_gb:.2f}GB > {memory_limit_gb}GB")
                        process.terminate()
                        break
                        
                    time.sleep(1)  # Check every second
                    
                except psutil.NoSuchProcess:
                    break
                    
        except Exception as e:
            logger.warning(f"Error monitoring process {pid}: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, cleaning up processes")
        self.cleanup_all_processes()
        sys.exit(0)
    
    def cleanup_all_processes(self):
        """Cleanup all active processes"""
        
        for pid, process in self.active_processes.items():
            try:
                if process.is_alive():
                    logger.info(f"Terminating process {pid}")
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up process {pid}: {e}")
        
        self.active_processes.clear()

class ProcessMonitor:
    """Monitor system resources during isolated operations"""
    
    def __init__(self):
        self.monitoring_data: List[Dict[str, Any]] = []
    
    def start_monitoring(self, interval_seconds: int = 1):
        """Start resource monitoring"""
        pass  # Implementation for production systems
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        pass

def subprocess_worker(operation: str, model_config: Dict[str, Any], operation_args: Dict[str, Any]) -> Any:
    """Worker function for subprocess isolation"""
    
    # This would implement the same logic as _isolated_worker
    # but without multiprocessing-specific queue operations
    
    try:
        if operation == "generate_email":
            from agents.email_agent import EmailAgent
            
            agent = EmailAgent(
                model_config['model_id'],
                model_config['dtype'],
                model_config['quantization']
            )
            
            return agent.generate_email(
                operation_args['prompt'],
                operation_args['topic'],
                operation_args.get('style', 'professional'),
                operation_args.get('length', 'medium')
            )
            
        # Add other operations as needed
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        raise RuntimeError(f"Subprocess worker error: {e}")

# Global isolated runner instance
_global_runner: Optional[IsolatedModelRunner] = None

def get_isolated_runner(isolation_method: IsolationMethod = IsolationMethod.MULTIPROCESSING) -> IsolatedModelRunner:
    """Get global isolated runner instance"""
    global _global_runner
    
    if _global_runner is None:
        _global_runner = IsolatedModelRunner(isolation_method)
    
    return _global_runner