"""
Dask client implementation.

This module provides a comprehensive Dask distributed computing client wrapper
for parallel task execution, cluster management, and result gathering with explicit error handling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import time
import hashlib


class TaskStatus(Enum):
    """Status of a Dask task."""
    PENDING = "pending"
    PROCESSING = "processing"
    MEMORY = "memory"
    ERRED = "erred"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class WorkerStatus(Enum):
    """Status of a Dask worker."""
    RUNNING = "running"
    IDLE = "idle"
    BUSY = "busy"
    CLOSED = "closed"


@dataclass
class DaskConfig:
    """
    Configuration for Dask client.
    
    Attributes:
        scheduler_address: Scheduler address (host:port)
        n_workers: Number of workers (for local cluster)
        threads_per_worker: Threads per worker
        memory_limit: Memory limit per worker (e.g., "4GB")
        timeout_seconds: Default timeout for operations
        heartbeat_interval_seconds: Heartbeat interval
        enable_work_stealing: Enable work stealing between workers
    """
    scheduler_address: str = ""
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit: str = "4GB"
    timeout_seconds: float = 300.0
    heartbeat_interval_seconds: float = 1.0
    enable_work_stealing: bool = True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.n_workers < 1:
            return (False, "n_workers must be at least 1")
        
        if self.threads_per_worker < 1:
            return (False, "threads_per_worker must be at least 1")
        
        if not self.memory_limit:
            return (False, "memory_limit cannot be empty")
        
        if self.timeout_seconds <= 0:
            return (False, "timeout_seconds must be positive")
        
        if self.heartbeat_interval_seconds <= 0:
            return (False, "heartbeat_interval_seconds must be positive")
        
        return (True, "")


@dataclass
class DaskTask:
    """
    A Dask task.
    
    Attributes:
        task_id: Unique task identifier
        function_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        status: Current status
        result: Task result
        error_message: Error message if failed
        worker_id: Worker processing the task
        submitted_at: Submission timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        dependencies: List of dependent task IDs
    """
    task_id: str
    function_name: str
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error_message: str = ""
    worker_id: str = ""
    submitted_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if task is valid."""
        return bool(self.task_id and self.function_name)
    
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in {TaskStatus.FINISHED, TaskStatus.ERRED, TaskStatus.CANCELLED}
    
    def duration_seconds(self) -> float:
        """Get duration of task execution."""
        if self.started_at == 0:
            return 0.0
        
        end_time = self.completed_at if self.completed_at > 0 else time.time()
        return end_time - self.started_at


@dataclass
class Worker:
    """
    A Dask worker.
    
    Attributes:
        worker_id: Unique worker identifier
        address: Worker address
        status: Current status
        n_cores: Number of cores
        memory_limit: Memory limit in bytes
        memory_used: Memory currently used
        tasks_processing: Number of tasks being processed
        tasks_completed: Number of tasks completed
        last_heartbeat: Last heartbeat timestamp
    """
    worker_id: str
    address: str
    status: WorkerStatus
    n_cores: int
    memory_limit: int
    memory_used: int = 0
    tasks_processing: int = 0
    tasks_completed: int = 0
    last_heartbeat: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if worker is valid."""
        return bool(self.worker_id and self.address)
    
    def is_available(self) -> bool:
        """Check if worker is available for tasks."""
        return self.status in {WorkerStatus.RUNNING, WorkerStatus.IDLE}
    
    def memory_available(self) -> int:
        """Get available memory in bytes."""
        return max(0, self.memory_limit - self.memory_used)


class DaskClient:
    """
    Dask distributed computing client wrapper.
    
    Provides task submission, cluster management, and result gathering
    with explicit error handling and resource tracking.
    
    Note: This is a mock implementation for the zero-error architecture.
    In production, this would wrap the actual Dask distributed client.
    """
    
    def __init__(self, config: DaskConfig):
        """
        Initialize Dask client.
        
        Args:
            config: Dask configuration
        """
        self.config = config
        self.connected = False
        self.cluster_started = False
        
        # Mock storage
        self.tasks: Dict[str, DaskTask] = {}
        self.workers: Dict[str, Worker] = {}
        self.futures: Dict[str, Any] = {}
        
        # Statistics
        self.tasks_submitted = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.connection_time = 0.0
        self.task_counter = 0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate client configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def connect(self, start_cluster: bool = False) -> Tuple[bool, str]:
        """
        Connect to Dask scheduler or start local cluster.
        
        Args:
            start_cluster: Whether to start a local cluster
        
        Returns:
            Tuple of (success, message)
        """
        # Validate config first
        is_valid, error_msg = self.validate_config()
        if not is_valid:
            return (False, f"Invalid config: {error_msg}")
        
        if start_cluster:
            # Start local cluster
            success, msg = self._start_local_cluster()
            if not success:
                return (False, msg)
        
        # Simulate connection
        self.connected = True
        self.connection_time = time.time()
        
        if self.config.scheduler_address:
            return (True, f"Connected to Dask scheduler at {self.config.scheduler_address}")
        else:
            return (True, f"Connected to local Dask cluster with {len(self.workers)} workers")
    
    def disconnect(self) -> Tuple[bool, str]:
        """
        Disconnect from Dask cluster.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected")
        
        self.connected = False
        
        if self.cluster_started:
            self._stop_local_cluster()
        
        return (True, "Disconnected from Dask")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected
    
    def _start_local_cluster(self) -> Tuple[bool, str]:
        """
        Start a local Dask cluster.
        
        Returns:
            Tuple of (success, message)
        """
        # Create workers
        for i in range(self.config.n_workers):
            worker_id = f"worker_{i}"
            address = f"tcp://localhost:{8786 + i}"
            
            # Parse memory limit (simple parsing)
            memory_str = self.config.memory_limit.upper()
            if "GB" in memory_str:
                memory_limit = int(memory_str.replace("GB", "")) * 1024 * 1024 * 1024
            elif "MB" in memory_str:
                memory_limit = int(memory_str.replace("MB", "")) * 1024 * 1024
            else:
                memory_limit = 4 * 1024 * 1024 * 1024  # Default 4GB
            
            worker = Worker(
                worker_id=worker_id,
                address=address,
                status=WorkerStatus.RUNNING,
                n_cores=self.config.threads_per_worker,
                memory_limit=memory_limit,
                last_heartbeat=time.time()
            )
            
            self.workers[worker_id] = worker
        
        self.cluster_started = True
        
        return (True, f"Started local cluster with {len(self.workers)} workers")
    
    def _stop_local_cluster(self) -> None:
        """Stop the local Dask cluster."""
        for worker in self.workers.values():
            worker.status = WorkerStatus.CLOSED
        
        self.cluster_started = False
    
    # Task submission
    
    def submit(
        self,
        function_name: str,
        *args,
        **kwargs
    ) -> Tuple[bool, Optional[str], str]:
        """
        Submit a task for execution.
        
        Args:
            function_name: Name of the function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Tuple of (success, task_id or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Dask")
        
        if not function_name:
            return (False, None, "function_name cannot be empty")
        
        # Generate task ID
        task_id = self._generate_task_id(function_name)
        
        # Create task
        current_time = time.time()
        task = DaskTask(
            task_id=task_id,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            status=TaskStatus.PENDING,
            submitted_at=current_time
        )
        
        if not task.is_valid():
            return (False, None, "Invalid task created")
        
        # Store task
        self.tasks[task_id] = task
        self.tasks_submitted += 1
        
        return (True, task_id, f"Task submitted: {task_id}")
    
    def submit_batch(
        self,
        function_name: str,
        args_list: List[tuple]
    ) -> Tuple[bool, List[str], str]:
        """
        Submit a batch of tasks.
        
        Args:
            function_name: Name of the function to execute
            args_list: List of argument tuples
        
        Returns:
            Tuple of (success, task_ids list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Dask")
        
        if not function_name:
            return (False, [], "function_name cannot be empty")
        
        if not args_list:
            return (False, [], "args_list cannot be empty")
        
        task_ids = []
        
        for args in args_list:
            success, task_id, msg = self.submit(function_name, *args)
            if success and task_id:
                task_ids.append(task_id)
        
        return (True, task_ids, f"Submitted {len(task_ids)} tasks")
    
    def map(
        self,
        function_name: str,
        iterables: List[Any]
    ) -> Tuple[bool, List[str], str]:
        """
        Map a function over iterables.
        
        Args:
            function_name: Name of the function to execute
            iterables: List of items to map over
        
        Returns:
            Tuple of (success, task_ids list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Dask")
        
        if not function_name:
            return (False, [], "function_name cannot be empty")
        
        if not iterables:
            return (False, [], "iterables cannot be empty")
        
        args_list = [(item,) for item in iterables]
        
        return self.submit_batch(function_name, args_list)
    
    # Result gathering
    
    def get_result(
        self,
        task_id: str,
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, Any, str]:
        """
        Get result of a task.
        
        Args:
            task_id: Task identifier
            timeout_seconds: Timeout in seconds (uses default if None)
        
        Returns:
            Tuple of (success, result, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Dask")
        
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if task_id not in self.tasks:
            return (False, None, f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.FINISHED:
            return (True, task.result, "Result retrieved")
        
        if task.status == TaskStatus.ERRED:
            return (False, None, f"Task failed: {task.error_message}")
        
        if task.status == TaskStatus.CANCELLED:
            return (False, None, "Task was cancelled")
        
        # Wait for completion
        if timeout_seconds is None:
            timeout_seconds = self.config.timeout_seconds
        
        success, final_task, msg = self.wait_for_task(task_id, timeout_seconds)
        
        if not success:
            return (False, None, msg)
        
        if final_task and final_task.status == TaskStatus.FINISHED:
            return (True, final_task.result, "Result retrieved after waiting")
        
        return (False, None, "Task did not complete successfully")
    
    def gather(
        self,
        task_ids: List[str],
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, List[Any], str]:
        """
        Gather results from multiple tasks.
        
        Args:
            task_ids: List of task identifiers
            timeout_seconds: Timeout in seconds
        
        Returns:
            Tuple of (success, results list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Dask")
        
        if not task_ids:
            return (False, [], "task_ids cannot be empty")
        
        results = []
        failed_count = 0
        
        for task_id in task_ids:
            success, result, msg = self.get_result(task_id, timeout_seconds)
            
            if success:
                results.append(result)
            else:
                failed_count += 1
                results.append(None)
        
        if failed_count > 0:
            return (False, results, f"Failed to gather {failed_count}/{len(task_ids)} results")
        
        return (True, results, f"Gathered {len(results)} results")
    
    # Task management
    
    def get_task(self, task_id: str) -> Tuple[bool, Optional[DaskTask], str]:
        """
        Get task details.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, task or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Dask")
        
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if task_id not in self.tasks:
            return (False, None, f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        return (True, task, "Task retrieved")
    
    def cancel_task(self, task_id: str) -> Tuple[bool, str]:
        """
        Cancel a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Dask")
        
        if not task_id:
            return (False, "task_id cannot be empty")
        
        if task_id not in self.tasks:
            return (False, f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        if task.is_terminal():
            return (False, f"Task is already in terminal state: {task.status.value}")
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        
        return (True, f"Task {task_id} cancelled")
    
    def wait_for_task(
        self,
        task_id: str,
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, Optional[DaskTask], str]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task identifier
            timeout_seconds: Timeout in seconds
        
        Returns:
            Tuple of (success, task or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Dask")
        
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if task_id not in self.tasks:
            return (False, None, f"Task {task_id} not found")
        
        if timeout_seconds is None:
            timeout_seconds = self.config.timeout_seconds
        
        start_time = time.time()
        task = self.tasks[task_id]
        
        while not task.is_terminal():
            elapsed = time.time() - start_time
            
            if elapsed >= timeout_seconds:
                return (False, task, f"Timeout after {elapsed:.1f} seconds")
            
            # Simulate waiting
            time.sleep(0.1)
        
        return (True, task, f"Task completed with status: {task.status.value}")
    
    # Cluster management
    
    def get_workers(self) -> Tuple[bool, List[Worker], str]:
        """
        Get list of workers.
        
        Returns:
            Tuple of (success, workers list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Dask")
        
        workers = list(self.workers.values())
        
        return (True, workers, f"Retrieved {len(workers)} workers")
    
    def get_worker(self, worker_id: str) -> Tuple[bool, Optional[Worker], str]:
        """
        Get worker details.
        
        Args:
            worker_id: Worker identifier
        
        Returns:
            Tuple of (success, worker or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Dask")
        
        if not worker_id:
            return (False, None, "worker_id cannot be empty")
        
        if worker_id not in self.workers:
            return (False, None, f"Worker {worker_id} not found")
        
        worker = self.workers[worker_id]
        
        return (True, worker, "Worker retrieved")
    
    def _generate_task_id(self, function_name: str) -> str:
        """Generate unique task ID."""
        self.task_counter += 1
        timestamp = int(time.time())
        
        # Create hash of function name and counter
        content = f"{function_name}_{timestamp}_{self.task_counter}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return f"task_{hash_digest}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with stats
        """
        # Count tasks by status
        status_counts = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            status_counts[task.status.value] += 1
        
        # Worker stats
        total_cores = sum(w.n_cores for w in self.workers.values())
        total_memory = sum(w.memory_limit for w in self.workers.values())
        used_memory = sum(w.memory_used for w in self.workers.values())
        
        return {
            "connected": self.connected,
            "cluster_started": self.cluster_started,
            "total_workers": len(self.workers),
            "total_cores": total_cores,
            "total_memory_gb": total_memory / (1024**3),
            "used_memory_gb": used_memory / (1024**3),
            "total_tasks": len(self.tasks),
            "tasks_submitted": self.tasks_submitted,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_by_status": status_counts,
            "uptime_seconds": time.time() - self.connection_time if self.connected else 0
        }
    
    def clear_completed_tasks(
        self,
        older_than_seconds: float = 3600.0
    ) -> Tuple[bool, int, str]:
        """
        Clear completed tasks older than specified time.
        
        Args:
            older_than_seconds: Age threshold in seconds
        
        Returns:
            Tuple of (success, count_removed, message)
        """
        if not self.connected:
            return (False, 0, "Not connected to Dask")
        
        if older_than_seconds <= 0:
            return (False, 0, "older_than_seconds must be positive")
        
        current_time = time.time()
        cutoff_time = current_time - older_than_seconds
        
        # Find old completed tasks
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.is_terminal() and task.completed_at < cutoff_time
        ]
        
        # Remove them
        for task_id in to_remove:
            self.tasks.pop(task_id)
        
        return (True, len(to_remove), f"Removed {len(to_remove)} old tasks")
