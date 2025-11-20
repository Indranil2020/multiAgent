"""
Prefect client implementation.

This module provides a comprehensive Prefect workflow orchestration client wrapper
for submitting, monitoring, and managing distributed workflows with explicit error handling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import time
import json


class FlowRunState(Enum):
    """States of a flow run."""
    SCHEDULED = "scheduled"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CRASHED = "crashed"


class TaskRunState(Enum):
    """States of a task run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class PrefectConfig:
    """
    Configuration for Prefect client.
    
    Attributes:
        api_url: Prefect API URL
        api_key: Optional API key for authentication
        workspace: Workspace name
        project_name: Project name
        default_work_queue: Default work queue name
        timeout_seconds: Default timeout for operations
    """
    api_url: str
    api_key: str = ""
    workspace: str = "default"
    project_name: str = "default"
    default_work_queue: str = "default"
    timeout_seconds: float = 300.0
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if not self.api_url:
            return (False, "api_url cannot be empty")
        
        if not self.workspace:
            return (False, "workspace cannot be empty")
        
        if not self.project_name:
            return (False, "project_name cannot be empty")
        
        if not self.default_work_queue:
            return (False, "default_work_queue cannot be empty")
        
        if self.timeout_seconds <= 0:
            return (False, "timeout_seconds must be positive")
        
        return (True, "")


@dataclass
class FlowRun:
    """
    A Prefect flow run.
    
    Attributes:
        flow_run_id: Unique flow run identifier
        flow_name: Name of the flow
        state: Current state
        parameters: Flow parameters
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        error_message: Error message if failed
        tags: Flow run tags
    """
    flow_run_id: str
    flow_name: str
    state: FlowRunState
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""
    tags: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if flow run is valid."""
        return bool(self.flow_run_id and self.flow_name)
    
    def is_terminal(self) -> bool:
        """Check if flow run is in a terminal state."""
        return self.state in {
            FlowRunState.COMPLETED,
            FlowRunState.FAILED,
            FlowRunState.CANCELLED,
            FlowRunState.CRASHED
        }
    
    def duration_seconds(self) -> float:
        """Get duration of flow run."""
        if self.started_at == 0:
            return 0.0
        
        end_time = self.completed_at if self.completed_at > 0 else time.time()
        return end_time - self.started_at


@dataclass
class TaskRun:
    """
    A Prefect task run within a flow.
    
    Attributes:
        task_run_id: Unique task run identifier
        task_name: Name of the task
        flow_run_id: Parent flow run ID
        state: Current state
        result: Task result
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        error_message: Error message if failed
    """
    task_run_id: str
    task_name: str
    flow_run_id: str
    state: TaskRunState
    result: Any = None
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    error_message: str = ""
    
    def is_valid(self) -> bool:
        """Check if task run is valid."""
        return bool(self.task_run_id and self.task_name and self.flow_run_id)
    
    def is_terminal(self) -> bool:
        """Check if task run is in a terminal state."""
        return self.state in {
            TaskRunState.COMPLETED,
            TaskRunState.FAILED,
            TaskRunState.SKIPPED,
            TaskRunState.CACHED
        }


class PrefectClient:
    """
    Prefect workflow orchestration client wrapper.
    
    Provides workflow submission, monitoring, and management capabilities
    with explicit error handling and state tracking.
    
    Note: This is a mock implementation for the zero-error architecture.
    In production, this would wrap the actual Prefect client library.
    """
    
    def __init__(self, config: PrefectConfig):
        """
        Initialize Prefect client.
        
        Args:
            config: Prefect configuration
        """
        self.config = config
        self.connected = False
        
        # Mock storage
        self.flow_runs: Dict[str, FlowRun] = {}
        self.task_runs: Dict[str, List[TaskRun]] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.flows_submitted = 0
        self.flows_completed = 0
        self.flows_failed = 0
        self.connection_time = 0.0
        self.flow_run_counter = 0
        self.task_run_counter = 0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate client configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def connect(self) -> Tuple[bool, str]:
        """
        Connect to Prefect API.
        
        Returns:
            Tuple of (success, message)
        """
        # Validate config first
        is_valid, error_msg = self.validate_config()
        if not is_valid:
            return (False, f"Invalid config: {error_msg}")
        
        # Simulate connection
        self.connected = True
        self.connection_time = time.time()
        
        return (True, f"Connected to Prefect at {self.config.api_url}")
    
    def disconnect(self) -> Tuple[bool, str]:
        """
        Disconnect from Prefect API.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected")
        
        self.connected = False
        
        return (True, "Disconnected from Prefect")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected
    
    # Flow operations
    
    def submit_flow(
        self,
        flow_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        work_queue: Optional[str] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Submit a flow for execution.
        
        Args:
            flow_name: Name of the flow to run
            parameters: Flow parameters
            tags: Optional tags
            work_queue: Work queue name (uses default if None)
        
        Returns:
            Tuple of (success, flow_run_id or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Prefect")
        
        if not flow_name:
            return (False, None, "flow_name cannot be empty")
        
        # Generate flow run ID
        flow_run_id = self._generate_flow_run_id(flow_name)
        
        # Create flow run
        current_time = time.time()
        flow_run = FlowRun(
            flow_run_id=flow_run_id,
            flow_name=flow_name,
            state=FlowRunState.SCHEDULED,
            parameters=parameters or {},
            created_at=current_time,
            tags=tags or []
        )
        
        if not flow_run.is_valid():
            return (False, None, "Invalid flow run created")
        
        # Store flow run
        self.flow_runs[flow_run_id] = flow_run
        self.flows_submitted += 1
        
        # Initialize task runs list
        self.task_runs[flow_run_id] = []
        
        queue = work_queue or self.config.default_work_queue
        
        return (True, flow_run_id, f"Flow submitted to queue '{queue}'")
    
    def get_flow_run(
        self,
        flow_run_id: str
    ) -> Tuple[bool, Optional[FlowRun], str]:
        """
        Get flow run details.
        
        Args:
            flow_run_id: Flow run identifier
        
        Returns:
            Tuple of (success, flow_run or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, None, "flow_run_id cannot be empty")
        
        if flow_run_id not in self.flow_runs:
            return (False, None, f"Flow run {flow_run_id} not found")
        
        flow_run = self.flow_runs[flow_run_id]
        
        return (True, flow_run, "Flow run retrieved")
    
    def update_flow_run_state(
        self,
        flow_run_id: str,
        state: FlowRunState,
        error_message: str = ""
    ) -> Tuple[bool, str]:
        """
        Update flow run state.
        
        Args:
            flow_run_id: Flow run identifier
            state: New state
            error_message: Error message if failed
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, "flow_run_id cannot be empty")
        
        if flow_run_id not in self.flow_runs:
            return (False, f"Flow run {flow_run_id} not found")
        
        flow_run = self.flow_runs[flow_run_id]
        old_state = flow_run.state
        flow_run.state = state
        
        current_time = time.time()
        
        # Update timestamps based on state
        if state == FlowRunState.RUNNING and flow_run.started_at == 0:
            flow_run.started_at = current_time
        
        if state in {FlowRunState.COMPLETED, FlowRunState.FAILED, FlowRunState.CANCELLED, FlowRunState.CRASHED}:
            if flow_run.completed_at == 0:
                flow_run.completed_at = current_time
            
            if state == FlowRunState.COMPLETED:
                self.flows_completed += 1
            elif state == FlowRunState.FAILED:
                self.flows_failed += 1
                flow_run.error_message = error_message
        
        return (True, f"Flow run state updated: {old_state.value} -> {state.value}")
    
    def cancel_flow_run(self, flow_run_id: str) -> Tuple[bool, str]:
        """
        Cancel a flow run.
        
        Args:
            flow_run_id: Flow run identifier
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, "flow_run_id cannot be empty")
        
        if flow_run_id not in self.flow_runs:
            return (False, f"Flow run {flow_run_id} not found")
        
        flow_run = self.flow_runs[flow_run_id]
        
        if flow_run.is_terminal():
            return (False, f"Flow run is already in terminal state: {flow_run.state.value}")
        
        return self.update_flow_run_state(flow_run_id, FlowRunState.CANCELLED)
    
    def wait_for_flow_run(
        self,
        flow_run_id: str,
        timeout_seconds: Optional[float] = None,
        poll_interval_seconds: float = 1.0
    ) -> Tuple[bool, Optional[FlowRun], str]:
        """
        Wait for a flow run to complete.
        
        Args:
            flow_run_id: Flow run identifier
            timeout_seconds: Timeout in seconds (uses default if None)
            poll_interval_seconds: Polling interval
        
        Returns:
            Tuple of (success, flow_run or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, None, "flow_run_id cannot be empty")
        
        if flow_run_id not in self.flow_runs:
            return (False, None, f"Flow run {flow_run_id} not found")
        
        if timeout_seconds is None:
            timeout_seconds = self.config.timeout_seconds
        
        if poll_interval_seconds <= 0:
            return (False, None, "poll_interval_seconds must be positive")
        
        start_time = time.time()
        
        while True:
            flow_run = self.flow_runs[flow_run_id]
            
            if flow_run.is_terminal():
                return (True, flow_run, f"Flow run completed with state: {flow_run.state.value}")
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout_seconds:
                return (False, flow_run, f"Timeout after {elapsed:.1f} seconds")
            
            # Simulate polling delay
            time.sleep(min(poll_interval_seconds, timeout_seconds - elapsed))
    
    # Task operations
    
    def add_task_run(
        self,
        flow_run_id: str,
        task_name: str,
        state: TaskRunState = TaskRunState.PENDING
    ) -> Tuple[bool, Optional[str], str]:
        """
        Add a task run to a flow run.
        
        Args:
            flow_run_id: Parent flow run ID
            task_name: Task name
            state: Initial state
        
        Returns:
            Tuple of (success, task_run_id or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, None, "flow_run_id cannot be empty")
        
        if not task_name:
            return (False, None, "task_name cannot be empty")
        
        if flow_run_id not in self.flow_runs:
            return (False, None, f"Flow run {flow_run_id} not found")
        
        # Generate task run ID
        task_run_id = self._generate_task_run_id(task_name)
        
        # Create task run
        current_time = time.time()
        task_run = TaskRun(
            task_run_id=task_run_id,
            task_name=task_name,
            flow_run_id=flow_run_id,
            state=state,
            created_at=current_time
        )
        
        if not task_run.is_valid():
            return (False, None, "Invalid task run created")
        
        # Add to flow run's task list
        self.task_runs[flow_run_id].append(task_run)
        
        return (True, task_run_id, f"Task run added to flow {flow_run_id}")
    
    def get_task_runs(
        self,
        flow_run_id: str
    ) -> Tuple[bool, List[TaskRun], str]:
        """
        Get all task runs for a flow run.
        
        Args:
            flow_run_id: Flow run identifier
        
        Returns:
            Tuple of (success, task_runs list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Prefect")
        
        if not flow_run_id:
            return (False, [], "flow_run_id cannot be empty")
        
        if flow_run_id not in self.task_runs:
            return (False, [], f"Flow run {flow_run_id} not found")
        
        task_runs = self.task_runs[flow_run_id]
        
        return (True, task_runs, f"Retrieved {len(task_runs)} task runs")
    
    # Query operations
    
    def list_flow_runs(
        self,
        flow_name: Optional[str] = None,
        state: Optional[FlowRunState] = None,
        limit: int = 100
    ) -> Tuple[bool, List[FlowRun], str]:
        """
        List flow runs with optional filters.
        
        Args:
            flow_name: Filter by flow name
            state: Filter by state
            limit: Maximum number of results
        
        Returns:
            Tuple of (success, flow_runs list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Prefect")
        
        if limit <= 0:
            return (False, [], "limit must be positive")
        
        # Filter flow runs
        filtered = list(self.flow_runs.values())
        
        if flow_name:
            filtered = [fr for fr in filtered if fr.flow_name == flow_name]
        
        if state:
            filtered = [fr for fr in filtered if fr.state == state]
        
        # Sort by creation time (newest first)
        filtered.sort(key=lambda fr: fr.created_at, reverse=True)
        
        # Apply limit
        filtered = filtered[:limit]
        
        return (True, filtered, f"Found {len(filtered)} flow runs")
    
    def _generate_flow_run_id(self, flow_name: str) -> str:
        """Generate unique flow run ID."""
        self.flow_run_counter += 1
        timestamp = int(time.time())
        return f"flow_{flow_name}_{timestamp}_{self.flow_run_counter}"
    
    def _generate_task_run_id(self, task_name: str) -> str:
        """Generate unique task run ID."""
        self.task_run_counter += 1
        timestamp = int(time.time())
        return f"task_{task_name}_{timestamp}_{self.task_run_counter}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with stats
        """
        total_task_runs = sum(len(tasks) for tasks in self.task_runs.values())
        
        # Count by state
        state_counts = {state.value: 0 for state in FlowRunState}
        for flow_run in self.flow_runs.values():
            state_counts[flow_run.state.value] += 1
        
        return {
            "connected": self.connected,
            "workspace": self.config.workspace,
            "project": self.config.project_name,
            "total_flow_runs": len(self.flow_runs),
            "total_task_runs": total_task_runs,
            "flows_submitted": self.flows_submitted,
            "flows_completed": self.flows_completed,
            "flows_failed": self.flows_failed,
            "flow_runs_by_state": state_counts,
            "uptime_seconds": time.time() - self.connection_time if self.connected else 0
        }
    
    def clear_completed_runs(
        self,
        older_than_seconds: float = 3600.0
    ) -> Tuple[bool, int, str]:
        """
        Clear completed flow runs older than specified time.
        
        Args:
            older_than_seconds: Age threshold in seconds
        
        Returns:
            Tuple of (success, count_removed, message)
        """
        if not self.connected:
            return (False, 0, "Not connected to Prefect")
        
        if older_than_seconds <= 0:
            return (False, 0, "older_than_seconds must be positive")
        
        current_time = time.time()
        cutoff_time = current_time - older_than_seconds
        
        # Find old completed runs
        to_remove = [
            flow_run_id for flow_run_id, flow_run in self.flow_runs.items()
            if flow_run.is_terminal() and flow_run.completed_at < cutoff_time
        ]
        
        # Remove them
        for flow_run_id in to_remove:
            self.flow_runs.pop(flow_run_id)
            self.task_runs.pop(flow_run_id, None)
        
        return (True, len(to_remove), f"Removed {len(to_remove)} old flow runs")
