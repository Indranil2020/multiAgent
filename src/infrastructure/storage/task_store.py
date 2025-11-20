"""
Task storage implementation.

This module provides comprehensive persistent storage for task specifications,
enabling task retrieval, querying, status management, priority handling,
and detailed analytics with thread-safe operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import time
import threading


class TaskStatus(Enum):
    """Status of a stored task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StoredTask:
    """
    A task stored in the task store.
    
    Attributes:
        task_id: Unique task identifier
        name: Task name
        description: Task description
        task_type: Type of task
        status: Current status
        priority: Task priority
        created_at: Creation timestamp
        updated_at: Last update timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        metadata: Additional metadata
        tags: Task tags
        dependencies: List of dependent task IDs
        retry_count: Number of retries
        max_retries: Maximum retry attempts
    """
    task_id: str
    name: str
    description: str
    task_type: str
    status: TaskStatus
    priority: TaskPriority
    created_at: float
    updated_at: float
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def is_valid(self) -> bool:
        """Check if stored task is valid."""
        return bool(
            self.task_id and
            self.name and
            self.description and
            self.task_type
        )
    
    def is_terminal(self) -> bool:
        """Check if task is in terminal state."""
        return self.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        if self.started_at == 0:
            return 0.0
        
        end_time = self.completed_at if self.completed_at > 0 else time.time()
        return end_time - self.started_at
    
    def age_seconds(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_at
    
    def has_tag(self, tag: str) -> bool:
        """Check if task has a specific tag."""
        return tag in self.tags


class TaskStore:
    """
    Comprehensive task storage with querying and analytics.
    
    Provides thread-safe storage and retrieval of task specifications
    with advanced querying capabilities, priority management, and
    detailed statistics collection.
    """
    
    def __init__(self, max_tasks: int = 100000):
        """
        Initialize task store.
        
        Args:
            max_tasks: Maximum number of tasks to store
        """
        self.max_tasks = max_tasks
        self.tasks: Dict[str, StoredTask] = {}
        self.task_counter = 0
        
        # Indices for efficient querying
        self.status_index: Dict[TaskStatus, Set[str]] = {status: set() for status in TaskStatus}
        self.type_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.priority_index: Dict[TaskPriority, Set[str]] = {priority: set() for priority in TaskPriority}
        
        # Thread safety
        self.lock = threading.Lock()
    
    def save_task(
        self,
        task_id: str,
        name: str,
        description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        Save a task to the store.
        
        Args:
            task_id: Unique task identifier
            name: Task name
            description: Task description
            task_type: Type of task
            priority: Task priority
            metadata: Optional metadata
            tags: Optional tags
            dependencies: Optional dependencies
            max_retries: Maximum retry attempts
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not task_id:
                return (False, "task_id cannot be empty")
            
            if not name:
                return (False, "name cannot be empty")
            
            if not description:
                return (False, "description cannot be empty")
            
            if not task_type:
                return (False, "task_type cannot be empty")
            
            if max_retries < 0:
                return (False, "max_retries cannot be negative")
            
            # Check task limit
            if task_id not in self.tasks and len(self.tasks) >= self.max_tasks:
                return (False, f"Maximum tasks ({self.max_tasks}) reached")
            
            current_time = time.time()
            
            # Check if task already exists
            if task_id in self.tasks:
                # Update existing task
                task = self.tasks[task_id]
                
                # Remove from old indices
                self._remove_from_indices(task)
                
                task.name = name
                task.description = description
                task.task_type = task_type
                task.priority = priority
                task.updated_at = current_time
                task.max_retries = max_retries
                
                if metadata:
                    task.metadata.update(metadata)
                
                if tags:
                    task.tags = tags
                
                if dependencies:
                    task.dependencies = dependencies
                
                # Add to new indices
                self._add_to_indices(task)
                
                return (True, f"Task {task_id} updated")
            
            # Create new task
            task = StoredTask(
                task_id=task_id,
                name=name,
                description=description,
                task_type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                created_at=current_time,
                updated_at=current_time,
                metadata=metadata or {},
                tags=tags or [],
                dependencies=dependencies or [],
                max_retries=max_retries
            )
            
            if not task.is_valid():
                return (False, "Invalid task created")
            
            self.tasks[task_id] = task
            self.task_counter += 1
            
            # Add to indices
            self._add_to_indices(task)
            
            return (True, f"Task {task_id} saved")
    
    def get_task(self, task_id: str) -> Tuple[bool, Optional[StoredTask], str]:
        """
        Retrieve a task by ID.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, task or None, message)
        """
        with self.lock:
            if not task_id:
                return (False, None, "task_id cannot be empty")
            
            if task_id not in self.tasks:
                return (False, None, f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            return (True, task, "Task retrieved")
    
    def update_status(
        self,
        task_id: str,
        status: TaskStatus
    ) -> Tuple[bool, str]:
        """
        Update task status.
        
        Args:
            task_id: Task identifier
            status: New status
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not task_id:
                return (False, "task_id cannot be empty")
            
            if task_id not in self.tasks:
                return (False, f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            old_status = task.status
            
            # Remove from old status index
            self.status_index[old_status].discard(task_id)
            
            # Update status
            task.status = status
            task.updated_at = time.time()
            
            # Update timestamps based on status
            if status == TaskStatus.RUNNING and task.started_at == 0:
                task.started_at = time.time()
            
            if status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                if task.completed_at == 0:
                    task.completed_at = time.time()
            
            # Add to new status index
            self.status_index[status].add(task_id)
            
            return (True, f"Task {task_id} status updated to {status.value}")
    
    def increment_retry(self, task_id: str) -> Tuple[bool, str]:
        """
        Increment retry count for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not task_id:
                return (False, "task_id cannot be empty")
            
            if task_id not in self.tasks:
                return (False, f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            task.retry_count += 1
            task.updated_at = time.time()
            
            return (True, f"Retry count: {task.retry_count}/{task.max_retries}")
    
    def can_retry(self, task_id: str) -> Tuple[bool, bool, str]:
        """
        Check if task can be retried.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, can_retry, message)
        """
        with self.lock:
            if not task_id:
                return (False, False, "task_id cannot be empty")
            
            if task_id not in self.tasks:
                return (False, False, f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            can_retry = task.retry_count < task.max_retries
            
            return (True, can_retry, f"Retries: {task.retry_count}/{task.max_retries}")
    
    def query_by_status(self, status: TaskStatus) -> List[StoredTask]:
        """
        Query tasks by status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of matching tasks
        """
        with self.lock:
            task_ids = self.status_index.get(status, set())
            return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
    
    def query_by_type(self, task_type: str) -> List[StoredTask]:
        """
        Query tasks by type.
        
        Args:
            task_type: Type to filter by
        
        Returns:
            List of matching tasks
        """
        with self.lock:
            if not task_type:
                return []
            
            task_ids = self.type_index.get(task_type, set())
            return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
    
    def query_by_tag(self, tag: str) -> List[StoredTask]:
        """
        Query tasks by tag.
        
        Args:
            tag: Tag to filter by
        
        Returns:
            List of matching tasks
        """
        with self.lock:
            if not tag:
                return []
            
            task_ids = self.tag_index.get(tag, set())
            return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
    
    def query_by_priority(self, priority: TaskPriority) -> List[StoredTask]:
        """
        Query tasks by priority.
        
        Args:
            priority: Priority to filter by
        
        Returns:
            List of matching tasks
        """
        with self.lock:
            task_ids = self.priority_index.get(priority, set())
            return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
    
    def get_pending_tasks(self, limit: int = 100) -> List[StoredTask]:
        """
        Get pending tasks sorted by priority.
        
        Args:
            limit: Maximum tasks to return
        
        Returns:
            List of pending tasks
        """
        pending = self.query_by_status(TaskStatus.PENDING)
        
        # Sort by priority (highest first) then by creation time
        pending.sort(key=lambda t: (-t.priority.value, t.created_at))
        
        return pending[:limit]
    
    def delete_task(self, task_id: str) -> Tuple[bool, str]:
        """
        Delete a task from the store.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not task_id:
                return (False, "task_id cannot be empty")
            
            if task_id not in self.tasks:
                return (False, f"Task {task_id} not found")
            
            task = self.tasks.pop(task_id)
            
            # Remove from indices
            self._remove_from_indices(task)
            
            return (True, f"Task {task_id} deleted")
    
    def _add_to_indices(self, task: StoredTask) -> None:
        """Add task to all indices."""
        # Status index
        self.status_index[task.status].add(task.task_id)
        
        # Type index
        if task.task_type not in self.type_index:
            self.type_index[task.task_type] = set()
        self.type_index[task.task_type].add(task.task_id)
        
        # Tag index
        for tag in task.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(task.task_id)
        
        # Priority index
        self.priority_index[task.priority].add(task.task_id)
    
    def _remove_from_indices(self, task: StoredTask) -> None:
        """Remove task from all indices."""
        # Status index
        self.status_index[task.status].discard(task.task_id)
        
        # Type index
        if task.task_type in self.type_index:
            self.type_index[task.task_type].discard(task.task_id)
        
        # Tag index
        for tag in task.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(task.task_id)
        
        # Priority index
        self.priority_index[task.priority].discard(task.task_id)
    
    def get_all_tasks(self) -> List[StoredTask]:
        """Get all tasks in the store."""
        with self.lock:
            return list(self.tasks.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive task store statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            status_counts = {
                status.value: len(self.status_index[status])
                for status in TaskStatus
            }
            
            priority_counts = {
                priority.value: len(self.priority_index[priority])
                for priority in TaskPriority
            }
            
            # Calculate average duration for completed tasks
            completed_tasks = self.query_by_status(TaskStatus.COMPLETED)
            avg_duration = 0.0
            if completed_tasks:
                avg_duration = sum(t.duration_seconds() for t in completed_tasks) / len(completed_tasks)
            
            return {
                "total_tasks": len(self.tasks),
                "max_tasks": self.max_tasks,
                "tasks_by_status": status_counts,
                "tasks_by_priority": priority_counts,
                "total_types": len(self.type_index),
                "total_tags": len(self.tag_index),
                "average_duration_seconds": f"{avg_duration:.2f}"
            }
    
    def clear(self) -> None:
        """Clear all tasks from the store."""
        with self.lock:
            self.tasks.clear()
            self.task_counter = 0
            
            # Clear indices
            for status in TaskStatus:
                self.status_index[status].clear()
            
            self.type_index.clear()
            self.tag_index.clear()
            
            for priority in TaskPriority:
                self.priority_index[priority].clear()
