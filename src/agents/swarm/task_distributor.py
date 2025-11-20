"""
Task Distributor for Agent Swarm.

This module distributes tasks to available agents based on task type,
priority, and current system load. Implements fair distribution and
priority-based scheduling.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskID, TaskType, PriorityLevel, TaskStatus
from ...core.voting.types import AgentConfig


@dataclass
class TaskAssignment:
    """
    Represents a task assignment to an agent.

    Attributes:
        task_id: ID of assigned task
        task_spec: Full task specification
        agent_config: Configuration for agent execution
        assigned_at: Timestamp of assignment
        deadline: Optional deadline timestamp
        attempts: Number of execution attempts
    """
    task_id: TaskID
    task_spec: TaskSpecification
    agent_config: AgentConfig
    assigned_at: float = field(default_factory=lambda: datetime.now().timestamp())
    deadline: Optional[float] = None
    attempts: int = 0

    def is_overdue(self) -> bool:
        """Check if assignment is past deadline."""
        if self.deadline is None:
            return False
        return datetime.now().timestamp() > self.deadline


@dataclass
class DistributionStats:
    """
    Statistics for task distribution.

    Attributes:
        total_tasks_distributed: Total tasks distributed
        tasks_by_type: Distribution count by task type
        tasks_by_priority: Distribution count by priority
        average_queue_time_ms: Average time tasks wait in queue
    """
    total_tasks_distributed: int = 0
    tasks_by_type: Dict[str, int] = field(default_factory=dict)
    tasks_by_priority: Dict[int, int] = field(default_factory=dict)
    average_queue_time_ms: float = 0.0


class TaskDistributor:
    """
    Distributes tasks to agents in the swarm.

    Implements priority-based scheduling with fair distribution across
    agent types. Manages task queues and assignment tracking.

    Design:
    - Priority-based task scheduling
    - Fair distribution across task types
    - Deadline awareness
    - Load balancing
    """

    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize task distributor.

        Args:
            max_concurrent_tasks: Maximum concurrent task assignments
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_queue: List[TaskSpecification] = []
        self.active_assignments: Dict[TaskID, TaskAssignment] = {}
        self.stats = DistributionStats()

    def enqueue_task(self, task: TaskSpecification) -> bool:
        """
        Add task to distribution queue.

        Args:
            task: Task to enqueue

        Returns:
            True if task was enqueued successfully
        """
        if task is None:
            return False

        if not task.is_valid():
            return False

        # Check if task is already queued or assigned
        if self._is_task_queued(task.id):
            return False

        if task.id in self.active_assignments:
            return False

        # Add to queue
        self.task_queue.append(task)

        # Sort queue by priority (higher priority first)
        self._sort_queue()

        return True

    def get_next_task(
        self,
        agent_config: AgentConfig,
        task_type_filter: Optional[TaskType] = None
    ) -> Optional[TaskAssignment]:
        """
        Get next task assignment for an agent.

        Args:
            agent_config: Configuration for agent
            task_type_filter: Optional filter for task type

        Returns:
            TaskAssignment or None if no tasks available
        """
        # Check if we're at capacity
        if len(self.active_assignments) >= self.max_concurrent_tasks:
            return None

        # Find next suitable task
        task = self._find_next_task(task_type_filter)
        if task is None:
            return None

        # Remove from queue
        self.task_queue.remove(task)

        # Create assignment
        assignment = TaskAssignment(
            task_id=task.id,
            task_spec=task,
            agent_config=agent_config
        )

        # Add to active assignments
        self.active_assignments[task.id] = assignment

        # Update stats
        self._record_distribution(task)

        return assignment

    def complete_task(self, task_id: TaskID) -> bool:
        """
        Mark task as completed.

        Args:
            task_id: ID of completed task

        Returns:
            True if task was removed from active assignments
        """
        if task_id not in self.active_assignments:
            return False

        del self.active_assignments[task_id]
        return True

    def retry_task(self, task_id: TaskID) -> bool:
        """
        Retry a failed task.

        Args:
            task_id: ID of task to retry

        Returns:
            True if task was re-queued
        """
        if task_id not in self.active_assignments:
            return False

        assignment = self.active_assignments[task_id]
        assignment.attempts += 1

        # Check if we should give up
        if assignment.attempts >= 3:  # Max 3 attempts
            del self.active_assignments[task_id]
            return False

        # Re-queue task
        task = assignment.task_spec
        del self.active_assignments[task_id]

        return self.enqueue_task(task)

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.

        Returns:
            Status dictionary
        """
        return {
            'queued_tasks': len(self.task_queue),
            'active_assignments': len(self.active_assignments),
            'capacity_used': len(self.active_assignments) / self.max_concurrent_tasks,
            'queue_by_type': self._count_by_type(self.task_queue),
            'queue_by_priority': self._count_by_priority(self.task_queue)
        }

    def get_stats(self) -> DistributionStats:
        """Get distribution statistics."""
        return self.stats

    def _find_next_task(
        self,
        task_type_filter: Optional[TaskType] = None
    ) -> Optional[TaskSpecification]:
        """
        Find next task to assign.

        Args:
            task_type_filter: Optional task type filter

        Returns:
            Next task or None
        """
        if not self.task_queue:
            return None

        # Queue is already sorted by priority
        # Find first task matching filter
        for task in self.task_queue:
            if task_type_filter is None or task.task_type == task_type_filter:
                return task

        return None

    def _is_task_queued(self, task_id: TaskID) -> bool:
        """Check if task is in queue."""
        return any(task.id == task_id for task in self.task_queue)

    def _sort_queue(self) -> None:
        """Sort queue by priority (higher first)."""
        self.task_queue.sort(
            key=lambda task: task.priority.value,
            reverse=True
        )

    def _count_by_type(self, tasks: List[TaskSpecification]) -> Dict[str, int]:
        """Count tasks by type."""
        counts: Dict[str, int] = {}
        for task in tasks:
            type_name = task.task_type.value
            if type_name not in counts:
                counts[type_name] = 0
            counts[type_name] += 1
        return counts

    def _count_by_priority(self, tasks: List[TaskSpecification]) -> Dict[int, int]:
        """Count tasks by priority."""
        counts: Dict[int, int] = {}
        for task in tasks:
            priority = task.priority.value
            if priority not in counts:
                counts[priority] = 0
            counts[priority] += 1
        return counts

    def _record_distribution(self, task: TaskSpecification) -> None:
        """Record task distribution in stats."""
        self.stats.total_tasks_distributed += 1

        # Record by type
        type_name = task.task_type.value
        if type_name not in self.stats.tasks_by_type:
            self.stats.tasks_by_type[type_name] = 0
        self.stats.tasks_by_type[type_name] += 1

        # Record by priority
        priority = task.priority.value
        if priority not in self.stats.tasks_by_priority:
            self.stats.tasks_by_priority[priority] = 0
        self.stats.tasks_by_priority[priority] += 1
