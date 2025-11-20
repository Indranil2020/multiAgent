"""
Task Management API Routes.

This module provides REST endpoints for task submission, status tracking,
and result retrieval in the zero-error system.

Endpoints:
- POST   /tasks          - Submit a new task
- GET    /tasks          - List all tasks
- GET    /tasks/{id}     - Get task details
- GET    /tasks/{id}/status - Get task status
- GET    /tasks/{id}/result - Get task result
- DELETE /tasks/{id}     - Cancel a task
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TaskSubmissionRequest:
    """
    Request to submit a new task.

    Attributes:
        name: Task name
        description: Task description
        task_type: Type of task
        inputs: Input parameters
        outputs: Expected outputs
        max_lines: Maximum lines of code
        max_complexity: Maximum cyclomatic complexity
        priority: Task priority (1-5)
        verification_level: Verification strictness
    """
    name: str
    description: str
    task_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    max_lines: int = 20
    max_complexity: int = 10
    priority: int = 3
    verification_level: str = "standard"

    def validate(self) -> bool:
        """Validate request."""
        if not self.name or len(self.name) < 3:
            return False

        if not self.description or len(self.description) < 10:
            return False

        if not self.task_type:
            return False

        if not (1 <= self.priority <= 5):
            return False

        if not (5 <= self.max_lines <= 100):
            return False

        if not (1 <= self.max_complexity <= 20):
            return False

        return True


@dataclass
class TaskResponse:
    """
    Response containing task information.

    Attributes:
        task_id: Unique task identifier
        name: Task name
        status: Current status
        created_at: Creation timestamp
        updated_at: Last update timestamp
        priority: Task priority
    """
    task_id: str
    name: str
    status: str
    created_at: float
    updated_at: float
    priority: int


@dataclass
class TaskStatusResponse:
    """
    Response containing detailed task status.

    Attributes:
        task_id: Task identifier
        status: Current status
        progress: Progress percentage (0-100)
        agents_assigned: Number of agents working on task
        agents_completed: Number of agents that completed
        consensus_achieved: Whether consensus was reached
        confidence_score: Confidence in result (0.0-1.0)
        estimated_completion: Estimated completion time
    """
    task_id: str
    status: str
    progress: int
    agents_assigned: int = 0
    agents_completed: int = 0
    consensus_achieved: bool = False
    confidence_score: float = 0.0
    estimated_completion: Optional[float] = None


@dataclass
class TaskResultResponse:
    """
    Response containing task execution result.

    Attributes:
        task_id: Task identifier
        success: Whether task completed successfully
        result: Task output/result
        quality_score: Quality score (0.0-1.0)
        execution_time_ms: Total execution time
        verification_passed: Whether verification passed
        agents_used: Number of agents that participated
        consensus_achieved: Whether consensus was reached
    """
    task_id: str
    success: bool
    result: Optional[Any]
    quality_score: float
    execution_time_ms: int
    verification_passed: bool
    agents_used: int
    consensus_achieved: bool


class TaskRouteHandler:
    """
    Handler for task-related API routes.

    This class implements the business logic for task management
    endpoints, integrating with the swarm coordinator and task
    distributor.
    """

    def __init__(self, swarm_coordinator=None, task_distributor=None):
        """
        Initialize task route handler.

        Args:
            swarm_coordinator: Agent swarm coordinator
            task_distributor: Task distributor
        """
        self.swarm_coordinator = swarm_coordinator
        self.task_distributor = task_distributor
        self.tasks: Dict[str, Any] = {}

    def submit_task(self, request: TaskSubmissionRequest) -> Optional[TaskResponse]:
        """
        Submit a new task for execution.

        Args:
            request: Task submission request

        Returns:
            Task response or None if submission failed
        """
        if not request.validate():
            return None

        if self.swarm_coordinator is None:
            return None

        # Generate task ID
        import hashlib
        task_id = hashlib.sha256(
            f"{request.name}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        # Create task specification
        # In production, would use actual TaskSpecification builder
        task_spec = {
            'id': task_id,
            'name': request.name,
            'description': request.description,
            'task_type': request.task_type,
            'inputs': request.inputs,
            'outputs': request.outputs,
            'max_lines': request.max_lines,
            'max_complexity': request.max_complexity,
            'priority': request.priority
        }

        # Store task
        now = datetime.now().timestamp()
        self.tasks[task_id] = {
            'spec': task_spec,
            'status': 'pending',
            'created_at': now,
            'updated_at': now
        }

        # Submit to swarm (would call actual swarm coordinator)
        # self.swarm_coordinator.submit_task(task_spec)

        return TaskResponse(
            task_id=task_id,
            name=request.name,
            status='pending',
            created_at=now,
            updated_at=now,
            priority=request.priority
        )

    def get_task(self, task_id: str) -> Optional[TaskResponse]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task response or None if not found
        """
        if not task_id or task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        return TaskResponse(
            task_id=task_id,
            name=task['spec']['name'],
            status=task['status'],
            created_at=task['created_at'],
            updated_at=task['updated_at'],
            priority=task['spec']['priority']
        )

    def list_tasks(
        self,
        status_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[TaskResponse]:
        """
        List all tasks.

        Args:
            status_filter: Optional status filter
            limit: Maximum number of tasks to return

        Returns:
            List of task responses
        """
        if limit <= 0:
            limit = 100

        responses = []

        for task_id, task in list(self.tasks.items())[:limit]:
            if status_filter and task['status'] != status_filter:
                continue

            responses.append(TaskResponse(
                task_id=task_id,
                name=task['spec']['name'],
                status=task['status'],
                created_at=task['created_at'],
                updated_at=task['updated_at'],
                priority=task['spec']['priority']
            ))

        return responses

    def get_task_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """
        Get detailed task status.

        Args:
            task_id: Task identifier

        Returns:
            Task status response or None if not found
        """
        if not task_id or task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        # Calculate progress (simplified)
        progress = 0
        if task['status'] == 'pending':
            progress = 0
        elif task['status'] == 'in_progress':
            progress = 50
        elif task['status'] == 'completed':
            progress = 100

        return TaskStatusResponse(
            task_id=task_id,
            status=task['status'],
            progress=progress,
            agents_assigned=task.get('agents_assigned', 0),
            agents_completed=task.get('agents_completed', 0),
            consensus_achieved=task.get('consensus_achieved', False),
            confidence_score=task.get('confidence_score', 0.0)
        )

    def get_task_result(self, task_id: str) -> Optional[TaskResultResponse]:
        """
        Get task execution result.

        Args:
            task_id: Task identifier

        Returns:
            Task result response or None if not found/not completed
        """
        if not task_id or task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        if task['status'] != 'completed':
            return None

        return TaskResultResponse(
            task_id=task_id,
            success=task.get('success', False),
            result=task.get('result'),
            quality_score=task.get('quality_score', 0.0),
            execution_time_ms=task.get('execution_time_ms', 0),
            verification_passed=task.get('verification_passed', False),
            agents_used=task.get('agents_used', 0),
            consensus_achieved=task.get('consensus_achieved', False)
        )

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier

        Returns:
            True if task was cancelled
        """
        if not task_id or task_id not in self.tasks:
            return False

        task = self.tasks[task_id]

        if task['status'] not in ['pending', 'in_progress']:
            return False

        task['status'] = 'cancelled'
        task['updated_at'] = datetime.now().timestamp()

        return True
