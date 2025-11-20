"""
Agent Management API Routes.

This module provides REST endpoints for agent coordination, monitoring,
and management in the zero-error system.

Endpoints:
- GET    /agents          - List active agents
- GET    /agents/{id}     - Get agent details
- GET    /agents/stats    - Get agent statistics
- POST   /agents/spawn    - Manually spawn agents
- DELETE /agents/{id}     - Terminate an agent
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentInfo:
    """
    Information about an agent.

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type of agent (archetype)
        status: Current status (idle/busy/completed)
        task_id: ID of current task (if busy)
        created_at: Creation timestamp
        last_active: Last activity timestamp
    """
    agent_id: str
    agent_type: str
    status: str
    task_id: Optional[str] = None
    created_at: float = 0.0
    last_active: float = 0.0


@dataclass
class AgentStatsResponse:
    """
    Response containing agent statistics.

    Attributes:
        total_agents_spawned: Total agents created
        currently_active: Number of active agents
        currently_idle: Number of idle agents
        currently_busy: Number of busy agents
        total_tasks_completed: Tasks completed by agents
        avg_task_time_ms: Average task completion time
        agents_by_type: Count of agents by type
    """
    total_agents_spawned: int
    currently_active: int
    currently_idle: int
    currently_busy: int
    total_tasks_completed: int
    avg_task_time_ms: float
    agents_by_type: Dict[str, int] = field(default_factory=dict)


@dataclass
class SpawnAgentsRequest:
    """
    Request to manually spawn agents.

    Attributes:
        agent_type: Type of agent to spawn
        count: Number of agents to spawn
        task_id: Optional task ID to assign to
    """
    agent_type: str
    count: int = 1
    task_id: Optional[str] = None

    def validate(self) -> bool:
        """Validate request."""
        valid_types = [
            'decomposer', 'architect', 'coder',
            'verifier', 'tester', 'reviewer',
            'documenter', 'optimizer'
        ]

        if self.agent_type not in valid_types:
            return False

        if not (1 <= self.count <= 10):
            return False

        return True


@dataclass
class SpawnAgentsResponse:
    """
    Response from spawning agents.

    Attributes:
        agents_spawned: Number of agents spawned
        agent_ids: List of spawned agent IDs
    """
    agents_spawned: int
    agent_ids: List[str] = field(default_factory=list)


class AgentRouteHandler:
    """
    Handler for agent-related API routes.

    This class implements the business logic for agent management
    endpoints, integrating with the swarm coordinator and pool manager.
    """

    def __init__(self, swarm_coordinator=None, pool_manager=None):
        """
        Initialize agent route handler.

        Args:
            swarm_coordinator: Agent swarm coordinator
            pool_manager: Agent pool manager
        """
        self.swarm_coordinator = swarm_coordinator
        self.pool_manager = pool_manager
        self.agent_registry: Dict[str, AgentInfo] = {}
        self.stats = {
            'total_spawned': 0,
            'total_completed': 0,
            'total_task_time_ms': 0
        }

    def list_agents(
        self,
        status_filter: Optional[str] = None,
        agent_type_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentInfo]:
        """
        List all agents.

        Args:
            status_filter: Optional status filter
            agent_type_filter: Optional agent type filter
            limit: Maximum number of agents to return

        Returns:
            List of agent information
        """
        if limit <= 0:
            limit = 100

        results = []

        for agent_id, agent_info in list(self.agent_registry.items())[:limit]:
            if status_filter and agent_info.status != status_filter:
                continue

            if agent_type_filter and agent_info.agent_type != agent_type_filter:
                continue

            results.append(agent_info)

        return results

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """
        Get agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent information or None if not found
        """
        if not agent_id or agent_id not in self.agent_registry:
            return None

        return self.agent_registry[agent_id]

    def get_agent_stats(self) -> AgentStatsResponse:
        """
        Get agent statistics.

        Returns:
            Agent statistics response
        """
        currently_active = 0
        currently_idle = 0
        currently_busy = 0
        agents_by_type: Dict[str, int] = {}

        for agent_info in self.agent_registry.values():
            if agent_info.status == 'idle':
                currently_idle += 1
                currently_active += 1
            elif agent_info.status == 'busy':
                currently_busy += 1
                currently_active += 1

            agent_type = agent_info.agent_type
            agents_by_type[agent_type] = agents_by_type.get(agent_type, 0) + 1

        avg_time = 0.0
        if self.stats['total_completed'] > 0:
            avg_time = self.stats['total_task_time_ms'] / self.stats['total_completed']

        return AgentStatsResponse(
            total_agents_spawned=self.stats['total_spawned'],
            currently_active=currently_active,
            currently_idle=currently_idle,
            currently_busy=currently_busy,
            total_tasks_completed=self.stats['total_completed'],
            avg_task_time_ms=avg_time,
            agents_by_type=agents_by_type
        )

    def spawn_agents(self, request: SpawnAgentsRequest) -> Optional[SpawnAgentsResponse]:
        """
        Manually spawn agents.

        Args:
            request: Spawn agents request

        Returns:
            Spawn response or None if spawn failed
        """
        if not request.validate():
            return None

        agent_ids = []

        for i in range(request.count):
            # Generate agent ID
            import hashlib
            agent_id = hashlib.sha256(
                f"{request.agent_type}_{datetime.now().timestamp()}_{i}".encode()
            ).hexdigest()[:16]

            # Create agent info
            now = datetime.now().timestamp()
            agent_info = AgentInfo(
                agent_id=agent_id,
                agent_type=request.agent_type,
                status='idle' if not request.task_id else 'busy',
                task_id=request.task_id,
                created_at=now,
                last_active=now
            )

            # Register agent
            self.agent_registry[agent_id] = agent_info
            agent_ids.append(agent_id)

            # Update stats
            self.stats['total_spawned'] += 1

        return SpawnAgentsResponse(
            agents_spawned=request.count,
            agent_ids=agent_ids
        )

    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was terminated
        """
        if not agent_id or agent_id not in self.agent_registry:
            return False

        agent_info = self.agent_registry[agent_id]

        # Can only terminate idle agents
        if agent_info.status != 'idle':
            return False

        # Remove from registry
        del self.agent_registry[agent_id]

        return True

    def update_agent_status(
        self,
        agent_id: str,
        status: str,
        task_id: Optional[str] = None
    ) -> bool:
        """
        Update agent status.

        Args:
            agent_id: Agent identifier
            status: New status
            task_id: Optional task ID

        Returns:
            True if status updated
        """
        if not agent_id or agent_id not in self.agent_registry:
            return False

        valid_statuses = ['idle', 'busy', 'completed']
        if status not in valid_statuses:
            return False

        agent_info = self.agent_registry[agent_id]
        agent_info.status = status
        agent_info.task_id = task_id
        agent_info.last_active = datetime.now().timestamp()

        return True

    def record_task_completion(self, agent_id: str, execution_time_ms: int) -> bool:
        """
        Record task completion for statistics.

        Args:
            agent_id: Agent identifier
            execution_time_ms: Task execution time

        Returns:
            True if recorded
        """
        if not agent_id or agent_id not in self.agent_registry:
            return False

        self.stats['total_completed'] += 1
        self.stats['total_task_time_ms'] += execution_time_ms

        return True
