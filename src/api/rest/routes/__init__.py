"""
REST API Routes Module.

This module provides all REST API route handlers.

Available route handlers:
- tasks: Task management operations
- agents: Agent coordination operations
- verification: Code verification operations
- monitoring: System monitoring operations

Each handler implements the business logic for its respective endpoints
and integrates with the core system components.

Usage:
    from api.rest.routes.tasks import TaskRouteHandler
    from api.rest.routes.agents import AgentRouteHandler

    task_handler = TaskRouteHandler(
        swarm_coordinator=coordinator,
        task_distributor=distributor
    )

    agent_handler = AgentRouteHandler(
        swarm_coordinator=coordinator,
        pool_manager=pool_mgr
    )
"""

from .tasks import (
    TaskRouteHandler,
    TaskSubmissionRequest,
    TaskResponse,
    TaskStatusResponse,
    TaskResultResponse
)

from .agents import (
    AgentRouteHandler,
    AgentInfo,
    AgentStatsResponse,
    SpawnAgentsRequest,
    SpawnAgentsResponse
)

from .verification import (
    VerificationRouteHandler,
    VerificationRequest,
    VerificationResponse,
    VerificationStatsResponse,
    LayerResult
)

from .monitoring import (
    MonitoringRouteHandler,
    HealthCheckResponse,
    MetricsResponse,
    RedFlagsResponse,
    RedFlagEvent,
    ConsensusStats
)

__all__ = [
    # Tasks
    'TaskRouteHandler',
    'TaskSubmissionRequest',
    'TaskResponse',
    'TaskStatusResponse',
    'TaskResultResponse',

    # Agents
    'AgentRouteHandler',
    'AgentInfo',
    'AgentStatsResponse',
    'SpawnAgentsRequest',
    'SpawnAgentsResponse',

    # Verification
    'VerificationRouteHandler',
    'VerificationRequest',
    'VerificationResponse',
    'VerificationStatsResponse',
    'LayerResult',

    # Monitoring
    'MonitoringRouteHandler',
    'HealthCheckResponse',
    'MetricsResponse',
    'RedFlagsResponse',
    'RedFlagEvent',
    'ConsensusStats',
]
