"""
Agent Swarm Module.

This module provides swarm coordination capabilities for multi-agent execution:
- Swarm coordination and orchestration
- Agent pool management
- Task distribution
- Result aggregation

Components:
- coordinator: Main swarm orchestrator
- pool_manager: Agent lifecycle management
- task_distributor: Task scheduling and distribution
- result_aggregator: Result collection and aggregation
"""

from .coordinator import (
    SwarmCoordinator,
    SwarmConfig
)

from .pool_manager import (
    AgentPoolManager,
    PoolStats
)

from .task_distributor import (
    TaskDistributor,
    TaskAssignment,
    DistributionStats
)

from .result_aggregator import (
    ResultAggregator,
    AggregatedResult
)

__all__ = [
    # Coordinator
    'SwarmCoordinator',
    'SwarmConfig',

    # Pool Manager
    'AgentPoolManager',
    'PoolStats',

    # Task Distributor
    'TaskDistributor',
    'TaskAssignment',
    'DistributionStats',

    # Result Aggregator
    'ResultAggregator',
    'AggregatedResult',
]
