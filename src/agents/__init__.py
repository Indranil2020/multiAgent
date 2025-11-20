"""
Agents Module for Zero-Error System.

This module provides the complete agent infrastructure including:
- Agent archetypes for different task types
- Swarm coordination for multi-agent execution
- Communication infrastructure for agent coordination
- Integration with voting and verification systems

Main Components:
- archetypes: Specialized agent implementations
- swarm: Swarm coordination and orchestration
- communication: Message-based communication
"""

from .archetypes import (
    BaseAgent,
    DecomposerAgent,
    ArchitectAgent,
    CoderAgent,
    VerifierAgent,
    TesterAgent,
    ReviewerAgent,
    DocumenterAgent,
    OptimizerAgent,
    AgentExecutionContext,
    AgentExecutionResult
)

from .swarm import (
    SwarmCoordinator,
    SwarmConfig,
    AgentPoolManager,
    TaskDistributor,
    ResultAggregator,
    AggregatedResult
)

from .communication import (
    Message,
    MessageType,
    MessagePriority,
    MessageBus,
    MessageSerializer,
    ProtocolValidator
)

__all__ = [
    # Archetypes
    'BaseAgent',
    'DecomposerAgent',
    'ArchitectAgent',
    'CoderAgent',
    'VerifierAgent',
    'TesterAgent',
    'ReviewerAgent',
    'DocumenterAgent',
    'OptimizerAgent',
    'AgentExecutionContext',
    'AgentExecutionResult',

    # Swarm
    'SwarmCoordinator',
    'SwarmConfig',
    'AgentPoolManager',
    'TaskDistributor',
    'ResultAggregator',
    'AggregatedResult',

    # Communication
    'Message',
    'MessageType',
    'MessagePriority',
    'MessageBus',
    'MessageSerializer',
    'ProtocolValidator',
]

__version__ = '0.1.0'
__author__ = 'Zero-Error System Team'
