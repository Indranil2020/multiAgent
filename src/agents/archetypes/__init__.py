"""
Agent Archetypes Module.

This module provides all agent archetype implementations for the zero-error system.
Each archetype specializes in a specific type of task.

Available Archetypes:
- BaseAgent: Abstract base class for all agents
- DecomposerAgent: Task decomposition
- ArchitectAgent: System architecture design
- CoderAgent: Code generation
- VerifierAgent: Code verification
- TesterAgent: Test generation
- ReviewerAgent: Code review
- DocumenterAgent: Documentation generation
- OptimizerAgent: Code optimization
"""

from .base_agent import (
    BaseAgent,
    AgentExecutionContext,
    AgentExecutionResult,
    LLMPool
)
from .decomposer_agent import DecomposerAgent
from .architect_agent import ArchitectAgent
from .coder_agent import CoderAgent
from .verifier_agent import VerifierAgent
from .tester_agent import TesterAgent
from .reviewer_agent import ReviewerAgent
from .documenter_agent import DocumenterAgent
from .optimizer_agent import OptimizerAgent

__all__ = [
    # Base
    'BaseAgent',
    'AgentExecutionContext',
    'AgentExecutionResult',
    'LLMPool',

    # Archetypes
    'DecomposerAgent',
    'ArchitectAgent',
    'CoderAgent',
    'VerifierAgent',
    'TesterAgent',
    'ReviewerAgent',
    'DocumenterAgent',
    'OptimizerAgent',
]
