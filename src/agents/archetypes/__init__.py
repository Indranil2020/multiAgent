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
- SpecificationAgent: Specification generation
"""

from .base_agent import BaseAgent, AgentExecutionContext, AgentExecutionResult
from .coder_agent import CoderAgent
from .decomposer_agent import DecomposerAgent
from .verifier_agent import VerifierAgent
from .tester_agent import TesterAgent
from .reviewer_agent import ReviewerAgent
from .optimizer_agent import OptimizerAgent
from .documenter_agent import DocumenterAgent
from .architect_agent import ArchitectAgent
from .specification_agent import SpecificationAgent

__all__ = [
    'BaseAgent',
    'AgentExecutionContext',
    'AgentExecutionResult',
    'CoderAgent',
    'DecomposerAgent',
    'VerifierAgent',
    'TesterAgent',
    'ReviewerAgent',
    'OptimizerAgent',
    'DocumenterAgent',
    'ArchitectAgent',
    'SpecificationAgent',
]
