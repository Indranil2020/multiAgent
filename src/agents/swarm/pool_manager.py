"""
Agent Pool Manager.

This module manages the pool of agent instances, coordinating their creation,
lifecycle, and resource allocation. Agents are ephemeral - they're created
for execution and then discarded.
"""

from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime

from ..archetypes.base_agent import BaseAgent, AgentExecutionContext, AgentExecutionResult
from ..archetypes.decomposer_agent import DecomposerAgent
from ..archetypes.architect_agent import ArchitectAgent
from ..archetypes.coder_agent import CoderAgent
from ..archetypes.verifier_agent import VerifierAgent
from ..archetypes.tester_agent import TesterAgent
from ..archetypes.reviewer_agent import ReviewerAgent
from ..archetypes.documenter_agent import DocumenterAgent
from ..archetypes.optimizer_agent import OptimizerAgent
from ...core.task_spec.types import TaskType


@dataclass
class PoolStats:
    """
    Statistics for the agent pool.

    Attributes:
        total_executions: Total number of agent executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        average_execution_time_ms: Average execution time
        executions_by_type: Count of executions by agent type
    """
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time_ms: float = 0.0
    executions_by_type: Dict[str, int] = field(default_factory=dict)

    def record_execution(
        self,
        agent_type: str,
        success: bool,
        execution_time_ms: int
    ) -> None:
        """Record an execution."""
        self.total_executions += 1

        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update average execution time
        if self.total_executions > 0:
            total_time = self.average_execution_time_ms * (self.total_executions - 1)
            total_time += execution_time_ms
            self.average_execution_time_ms = total_time / self.total_executions

        # Update type counts
        if agent_type not in self.executions_by_type:
            self.executions_by_type[agent_type] = 0
        self.executions_by_type[agent_type] += 1


class AgentPoolManager:
    """
    Manages the agent pool for the swarm.

    Agents are ephemeral - created on demand, execute, and then discarded.
    The pool manager handles agent creation, type selection, and resource
    coordination.

    Design Principles:
    - Agents are stateless and ephemeral
    - Single shared LLM model pool (no VRAM duplication)
    - Type-based agent selection
    - Execution tracking and metrics
    """

    def __init__(self, llm_pool):
        """
        Initialize agent pool manager.

        Args:
            llm_pool: Shared LLM model pool for all agents
        """
        self.llm_pool = llm_pool
        self.stats = PoolStats()

        # Agent type registry
        self.agent_types: Dict[TaskType, Type[BaseAgent]] = {
            TaskType.DECOMPOSITION: DecomposerAgent,
            TaskType.ARCHITECTURE: ArchitectAgent,
            TaskType.CODE_GENERATION: CoderAgent,
            TaskType.VERIFICATION: VerifierAgent,
            TaskType.TESTING: TesterAgent,
            TaskType.REVIEW: ReviewerAgent,
            TaskType.DOCUMENTATION: DocumenterAgent,
            TaskType.OPTIMIZATION: OptimizerAgent
        }

    def execute_agent(self, context: AgentExecutionContext) -> AgentExecutionResult:
        """
        Execute an agent with given context.

        Creates an appropriate agent instance, executes it, and returns result.
        The agent is ephemeral - created for this execution and then discarded.

        Args:
            context: Execution context

        Returns:
            Execution result
        """
        # Validate context
        if not context or not context.validate():
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id="invalid_context",
                execution_time_ms=0,
                error_message="Invalid execution context"
            )

        # Get appropriate agent type
        agent_class = self._get_agent_class(context.task.task_type)
        if agent_class is None:
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id="unknown_type",
                execution_time_ms=0,
                error_message=f"No agent class for task type {context.task.task_type}"
            )

        # Create agent instance (ephemeral)
        agent = agent_class()

        # Execute agent
        result = agent.execute(context)

        # Record stats
        self.stats.record_execution(
            agent_type=agent.agent_type,
            success=result.success,
            execution_time_ms=result.execution_time_ms
        )

        return result

    def get_agent_class(self, task_type: TaskType) -> Optional[Type[BaseAgent]]:
        """
        Get agent class for a task type.

        Args:
            task_type: Type of task

        Returns:
            Agent class or None if not found
        """
        return self._get_agent_class(task_type)

    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Reset pool statistics."""
        self.stats = PoolStats()

    def _get_agent_class(self, task_type: TaskType) -> Optional[Type[BaseAgent]]:
        """
        Get agent class for task type.

        Args:
            task_type: Task type

        Returns:
            Agent class or None
        """
        return self.agent_types.get(task_type)
