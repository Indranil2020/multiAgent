"""
Swarm Coordinator.

This module orchestrates the entire agent swarm, coordinating task distribution,
agent execution, result aggregation, and voting integration.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .pool_manager import AgentPoolManager
from .task_distributor import TaskDistributor, TaskAssignment
from .result_aggregator import ResultAggregator, AggregatedResult
from ..archetypes.base_agent import AgentExecutionContext, AgentExecutionResult
from ..communication.message_bus import MessageBus
from ..communication.protocol import Message, MessageType, MessagePriority
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskID
from ...core.voting.types import AgentConfig, VotingConfig, ConsensusState


@dataclass
class SwarmConfig:
    """
    Configuration for swarm operation.

    Attributes:
        max_concurrent_tasks: Maximum concurrent task executions
        agents_per_task: Number of agents to spawn per task (for voting)
        enable_voting: Whether to enable voting mechanism
        voting_config: Configuration for voting system
    """
    max_concurrent_tasks: int = 10
    agents_per_task: int = 5
    enable_voting: bool = True
    voting_config: VotingConfig = None

    def __post_init__(self):
        """Initialize voting config if not provided."""
        if self.voting_config is None:
            self.voting_config = VotingConfig()


class SwarmCoordinator:
    """
    Coordinates the entire agent swarm.

    This is the main orchestrator that brings together all swarm components:
    - Agent pool management
    - Task distribution
    - Result aggregation
    - Communication
    - Voting integration

    Design:
    - Centralized coordination
    - Message-based communication
    - Voting-based consensus
    - Metrics and observability
    """

    def __init__(
        self,
        llm_pool,
        config: Optional[SwarmConfig] = None
    ):
        """
        Initialize swarm coordinator.

        Args:
            llm_pool: Shared LLM model pool
            config: Swarm configuration
        """
        self.llm_pool = llm_pool
        self.config = config or SwarmConfig()

        # Initialize components
        self.pool_manager = AgentPoolManager(llm_pool)
        self.task_distributor = TaskDistributor(
            max_concurrent_tasks=self.config.max_concurrent_tasks
        )
        self.result_aggregator = ResultAggregator()
        self.message_bus = MessageBus()

        # Execution tracking
        self.active_tasks: Dict[TaskID, TaskSpecification] = {}
        self.completed_tasks: Dict[TaskID, AggregatedResult] = {}

    def submit_task(self, task: TaskSpecification) -> bool:
        """
        Submit a task to the swarm.

        Args:
            task: Task to execute

        Returns:
            True if task was submitted successfully
        """
        if task is None:
            return False

        if not task.is_valid():
            return False

        # Add to task distributor
        if not self.task_distributor.enqueue_task(task):
            return False

        # Track active task
        self.active_tasks[task.id] = task

        return True

    def execute_task(
        self,
        task: TaskSpecification,
        agent_config: AgentConfig
    ) -> Optional[AggregatedResult]:
        """
        Execute a task with multiple agents and aggregate results.

        Args:
            task: Task to execute
            agent_config: Base configuration for agents

        Returns:
            Aggregated result or None if execution failed
        """
        if task is None or not task.is_valid():
            return None

        # Determine number of agents based on voting config
        num_agents = self.config.agents_per_task if self.config.enable_voting else 1

        # Execute agents
        results = []
        for i in range(num_agents):
            # Create diverse agent config
            diverse_config = self._create_diverse_config(agent_config, i)

            # Create execution context
            context = AgentExecutionContext(
                task=task,
                config=diverse_config,
                llm_pool=self.llm_pool
            )

            # Execute agent
            result = self.pool_manager.execute_agent(context)

            # Add result to aggregator
            self.result_aggregator.add_result(task.id, result)
            results.append(result)

        # Aggregate results
        consensus_state = self._run_voting(task.id, results) if self.config.enable_voting else None
        aggregated = self.result_aggregator.aggregate(task.id, consensus_state)

        # Mark task as completed
        if aggregated:
            self.completed_tasks[task.id] = aggregated
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]

        return aggregated

    def get_task_status(self, task_id: TaskID) -> Dict[str, Any]:
        """
        Get status of a task.

        Args:
            task_id: Task ID

        Returns:
            Status dictionary
        """
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                'status': 'completed',
                'consensus_achieved': result.consensus_achieved,
                'confidence_score': result.confidence_score,
                'total_agents': result.total_agents,
                'successful_agents': result.successful_agents
            }

        if task_id in self.active_tasks:
            return {
                'status': 'in_progress'
            }

        return {
            'status': 'unknown'
        }

    def get_swarm_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive swarm statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'pool_stats': self.pool_manager.get_stats(),
            'distributor_stats': self.task_distributor.get_stats(),
            'aggregator_stats': self.result_aggregator.get_stats(),
            'message_bus_stats': self.message_bus.get_stats(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks)
        }

    def _create_diverse_config(
        self,
        base_config: AgentConfig,
        diversity_index: int
    ) -> AgentConfig:
        """
        Create diverse agent configuration.

        Args:
            base_config: Base configuration
            diversity_index: Index for diversity

        Returns:
            Diverse agent configuration
        """
        # Create variation in temperature
        temp_min, temp_max = self.config.voting_config.diversity_temperature_range
        temp_range = temp_max - temp_min
        temperature = temp_min + (diversity_index * temp_range / max(1, self.config.agents_per_task - 1))

        return AgentConfig(
            model_name=base_config.model_name,
            temperature=temperature,
            system_prompt=base_config.system_prompt,
            diversity_index=diversity_index
        )

    def _run_voting(
        self,
        task_id: TaskID,
        results: List[AgentExecutionResult]
    ) -> Optional[ConsensusState]:
        """
        Run voting on agent results.

        Args:
            task_id: Task ID
            results: List of agent results

        Returns:
            Consensus state or None
        """
        # In production, would integrate with full voting engine
        # For now, simulate basic voting

        if not results:
            return None

        # Count results by output (simplified semantic equivalence)
        votes_by_output: Dict[str, List[AgentExecutionResult]] = {}

        for result in results:
            if not result.success:
                continue

            output_key = str(result.output)[:100]  # Simplified signature

            if output_key not in votes_by_output:
                votes_by_output[output_key] = []

            votes_by_output[output_key].append(result)

        if not votes_by_output:
            return None

        # Find winning output
        winning_signature = max(votes_by_output.keys(), key=lambda k: len(votes_by_output[k]))
        winning_votes = len(votes_by_output[winning_signature])
        total_votes = sum(len(v) for v in votes_by_output.values())

        # Create consensus state
        consensus_state = ConsensusState(
            winning_signature=winning_signature,
            total_attempts=len(results),
            confidence_score=winning_votes / total_votes if total_votes > 0 else 0.0
        )

        # Add votes
        for signature, vote_results in votes_by_output.items():
            for vote_result in vote_results:
                from ...core.voting.types import VoteResult
                vote = VoteResult(
                    agent_id=vote_result.agent_id,
                    output=vote_result.output,
                    semantic_signature=signature,
                    quality_score=vote_result.quality_score,
                    execution_time_ms=vote_result.execution_time_ms,
                    metadata=vote_result.metadata
                )
                consensus_state.add_vote(vote)

        return consensus_state
