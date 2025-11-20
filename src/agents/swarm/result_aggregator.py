"""
Result Aggregator for Agent Swarm.

This module aggregates results from multiple agent executions, handles
voting integration, and produces final consolidated results.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..archetypes.base_agent import AgentExecutionResult
from ...core.task_spec.types import TaskID
from ...core.voting.types import VoteResult, ConsensusState


@dataclass
class AggregatedResult:
    """
    Aggregated result from multiple agent executions.

    Attributes:
        task_id: ID of task
        results: List of individual agent results
        consensus_achieved: Whether consensus was reached
        winning_result: The consensus/winning result
        confidence_score: Confidence in the result (0.0 to 1.0)
        total_agents: Total number of agents executed
        successful_agents: Number of successful executions
        average_quality_score: Average quality across all results
        aggregated_at: Timestamp of aggregation
    """
    task_id: TaskID
    results: List[AgentExecutionResult]
    consensus_achieved: bool
    winning_result: Optional[Any]
    confidence_score: float
    total_agents: int
    successful_agents: int
    average_quality_score: float
    aggregated_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def get_best_result(self) -> Optional[AgentExecutionResult]:
        """Get result with highest quality score."""
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return None

        return max(successful_results, key=lambda r: r.quality_score)

    def get_failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_agents == 0:
            return 0.0
        failed = self.total_agents - self.successful_agents
        return failed / self.total_agents


class ResultAggregator:
    """
    Aggregates results from multiple agent executions.

    Collects results from swarm agents, applies voting logic, and produces
    consolidated results with confidence scores.

    Design:
    - Collects results from multiple agents
    - Integrates with voting system
    - Calculates confidence scores
    - Handles tie-breaking
    """

    def __init__(self):
        """Initialize result aggregator."""
        self.results_by_task: Dict[TaskID, List[AgentExecutionResult]] = {}
        self.aggregated_results: Dict[TaskID, AggregatedResult] = {}

    def add_result(
        self,
        task_id: TaskID,
        result: AgentExecutionResult
    ) -> bool:
        """
        Add agent execution result.

        Args:
            task_id: ID of task
            result: Execution result to add

        Returns:
            True if result was added
        """
        if task_id is None or not task_id.strip():
            return False

        if result is None:
            return False

        # Initialize list if needed
        if task_id not in self.results_by_task:
            self.results_by_task[task_id] = []

        # Add result
        self.results_by_task[task_id].append(result)

        return True

    def aggregate(
        self,
        task_id: TaskID,
        consensus_state: Optional[ConsensusState] = None
    ) -> Optional[AggregatedResult]:
        """
        Aggregate results for a task.

        Args:
            task_id: ID of task to aggregate
            consensus_state: Optional consensus state from voting

        Returns:
            Aggregated result or None if no results available
        """
        if task_id not in self.results_by_task:
            return None

        results = self.results_by_task[task_id]

        if not results:
            return None

        # Calculate statistics
        total_agents = len(results)
        successful_agents = sum(1 for r in results if r.success)
        average_quality = self._calculate_average_quality(results)

        # Determine winning result
        winning_result = None
        consensus_achieved = False
        confidence_score = 0.0

        if consensus_state and consensus_state.winning_signature:
            # Use consensus from voting system
            consensus_achieved = True
            winning_result = self._get_result_by_signature(
                results,
                consensus_state.winning_signature
            )
            confidence_score = consensus_state.confidence_score
        else:
            # No consensus, use best result
            best_result = self._get_best_result(results)
            if best_result:
                winning_result = best_result.output
                confidence_score = best_result.quality_score

        # Create aggregated result
        aggregated = AggregatedResult(
            task_id=task_id,
            results=results,
            consensus_achieved=consensus_achieved,
            winning_result=winning_result,
            confidence_score=confidence_score,
            total_agents=total_agents,
            successful_agents=successful_agents,
            average_quality_score=average_quality
        )

        # Cache aggregated result
        self.aggregated_results[task_id] = aggregated

        return aggregated

    def get_aggregated_result(self, task_id: TaskID) -> Optional[AggregatedResult]:
        """
        Get previously aggregated result.

        Args:
            task_id: Task ID

        Returns:
            Aggregated result or None
        """
        return self.aggregated_results.get(task_id)

    def clear_results(self, task_id: TaskID) -> bool:
        """
        Clear results for a task.

        Args:
            task_id: Task ID

        Returns:
            True if results were cleared
        """
        if task_id in self.results_by_task:
            del self.results_by_task[task_id]

        if task_id in self.aggregated_results:
            del self.aggregated_results[task_id]

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'tasks_tracked': len(self.results_by_task),
            'tasks_aggregated': len(self.aggregated_results),
            'total_results': sum(len(r) for r in self.results_by_task.values())
        }

    def _calculate_average_quality(
        self,
        results: List[AgentExecutionResult]
    ) -> float:
        """Calculate average quality score."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return 0.0

        total_quality = sum(r.quality_score for r in successful_results)
        return total_quality / len(successful_results)

    def _get_best_result(
        self,
        results: List[AgentExecutionResult]
    ) -> Optional[AgentExecutionResult]:
        """Get result with highest quality score."""
        successful_results = [r for r in results if r.success]

        if not successful_results:
            return None

        return max(successful_results, key=lambda r: r.quality_score)

    def _get_result_by_signature(
        self,
        results: List[AgentExecutionResult],
        signature: str
    ) -> Optional[Any]:
        """
        Get result output by semantic signature.

        Args:
            results: List of results
            signature: Semantic signature to match

        Returns:
            Result output or None
        """
        # In production, would match results to signatures
        # For now, return best result
        best = self._get_best_result(results)
        return best.output if best else None
