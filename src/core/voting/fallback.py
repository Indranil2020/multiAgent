"""
Fallback strategies when consensus cannot be reached.

This module provides various fallback mechanisms to handle cases where
the voting process fails to achieve consensus within normal parameters.
"""

from typing import Any, Optional, Callable
from .types import (
    FallbackStrategy,
    VotingOutcome,
    ConsensusState,
    ConsensusStatus,
    Result,
    VotingConfig,
    VoteResult
)


class FallbackManager:
    """
    Manages fallback execution when consensus fails.
    
    Provides a chain of fallback strategies that can be attempted
    in order when normal voting doesn't achieve consensus.
    """
    
    def __init__(self, config: VotingConfig):
        """
        Initialize fallback manager.
        
        Args:
            config: Voting configuration with fallback strategies
        """
        self.config = config
        self.fallback_chain = config.fallback_strategies
    
    def execute_fallback(
        self,
        consensus_state: ConsensusState,
        task_spec: Any,
        agent_spawner: Optional[Callable] = None
    ) -> Result[VotingOutcome, str]:
        """
        Execute appropriate fallback strategy.
        
        Tries fallback strategies in order until one succeeds or all fail.
        
        Args:
            consensus_state: Current voting state
            task_spec: Specification of the task being voted on
            agent_spawner: Optional function to spawn additional agents
        
        Returns:
            Result containing VotingOutcome from fallback
        """
        # Try each fallback strategy in order
        for strategy in self.fallback_chain:
            result = self._execute_strategy(
                strategy,
                consensus_state,
                task_spec,
                agent_spawner
            )
            
            if result.is_ok():
                return result
        
        # All fallbacks failed
        return self._create_failure_outcome(consensus_state)
    
    def _execute_strategy(
        self,
        strategy: FallbackStrategy,
        consensus_state: ConsensusState,
        task_spec: Any,
        agent_spawner: Optional[Callable]
    ) -> Result[VotingOutcome, str]:
        """
        Execute a specific fallback strategy.
        
        Args:
            strategy: The fallback strategy to execute
            consensus_state: Current voting state
            task_spec: Task specification
            agent_spawner: Function to spawn agents
        
        Returns:
            Result containing VotingOutcome
        """
        if strategy == FallbackStrategy.ACCEPT_PLURALITY:
            return self._accept_plurality(consensus_state)
        
        elif strategy == FallbackStrategy.INCREASE_AGENT_POOL:
            return self._increase_agent_pool(consensus_state, task_spec, agent_spawner)
        
        elif strategy == FallbackStrategy.FORMAL_VERIFICATION:
            return self._escalate_to_formal_verification(consensus_state, task_spec)
        
        elif strategy == FallbackStrategy.DECOMPOSE_FURTHER:
            return self._decompose_further(consensus_state, task_spec)
        
        elif strategy == FallbackStrategy.ESCALATE_TO_HUMAN:
            return self._escalate_to_human(consensus_state, task_spec)
        
        else:
            return Result(success=False, error=f"Unknown fallback strategy: {strategy}")
    
    def _accept_plurality(
        self,
        consensus_state: ConsensusState
    ) -> Result[VotingOutcome, str]:
        """
        Accept the plurality result (most votes, even without k margin).
        
        This is the least strict fallback - accepts the result with
        the most votes even if it doesn't meet the k-ahead threshold.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            Result containing VotingOutcome with plurality winner
        """
        vote_counts = consensus_state.get_vote_counts()
        
        if len(vote_counts) == 0:
            return Result(success=False, error="No votes available for plurality")
        
        # Find signature with most votes
        max_votes = consensus_state.get_max_votes()
        winning_signatures = [
            sig for sig, count in vote_counts.items()
            if count == max_votes
        ]
        
        if len(winning_signatures) == 0:
            return Result(success=False, error="No winning signature found")
        
        # If multiple tied, take first (arbitrary but deterministic)
        winning_signature = winning_signatures[0]
        winning_votes = consensus_state.votes_by_signature[winning_signature]
        
        # Select best quality from winning group
        best_vote = self._select_best_quality(winning_votes)
        
        # Create outcome
        consensus_state.status = ConsensusStatus.ACHIEVED
        consensus_state.winning_signature = winning_signature
        
        outcome = VotingOutcome(
            success=True,
            result=best_vote,
            consensus_state=consensus_state,
            fallback_used=FallbackStrategy.ACCEPT_PLURALITY,
            total_time_ms=0,  # Will be set by caller
            message=f"Consensus achieved via plurality fallback ({max_votes} votes)"
        )
        
        return Result(success=True, value=outcome)
    
    def _increase_agent_pool(
        self,
        consensus_state: ConsensusState,
        task_spec: Any,
        agent_spawner: Optional[Callable]
    ) -> Result[VotingOutcome, str]:
        """
        Increase agent pool and continue voting.
        
        Spawns additional agents to get more votes and potentially
        achieve consensus.
        
        Args:
            consensus_state: Current voting state
            task_spec: Task specification
            agent_spawner: Function to spawn additional agents
        
        Returns:
            Result indicating whether to continue voting
        """
        # This strategy signals to continue voting with more agents
        # The actual implementation is handled by the voting engine
        return Result(
            success=False,
            error="CONTINUE_VOTING_WITH_MORE_AGENTS"
        )
    
    def _escalate_to_formal_verification(
        self,
        consensus_state: ConsensusState,
        task_spec: Any
    ) -> Result[VotingOutcome, str]:
        """
        Escalate to formal verification to resolve disagreement.
        
        Uses formal methods to verify which result (if any) is correct.
        
        Args:
            consensus_state: Current voting state
            task_spec: Task specification
        
        Returns:
            Result containing VotingOutcome from formal verification
        """
        # Get top candidates for formal verification
        vote_counts = consensus_state.get_vote_counts()
        
        if len(vote_counts) == 0:
            return Result(success=False, error="No votes for formal verification")
        
        # Sort signatures by vote count
        sorted_sigs = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top 2-3 candidates
        top_candidates = sorted_sigs[:min(3, len(sorted_sigs))]
        
        # In real implementation, would run formal verification here
        # For now, return indication that formal verification is needed
        consensus_state.status = ConsensusStatus.FAILED
        
        outcome = VotingOutcome(
            success=False,
            result=None,
            consensus_state=consensus_state,
            fallback_used=FallbackStrategy.FORMAL_VERIFICATION,
            total_time_ms=0,
            message=f"Escalated to formal verification with {len(top_candidates)} candidates"
        )
        
        return Result(success=False, error="FORMAL_VERIFICATION_REQUIRED")
    
    def _decompose_further(
        self,
        consensus_state: ConsensusState,
        task_spec: Any
    ) -> Result[VotingOutcome, str]:
        """
        Decompose task further into smaller subtasks.
        
        If consensus can't be reached, the task may be too complex
        and should be broken down further.
        
        Args:
            consensus_state: Current voting state
            task_spec: Task specification
        
        Returns:
            Result indicating task should be decomposed
        """
        consensus_state.status = ConsensusStatus.FAILED
        
        outcome = VotingOutcome(
            success=False,
            result=None,
            consensus_state=consensus_state,
            fallback_used=FallbackStrategy.DECOMPOSE_FURTHER,
            total_time_ms=0,
            message="Task requires further decomposition"
        )
        
        return Result(success=False, error="DECOMPOSE_TASK_FURTHER")
    
    def _escalate_to_human(
        self,
        consensus_state: ConsensusState,
        task_spec: Any
    ) -> Result[VotingOutcome, str]:
        """
        Escalate to human for manual decision.
        
        When all automated strategies fail, request human intervention.
        
        Args:
            consensus_state: Current voting state
            task_spec: Task specification
        
        Returns:
            Result indicating human review needed
        """
        # Get top candidates for human review
        vote_counts = consensus_state.get_vote_counts()
        sorted_sigs = sorted(
            vote_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_candidates = sorted_sigs[:min(3, len(sorted_sigs))]
        
        consensus_state.status = ConsensusStatus.FAILED
        
        outcome = VotingOutcome(
            success=False,
            result=None,
            consensus_state=consensus_state,
            fallback_used=FallbackStrategy.ESCALATE_TO_HUMAN,
            total_time_ms=0,
            message=f"Human review required. Top {len(top_candidates)} candidates available."
        )
        
        return Result(success=False, error="HUMAN_REVIEW_REQUIRED")
    
    def _select_best_quality(self, votes: list) -> VoteResult:
        """
        Select best quality vote from a list.
        
        Args:
            votes: List of vote results
        
        Returns:
            Highest quality vote
        """
        if len(votes) == 1:
            return votes[0]
        
        # Sort by quality score (descending), then execution time (ascending)
        sorted_votes = sorted(
            votes,
            key=lambda v: (-v.quality_score, v.execution_time_ms)
        )
        
        return sorted_votes[0]
    
    def _create_failure_outcome(
        self,
        consensus_state: ConsensusState
    ) -> Result[VotingOutcome, str]:
        """
        Create failure outcome when all fallbacks fail.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            Result with failure outcome
        """
        consensus_state.status = ConsensusStatus.FAILED
        
        outcome = VotingOutcome(
            success=False,
            result=None,
            consensus_state=consensus_state,
            fallback_used=None,
            total_time_ms=0,
            message="All fallback strategies failed"
        )
        
        return Result(success=False, error="ALL_FALLBACKS_FAILED")
