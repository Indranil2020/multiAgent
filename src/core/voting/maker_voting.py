"""
MAKER first-to-ahead-by-k voting algorithm implementation.

This module implements the core MAKER voting algorithm as described in the
architecture. The algorithm achieves consensus when one result is ahead of
all others by k votes, providing strong error correction guarantees.
"""

from typing import List, Optional
from .types import (
    VoteResult,
    ConsensusState,
    ConsensusStatus,
    Result,
    VotingConfig
)


class MAKERVoting:
    """
    Implementation of the MAKER first-to-ahead-by-k voting algorithm.
    
    This class provides the core voting logic that determines when consensus
    has been reached based on vote counts across semantically equivalent results.
    """
    
    def __init__(self, config: VotingConfig):
        """
        Initialize MAKER voting with configuration.
        
        Args:
            config: Voting configuration parameters
        """
        self.config = config
    
    def vote(self, consensus_state: ConsensusState) -> Result[Optional[VoteResult], str]:
        """
        Execute first-to-ahead-by-k voting logic.
        
        Checks if any result has achieved consensus by being ahead of all
        others by k votes. If consensus is achieved, selects the best quality
        result from the winning group.
        
        Args:
            consensus_state: Current state of voting with all votes
        
        Returns:
            Result containing winning VoteResult if consensus achieved, None otherwise
        """
        # Check if we have minimum votes
        if consensus_state.total_attempts < self.config.min_votes:
            return Result(success=True, value=None)
        
        # Check if consensus has been achieved
        is_consensus = self._is_ahead_by_k(consensus_state)
        
        if not is_consensus:
            return Result(success=True, value=None)
        
        # Select winner from the leading group
        winner_result = self._select_winner(consensus_state)
        
        if winner_result.is_err():
            return winner_result
        
        # Update consensus state
        consensus_state.status = ConsensusStatus.ACHIEVED
        consensus_state.winning_signature = winner_result.unwrap().semantic_signature
        consensus_state.confidence_score = self._calculate_confidence(consensus_state)
        
        return Result(success=True, value=winner_result.unwrap())
    
    def _is_ahead_by_k(self, consensus_state: ConsensusState) -> bool:
        """
        Check if any result is ahead by k votes.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            True if a result is ahead by k votes, False otherwise
        """
        vote_counts = consensus_state.get_vote_counts()
        
        if len(vote_counts) == 0:
            return False
        
        max_votes = consensus_state.get_max_votes()
        second_max_votes = consensus_state.get_second_max_votes()
        
        # Check if leader is ahead by k
        return max_votes >= second_max_votes + self.config.k
    
    def _calculate_vote_counts(self, consensus_state: ConsensusState) -> dict:
        """
        Calculate current vote distribution.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            Dictionary mapping signatures to vote counts
        """
        return consensus_state.get_vote_counts()
    
    def _select_winner(self, consensus_state: ConsensusState) -> Result[VoteResult, str]:
        """
        Select the winning result from the consensus group.
        
        Selects the highest quality result from the group with the most votes.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            Result containing the winning VoteResult
        """
        vote_counts = consensus_state.get_vote_counts()
        
        if len(vote_counts) == 0:
            return Result(success=False, error="No votes available")
        
        # Find signature with most votes
        max_votes = consensus_state.get_max_votes()
        winning_signatures = [
            sig for sig, count in vote_counts.items() 
            if count == max_votes
        ]
        
        if len(winning_signatures) == 0:
            return Result(success=False, error="No winning signature found")
        
        # If multiple signatures tied at max, this shouldn't happen with ahead-by-k
        # but we handle it by selecting first one
        winning_signature = winning_signatures[0]
        
        # Get all results with winning signature
        winning_results = consensus_state.votes_by_signature[winning_signature]
        
        if len(winning_results) == 0:
            return Result(success=False, error="No results for winning signature")
        
        # Select highest quality result
        best_result = self._select_best_quality(winning_results)
        
        return Result(success=True, value=best_result)
    
    def _select_best_quality(self, results: List[VoteResult]) -> VoteResult:
        """
        Select the best quality result from a list.
        
        Uses quality_score as primary criterion. If scores are equal,
        prefers results with shorter execution time.
        
        Args:
            results: List of vote results to choose from
        
        Returns:
            The highest quality result
        """
        if len(results) == 1:
            return results[0]
        
        # Sort by quality score (descending), then execution time (ascending)
        sorted_results = sorted(
            results,
            key=lambda r: (-r.quality_score, r.execution_time_ms)
        )
        
        return sorted_results[0]
    
    def _calculate_confidence(self, consensus_state: ConsensusState) -> float:
        """
        Calculate confidence score for the consensus.
        
        Confidence is based on:
        - Margin of victory (how far ahead the winner is)
        - Total number of votes
        - Agreement within winning group
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        max_votes = consensus_state.get_max_votes()
        second_max = consensus_state.get_second_max_votes()
        total_votes = consensus_state.total_attempts - consensus_state.red_flagged_count
        
        if total_votes == 0:
            return 0.0
        
        # Margin component: how far ahead is the winner
        margin = max_votes - second_max
        margin_score = min(margin / (2 * self.config.k), 1.0)
        
        # Coverage component: what fraction of valid votes went to winner
        coverage_score = max_votes / total_votes if total_votes > 0 else 0.0
        
        # Sample size component: more votes = higher confidence
        sample_score = min(total_votes / (3 * self.config.min_votes), 1.0)
        
        # Weighted combination
        confidence = (
            0.4 * margin_score +
            0.4 * coverage_score +
            0.2 * sample_score
        )
        
        return min(confidence, 1.0)
    
    def check_timeout(self, consensus_state: ConsensusState, elapsed_ms: int) -> bool:
        """
        Check if voting has exceeded timeout.
        
        Args:
            consensus_state: Current voting state
            elapsed_ms: Elapsed time in milliseconds
        
        Returns:
            True if timeout exceeded
        """
        timeout_ms = self.config.timeout_seconds * 1000
        return elapsed_ms >= timeout_ms
    
    def check_max_attempts(self, consensus_state: ConsensusState) -> bool:
        """
        Check if maximum attempts have been reached.
        
        Args:
            consensus_state: Current voting state
        
        Returns:
            True if max attempts reached
        """
        return consensus_state.total_attempts >= self.config.max_attempts
