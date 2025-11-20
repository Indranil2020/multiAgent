"""
Consensus mechanisms and decision-making logic.

This module manages the consensus state and provides logic for determining
when consensus has been reached, tracking vote distribution, and calculating
confidence metrics.
"""

from typing import Dict, List, Optional
from .types import (
    VoteResult,
    ConsensusState,
    ConsensusStatus,
    Result,
    VotingConfig
)


class ConsensusManager:
    """
    Manages consensus state and decision-making during voting.
    
    This class tracks votes, determines when consensus is achieved,
    and provides metrics about the quality of consensus.
    """
    
    def __init__(self, config: VotingConfig):
        """
        Initialize consensus manager.
        
        Args:
            config: Voting configuration
        """
        self.config = config
        self.state = ConsensusState()
    
    def reset(self) -> None:
        """Reset consensus state to initial values."""
        self.state = ConsensusState()
    
    def update_votes(self, vote: VoteResult) -> Result[bool, str]:
        """
        Update vote counts with a new result.
        
        Args:
            vote: New vote result to add
        
        Returns:
            Result indicating success
        """
        # Validate vote
        if not vote.semantic_signature:
            return Result(success=False, error="Vote missing semantic signature")
        
        if not vote.agent_id:
            return Result(success=False, error="Vote missing agent ID")
        
        # Add vote to state
        self.state.add_vote(vote)
        
        return Result(success=True, value=True)
    
    def increment_red_flagged(self) -> None:
        """Increment count of red-flagged results."""
        self.state.red_flagged_count += 1
        self.state.total_attempts += 1
    
    def check_consensus(self) -> Result[bool, str]:
        """
        Check if consensus has been reached.
        
        Consensus is achieved when one signature has k more votes than
        any other signature.
        
        Returns:
            Result containing True if consensus reached, False otherwise
        """
        # Need minimum votes before checking consensus
        valid_votes = self.state.total_attempts - self.state.red_flagged_count
        if valid_votes < self.config.min_votes:
            return Result(success=True, value=False)
        
        # Check if any signature is ahead by k
        max_votes = self.state.get_max_votes()
        second_max = self.state.get_second_max_votes()
        
        is_consensus = max_votes >= second_max + self.config.k
        
        if is_consensus:
            self.state.status = ConsensusStatus.ACHIEVED
            # Find winning signature
            vote_counts = self.state.get_vote_counts()
            for signature, count in vote_counts.items():
                if count == max_votes:
                    self.state.winning_signature = signature
                    break
        
        return Result(success=True, value=is_consensus)
    
    def get_winning_group(self) -> Result[List[VoteResult], str]:
        """
        Get all votes in the winning group.
        
        Returns:
            Result containing list of votes in winning group
        """
        if self.state.status != ConsensusStatus.ACHIEVED:
            return Result(success=False, error="Consensus not achieved")
        
        if not self.state.winning_signature:
            return Result(success=False, error="No winning signature set")
        
        winning_votes = self.state.votes_by_signature.get(
            self.state.winning_signature,
            []
        )
        
        if len(winning_votes) == 0:
            return Result(success=False, error="No votes for winning signature")
        
        return Result(success=True, value=winning_votes)
    
    def calculate_confidence(self) -> float:
        """
        Calculate confidence score for current consensus.
        
        Confidence is based on:
        - Margin of victory
        - Vote coverage (fraction of votes for winner)
        - Sample size
        - Vote distribution entropy
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        max_votes = self.state.get_max_votes()
        second_max = self.state.get_second_max_votes()
        total_valid = self.state.total_attempts - self.state.red_flagged_count
        
        if total_valid == 0:
            return 0.0
        
        # Margin component: how far ahead is winner
        margin = max_votes - second_max
        margin_score = min(margin / (2 * self.config.k), 1.0)
        
        # Coverage component: fraction of valid votes for winner
        coverage_score = max_votes / total_valid if total_valid > 0 else 0.0
        
        # Sample size component: more votes = higher confidence
        sample_score = min(total_valid / (3 * self.config.min_votes), 1.0)
        
        # Entropy component: lower entropy = higher confidence
        entropy_score = 1.0 - self._calculate_entropy()
        
        # Weighted combination
        confidence = (
            0.35 * margin_score +
            0.35 * coverage_score +
            0.15 * sample_score +
            0.15 * entropy_score
        )
        
        return min(confidence, 1.0)
    
    def _calculate_entropy(self) -> float:
        """
        Calculate entropy of vote distribution.
        
        Lower entropy means votes are more concentrated,
        indicating stronger consensus.
        
        Returns:
            Normalized entropy score between 0.0 and 1.0
        """
        vote_counts = self.state.get_vote_counts()
        total_votes = sum(vote_counts.values())
        
        if total_votes == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in vote_counts.values():
            if count > 0:
                p = count / total_votes
                entropy -= p * (p ** 0.5)  # Simplified entropy calculation
        
        # Normalize to [0, 1]
        max_entropy = 1.0  # Maximum possible entropy
        normalized_entropy = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def get_vote_distribution(self) -> Dict[str, int]:
        """
        Get current vote distribution.
        
        Returns:
            Dictionary mapping signatures to vote counts
        """
        return self.state.get_vote_counts()
    
    def get_convergence_rate(self) -> float:
        """
        Calculate rate of convergence toward consensus.
        
        Returns:
            Convergence rate between 0.0 (diverging) and 1.0 (converging)
        """
        vote_counts = self.state.get_vote_counts()
        
        if len(vote_counts) == 0:
            return 0.0
        
        max_votes = self.state.get_max_votes()
        total_votes = sum(vote_counts.values())
        
        if total_votes == 0:
            return 0.0
        
        # Convergence is high when most votes go to one signature
        convergence = max_votes / total_votes
        
        return convergence
    
    def get_state(self) -> ConsensusState:
        """
        Get current consensus state.
        
        Returns:
            Current ConsensusState
        """
        return self.state
    
    def set_status(self, status: ConsensusStatus) -> None:
        """
        Set consensus status.
        
        Args:
            status: New consensus status
        """
        self.state.status = status
    
    def is_converging(self, threshold: float = 0.6) -> bool:
        """
        Check if votes are converging toward consensus.
        
        Args:
            threshold: Minimum convergence rate to consider converging
        
        Returns:
            True if converging, False otherwise
        """
        convergence = self.get_convergence_rate()
        return convergence >= threshold
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive statistics about voting state.
        
        Returns:
            Dictionary with voting statistics
        """
        return {
            'total_attempts': self.state.total_attempts,
            'red_flagged_count': self.state.red_flagged_count,
            'valid_votes': self.state.total_attempts - self.state.red_flagged_count,
            'unique_signatures': len(self.state.votes_by_signature),
            'max_votes': self.state.get_max_votes(),
            'second_max_votes': self.state.get_second_max_votes(),
            'convergence_rate': self.get_convergence_rate(),
            'confidence_score': self.calculate_confidence(),
            'status': self.state.status.value,
            'winning_signature': self.state.winning_signature
        }
