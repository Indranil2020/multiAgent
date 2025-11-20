"""
Type definitions and data structures for the voting system.

This module provides core types used throughout the voting system, including
configuration, results, and state tracking structures. All types are designed
to support explicit error handling without exceptions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generic, TypeVar
from enum import Enum


# Generic type for Result pattern
T = TypeVar('T')
E = TypeVar('E')


class ConsensusStatus(Enum):
    """Status of consensus during voting process."""
    PENDING = "pending"
    ACHIEVED = "achieved"
    FAILED = "failed"
    TIMEOUT = "timeout"


class FallbackStrategy(Enum):
    """Available fallback strategies when consensus fails."""
    FORMAL_VERIFICATION = "formal_verification"
    INCREASE_AGENT_POOL = "increase_agent_pool"
    DECOMPOSE_FURTHER = "decompose_further"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    ACCEPT_PLURALITY = "accept_plurality"


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    Result type for explicit error handling without exceptions.
    
    Attributes:
        success: Whether the operation succeeded
        value: The successful value (None if failed)
        error: The error value (None if succeeded)
    """
    success: bool
    value: Optional[T] = None
    error: Optional[E] = None
    
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self.success
    
    def is_err(self) -> bool:
        """Check if result is an error."""
        return not self.success
    
    def unwrap(self) -> T:
        """Get value, assuming success (caller must check first)."""
        assert self.success, f"Called unwrap on error result: {self.error}"
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get value or default if error."""
        return self.value if self.success else default
    
    def error_value(self) -> E:
        """Get error value, assuming failure (caller must check first)."""
        assert not self.success, "Called error_value on success result"
        return self.error


@dataclass
class VotingConfig:
    """
    Configuration parameters for the voting system.
    
    Attributes:
        k: Number of votes needed to be ahead for consensus (default: 3)
        max_attempts: Maximum number of agent executions (default: 20)
        timeout_seconds: Timeout for entire voting process (default: 300)
        min_votes: Minimum votes required before checking consensus (default: 5)
        enable_semantic_checking: Whether to use semantic equivalence (default: True)
        fallback_strategies: Ordered list of fallback strategies to try
        diversity_temperature_range: Range of temperatures for agent diversity
        diversity_model_rotation: Whether to rotate between different models
    """
    k: int = 3
    max_attempts: int = 20
    timeout_seconds: int = 300
    min_votes: int = 5
    enable_semantic_checking: bool = True
    fallback_strategies: List[FallbackStrategy] = field(default_factory=lambda: [
        FallbackStrategy.INCREASE_AGENT_POOL,
        FallbackStrategy.FORMAL_VERIFICATION,
        FallbackStrategy.DECOMPOSE_FURTHER,
        FallbackStrategy.ESCALATE_TO_HUMAN
    ])
    diversity_temperature_range: tuple = (0.1, 0.9)
    diversity_model_rotation: bool = True
    
    def validate(self) -> Result[bool, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Result indicating if configuration is valid
        """
        if self.k < 1:
            return Result(success=False, error="k must be at least 1")
        
        if self.max_attempts < self.k:
            return Result(success=False, error="max_attempts must be >= k")
        
        if self.timeout_seconds <= 0:
            return Result(success=False, error="timeout_seconds must be positive")
        
        if self.min_votes < self.k:
            return Result(success=False, error="min_votes must be >= k")
        
        temp_min, temp_max = self.diversity_temperature_range
        if temp_min < 0 or temp_max > 1 or temp_min >= temp_max:
            return Result(success=False, error="Invalid temperature range")
        
        return Result(success=True, value=True)


@dataclass
class VoteResult:
    """
    Result of a single agent execution.
    
    Attributes:
        agent_id: Unique identifier for the agent
        output: The output produced by the agent
        semantic_signature: Hash representing semantic equivalence class
        quality_score: Quality score for this result (higher is better)
        execution_time_ms: Time taken to execute in milliseconds
        metadata: Additional metadata about the execution
    """
    agent_id: str
    output: Any
    semantic_signature: str
    quality_score: float
    execution_time_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusState:
    """
    Tracks the state of consensus during voting.
    
    Attributes:
        votes_by_signature: Mapping of semantic signatures to vote results
        total_attempts: Total number of agent executions attempted
        red_flagged_count: Number of results that were red-flagged
        status: Current consensus status
        winning_signature: Signature of winning group (if consensus achieved)
        confidence_score: Confidence in the consensus (0.0 to 1.0)
    """
    votes_by_signature: Dict[str, List[VoteResult]] = field(default_factory=dict)
    total_attempts: int = 0
    red_flagged_count: int = 0
    status: ConsensusStatus = ConsensusStatus.PENDING
    winning_signature: Optional[str] = None
    confidence_score: float = 0.0
    
    def get_vote_counts(self) -> Dict[str, int]:
        """Get count of votes for each signature."""
        return {sig: len(results) for sig, results in self.votes_by_signature.items()}
    
    def get_max_votes(self) -> int:
        """Get the maximum number of votes for any signature."""
        counts = self.get_vote_counts()
        return max(counts.values()) if counts else 0
    
    def get_second_max_votes(self) -> int:
        """Get the second highest number of votes."""
        counts = self.get_vote_counts()
        if len(counts) < 2:
            return 0
        sorted_counts = sorted(counts.values(), reverse=True)
        return sorted_counts[1]
    
    def add_vote(self, vote: VoteResult) -> None:
        """Add a vote to the state."""
        signature = vote.semantic_signature
        if signature not in self.votes_by_signature:
            self.votes_by_signature[signature] = []
        self.votes_by_signature[signature].append(vote)
        self.total_attempts += 1


@dataclass
class VotingOutcome:
    """
    Final outcome of the voting process.
    
    Attributes:
        success: Whether voting succeeded in reaching consensus
        result: The winning result (if successful)
        consensus_state: Final state of consensus tracking
        fallback_used: Which fallback strategy was used (if any)
        total_time_ms: Total time taken for voting process
        message: Human-readable message about the outcome
    """
    success: bool
    result: Optional[VoteResult]
    consensus_state: ConsensusState
    fallback_used: Optional[FallbackStrategy]
    total_time_ms: int
    message: str
    
    def get_winning_votes(self) -> List[VoteResult]:
        """Get all votes in the winning group."""
        if not self.success or not self.consensus_state.winning_signature:
            return []
        return self.consensus_state.votes_by_signature.get(
            self.consensus_state.winning_signature, []
        )
    
    def get_vote_distribution(self) -> Dict[str, int]:
        """Get distribution of votes across signatures."""
        return self.consensus_state.get_vote_counts()


@dataclass
class AgentConfig:
    """
    Configuration for spawning a diverse agent.
    
    Attributes:
        model_name: Name of the LLM model to use
        temperature: Sampling temperature
        system_prompt: System prompt variant
        diversity_index: Index for tracking diversity
    """
    model_name: str
    temperature: float
    system_prompt: str
    diversity_index: int
