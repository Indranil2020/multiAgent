"""
Voting module for multi-agent consensus.

This module implements the MAKER-style first-to-ahead-by-k voting mechanism
for achieving zero-error guarantees through multi-agent consensus.

Main Components:
- VotingEngine: Main orchestrator for the voting process
- MAKERVoting: Core MAKER algorithm implementation
- SemanticChecker: Semantic equivalence checking
- ConsensusManager: Consensus state management
- FallbackManager: Fallback strategies when consensus fails

Example Usage:
    ```python
    from src.core.voting import VotingEngine, VotingConfig
    
    # Create voting engine with configuration
    config = VotingConfig(k=3, max_attempts=20)
    engine = VotingEngine(config)
    
    # Execute voting
    outcome = engine.execute_with_voting(
        task_spec=my_task,
        agent_executor=my_agent_function,
        result_validator=my_validator,
        test_cases=my_tests
    )
    
    if outcome.success:
        print(f"Consensus achieved: {outcome.result}")
    else:
        print(f"Voting failed: {outcome.message}")
    ```
"""

from .types import (
    # Core types
    Result,
    VotingConfig,
    VotingOutcome,
    VoteResult,
    ConsensusState,
    AgentConfig,
    
    # Enums
    ConsensusStatus,
    FallbackStrategy
)

from .engine import VotingEngine
from .maker_voting import MAKERVoting
from .semantic_checker import SemanticChecker
from .consensus import ConsensusManager
from .fallback import FallbackManager


__version__ = "0.1.0"

__all__ = [
    # Main classes
    "VotingEngine",
    "MAKERVoting",
    "SemanticChecker",
    "ConsensusManager",
    "FallbackManager",
    
    # Types
    "Result",
    "VotingConfig",
    "VotingOutcome",
    "VoteResult",
    "ConsensusState",
    "AgentConfig",
    
    # Enums
    "ConsensusStatus",
    "FallbackStrategy",
]
