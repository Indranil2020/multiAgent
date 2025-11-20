"""
Main voting engine orchestrator.

This module provides the VotingEngine class that coordinates the entire
voting process, integrating semantic checking, MAKER voting, consensus
management, and fallback strategies.
"""

import time
from typing import Any, Callable, Optional, List
from .types import (
    VotingConfig,
    VotingOutcome,
    VoteResult,
    ConsensusState,
    ConsensusStatus,
    AgentConfig,
    Result
)
from .maker_voting import MAKERVoting
from .semantic_checker import SemanticChecker
from .consensus import ConsensusManager
from .fallback import FallbackManager


class VotingEngine:
    """
    Main orchestrator for the voting process.
    
    Coordinates agent execution, semantic checking, vote counting,
    consensus detection, and fallback strategies to achieve reliable
    multi-agent consensus.
    """
    
    def __init__(self, config: Optional[VotingConfig] = None):
        """
        Initialize voting engine.
        
        Args:
            config: Voting configuration (uses defaults if None)
        """
        self.config = config if config else VotingConfig()
        
        # Validate configuration
        validation = self.config.validate()
        if validation.is_err():
            # Since we can't use exceptions, we'll store the error
            # and check it before executing
            self._config_error = validation.error
        else:
            self._config_error = None
        
        # Initialize components
        self.maker_voting = MAKERVoting(self.config)
        self.semantic_checker = SemanticChecker(
            use_test_based=self.config.enable_semantic_checking,
            use_ast_based=True
        )
        self.consensus_manager = ConsensusManager(self.config)
        self.fallback_manager = FallbackManager(self.config)
        
        # Diversity configuration
        self.available_models = ["gpt-4", "claude-3", "gemini-pro"]
        self.system_prompt_variants = self._generate_system_prompts()
    
    def execute_with_voting(
        self,
        task_spec: Any,
        agent_executor: Callable[[AgentConfig, Any], Result[Any, str]],
        result_validator: Callable[[Any, Any], Result[bool, str]],
        test_cases: Optional[List] = None
    ) -> VotingOutcome:
        """
        Execute task with multi-agent voting.
        
        This is the main entry point for the voting system. It orchestrates
        the entire voting process from agent execution through consensus
        detection and fallback handling.
        
        Args:
            task_spec: Specification of the task to execute
            agent_executor: Function that executes an agent and returns result
            result_validator: Function that validates a result
            test_cases: Optional test cases for semantic checking
        
        Returns:
            VotingOutcome with final result and metadata
        """
        # Check for configuration errors
        if self._config_error:
            return self._create_error_outcome(
                f"Invalid configuration: {self._config_error}"
            )
        
        # Reset consensus manager
        self.consensus_manager.reset()
        
        # Track timing
        start_time_ms = int(time.time() * 1000)
        
        # Main voting loop
        diversity_index = 0
        while diversity_index < self.config.max_attempts:
            # Check timeout
            elapsed_ms = int(time.time() * 1000) - start_time_ms
            if self.maker_voting.check_timeout(
                self.consensus_manager.state,
                elapsed_ms
            ):
                return self._handle_timeout(start_time_ms)
            
            # Spawn diverse agent
            agent_config = self._spawn_diverse_agent(diversity_index)
            
            # Execute agent
            execution_result = agent_executor(agent_config, task_spec)
            
            # Check if execution failed
            if execution_result.is_err():
                diversity_index += 1
                continue
            
            output = execution_result.unwrap()
            
            # Validate result
            validation_result = result_validator(output, task_spec)
            if validation_result.is_err() or not validation_result.unwrap():
                # Result is invalid or red-flagged
                self.consensus_manager.increment_red_flagged()
                diversity_index += 1
                continue
            
            # Compute semantic signature
            signature_result = self.semantic_checker.compute_signature(
                output,
                test_cases
            )
            
            if signature_result.is_err():
                # Failed to compute signature, skip this result
                self.consensus_manager.increment_red_flagged()
                diversity_index += 1
                continue
            
            # Create vote result
            vote = VoteResult(
                agent_id=f"agent_{diversity_index}",
                output=output,
                semantic_signature=signature_result.unwrap(),
                quality_score=self._calculate_quality_score(output),
                execution_time_ms=int(time.time() * 1000) - start_time_ms,
                metadata={
                    'model': agent_config.model_name,
                    'temperature': agent_config.temperature,
                    'diversity_index': diversity_index
                }
            )
            
            # Update votes
            update_result = self.consensus_manager.update_votes(vote)
            if update_result.is_err():
                diversity_index += 1
                continue
            
            # Check for consensus
            consensus_result = self.consensus_manager.check_consensus()
            if consensus_result.is_ok() and consensus_result.unwrap():
                # Consensus achieved!
                return self._create_success_outcome(start_time_ms)
            
            diversity_index += 1
        
        # Max attempts reached without consensus - try fallbacks
        return self._handle_no_consensus(start_time_ms, task_spec, agent_executor)
    
    def _spawn_diverse_agent(self, diversity_index: int) -> AgentConfig:
        """
        Create configuration for a diverse agent.
        
        Varies model, temperature, and system prompt to decorrelate errors.
        
        Args:
            diversity_index: Index for tracking diversity
        
        Returns:
            AgentConfig with diverse settings
        """
        # Rotate through models
        model_name = self.available_models[
            diversity_index % len(self.available_models)
        ] if self.config.diversity_model_rotation else self.available_models[0]
        
        # Vary temperature
        temp_min, temp_max = self.config.diversity_temperature_range
        temp_range = temp_max - temp_min
        temperature = temp_min + (diversity_index % 5) * (temp_range / 5)
        
        # Rotate through system prompts
        system_prompt = self.system_prompt_variants[
            diversity_index % len(self.system_prompt_variants)
        ]
        
        return AgentConfig(
            model_name=model_name,
            temperature=temperature,
            system_prompt=system_prompt,
            diversity_index=diversity_index
        )
    
    def _generate_system_prompts(self) -> List[str]:
        """
        Generate diverse system prompt variants.
        
        Returns:
            List of system prompt strings
        """
        base_prompt = "You are an expert software engineer."
        
        return [
            base_prompt + " Focus on code clarity and readability.",
            base_prompt + " Focus on performance and efficiency.",
            base_prompt + " Focus on correctness and safety.",
            base_prompt + " Focus on simplicity and maintainability.",
            base_prompt + " Focus on following best practices."
        ]
    
    def _calculate_quality_score(self, output: Any) -> float:
        """
        Calculate quality score for an output.
        
        Higher scores indicate better quality. Based on:
        - Code length (shorter is better for atomic tasks)
        - Complexity (lower is better)
        - Readability heuristics
        
        Args:
            output: The output to score
        
        Returns:
            Quality score (higher is better)
        """
        output_str = str(output)
        
        score = 100.0
        
        # Penalize excessive length
        line_count = output_str.count('\n') + 1
        if line_count > 20:
            score -= (line_count - 20) * 2
        
        # Penalize excessive nesting
        max_indent = 0
        for line in output_str.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 16:  # More than 4 levels of indentation
            score -= (max_indent - 16) * 3
        
        # Reward documentation
        if '"""' in output_str or "'''" in output_str:
            score += 10
        
        # Ensure score is non-negative
        return max(score, 0.0)
    
    def _create_success_outcome(self, start_time_ms: int) -> VotingOutcome:
        """
        Create successful voting outcome.
        
        Args:
            start_time_ms: Start time in milliseconds
        
        Returns:
            VotingOutcome indicating success
        """
        # Get winning result
        winning_group_result = self.consensus_manager.get_winning_group()
        
        if winning_group_result.is_err():
            return self._create_error_outcome(
                f"Failed to get winning group: {winning_group_result.error}"
            )
        
        winning_votes = winning_group_result.unwrap()
        
        # Select best quality from winning group
        best_vote = self._select_best_quality(winning_votes)
        
        # Calculate confidence
        confidence = self.consensus_manager.calculate_confidence()
        self.consensus_manager.state.confidence_score = confidence
        
        # Calculate total time
        total_time_ms = int(time.time() * 1000) - start_time_ms
        
        return VotingOutcome(
            success=True,
            result=best_vote,
            consensus_state=self.consensus_manager.get_state(),
            fallback_used=None,
            total_time_ms=total_time_ms,
            message=f"Consensus achieved with {len(winning_votes)} votes (confidence: {confidence:.2f})"
        )
    
    def _select_best_quality(self, votes: List[VoteResult]) -> VoteResult:
        """
        Select best quality vote from a list.
        
        Args:
            votes: List of vote results
        
        Returns:
            Highest quality vote
        """
        if len(votes) == 1:
            return votes[0]
        
        sorted_votes = sorted(
            votes,
            key=lambda v: (-v.quality_score, v.execution_time_ms)
        )
        
        return sorted_votes[0]
    
    def _handle_no_consensus(
        self,
        start_time_ms: int,
        task_spec: Any,
        agent_executor: Callable
    ) -> VotingOutcome:
        """
        Handle case where consensus was not reached.
        
        Attempts fallback strategies.
        
        Args:
            start_time_ms: Start time in milliseconds
            task_spec: Task specification
            agent_executor: Agent execution function
        
        Returns:
            VotingOutcome from fallback
        """
        # Try fallback strategies
        fallback_result = self.fallback_manager.execute_fallback(
            self.consensus_manager.get_state(),
            task_spec,
            agent_spawner=None  # Would be implemented for increase_agent_pool
        )
        
        total_time_ms = int(time.time() * 1000) - start_time_ms
        
        if fallback_result.is_ok():
            outcome = fallback_result.unwrap()
            outcome.total_time_ms = total_time_ms
            return outcome
        
        # All fallbacks failed
        return VotingOutcome(
            success=False,
            result=None,
            consensus_state=self.consensus_manager.get_state(),
            fallback_used=None,
            total_time_ms=total_time_ms,
            message=f"Failed to achieve consensus after {self.config.max_attempts} attempts"
        )
    
    def _handle_timeout(self, start_time_ms: int) -> VotingOutcome:
        """
        Handle voting timeout.
        
        Args:
            start_time_ms: Start time in milliseconds
        
        Returns:
            VotingOutcome indicating timeout
        """
        self.consensus_manager.set_status(ConsensusStatus.TIMEOUT)
        
        total_time_ms = int(time.time() * 1000) - start_time_ms
        
        return VotingOutcome(
            success=False,
            result=None,
            consensus_state=self.consensus_manager.get_state(),
            fallback_used=None,
            total_time_ms=total_time_ms,
            message=f"Voting timeout after {self.config.timeout_seconds} seconds"
        )
    
    def _create_error_outcome(self, error_message: str) -> VotingOutcome:
        """
        Create error outcome.
        
        Args:
            error_message: Error description
        
        Returns:
            VotingOutcome indicating error
        """
        return VotingOutcome(
            success=False,
            result=None,
            consensus_state=self.consensus_manager.get_state(),
            fallback_used=None,
            total_time_ms=0,
            message=error_message
        )
