"""
Base Agent Infrastructure.

This module provides the abstract base class for all agent archetypes in the
zero-error system. Agents are ephemeral, stateless executors that perform
specific tasks using LLM inference.

Key Concepts:
- An "agent" is a single LLM inference call, not a persistent process
- Agents are stateless and ephemeral
- All agents share the same LLM model pool (no VRAM duplication)
- Diversity is achieved through temperature, prompts, and model rotation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
import hashlib

from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import VoteResult, AgentConfig


class LLMPool(Protocol):
    """
    Protocol for LLM model pool interface.

    This defines the interface that LLM infrastructure must implement.
    Agents don't care about the underlying implementation.
    """

    def generate(
        self,
        prompt: str,
        model_name: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using LLM.

        Args:
            prompt: The input prompt
            model_name: Name of model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences

        Returns:
            Generated text
        """
        ...


@dataclass
class AgentExecutionContext:
    """
    Context for agent execution.

    Contains all information needed for an agent to execute a task,
    including the task specification, configuration, and runtime context.

    Attributes:
        task: The task specification to execute
        config: Agent configuration (model, temperature, etc.)
        llm_pool: Shared LLM pool for inference
        context_data: Additional context data
        execution_id: Unique identifier for this execution
        start_time: Execution start timestamp
    """
    task: TaskSpecification
    config: AgentConfig
    llm_pool: LLMPool
    context_data: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = field(default_factory=lambda: hashlib.sha256(
        f"{datetime.now().timestamp()}".encode()
    ).hexdigest()[:16])
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())

    def get_elapsed_time_ms(self) -> int:
        """Get elapsed execution time in milliseconds."""
        elapsed = datetime.now().timestamp() - self.start_time
        return int(elapsed * 1000)

    def validate(self) -> bool:
        """Validate execution context."""
        if not self.task:
            return False
        if not self.task.is_valid():
            return False
        if not self.config:
            return False
        if not self.llm_pool:
            return False
        return True


@dataclass
class AgentExecutionResult:
    """
    Result of agent execution.

    Encapsulates the output, metadata, and quality metrics from
    an agent's execution.

    Attributes:
        success: Whether execution succeeded
        output: The generated output
        agent_id: Unique agent identifier
        execution_time_ms: Execution time in milliseconds
        quality_score: Quality score (0.0 to 1.0)
        metadata: Additional metadata
        error_message: Error message if failed
    """
    success: bool
    output: Optional[Any]
    agent_id: str
    execution_time_ms: int
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_vote_result(self, semantic_signature: str) -> VoteResult:
        """
        Convert to VoteResult for voting system.

        Args:
            semantic_signature: Semantic equivalence signature

        Returns:
            VoteResult instance
        """
        return VoteResult(
            agent_id=self.agent_id,
            output=self.output,
            semantic_signature=semantic_signature,
            quality_score=self.quality_score,
            execution_time_ms=self.execution_time_ms,
            metadata=self.metadata
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agent archetypes.

    This class defines the interface and common functionality for all agents
    in the system. Each agent archetype inherits from this and implements
    specific behavior for their task type.

    Design Principles:
    - Stateless: Agents don't maintain state between executions
    - Ephemeral: Each execution is independent
    - Composable: Agents can be combined and coordinated
    - Type-safe: Full type annotations and validation

    Attributes:
        agent_type: The archetype type of this agent
        system_prompt_template: Template for system prompts
        default_temperature: Default temperature for this agent type
        default_max_tokens: Default maximum tokens to generate
    """

    def __init__(
        self,
        agent_type: str,
        system_prompt_template: str,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512
    ):
        """
        Initialize base agent.

        Args:
            agent_type: Type identifier for this agent
            system_prompt_template: Template for system prompts
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens to generate
        """
        self.agent_type = agent_type
        self.system_prompt_template = system_prompt_template
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        # Validation
        if not self.validate_configuration():
            raise ValueError(f"Invalid agent configuration for {agent_type}")

    def validate_configuration(self) -> bool:
        """
        Validate agent configuration.

        Returns:
            True if configuration is valid
        """
        if not self.agent_type or not self.agent_type.strip():
            return False
        if not self.system_prompt_template:
            return False
        if not (0.0 <= self.default_temperature <= 2.0):
            return False
        if self.default_max_tokens <= 0:
            return False
        return True

    def execute(self, context: AgentExecutionContext) -> AgentExecutionResult:
        """
        Execute the agent with given context.

        This is the main entry point for agent execution. It orchestrates
        the complete execution flow with proper error handling.

        Args:
            context: Execution context containing task and configuration

        Returns:
            AgentExecutionResult with output or error
        """
        # Validate context
        if not context.validate():
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id=self._generate_agent_id(context),
                execution_time_ms=context.get_elapsed_time_ms(),
                error_message="Invalid execution context"
            )

        # Validate task is appropriate for this agent
        if not self.can_handle_task(context.task):
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id=self._generate_agent_id(context),
                execution_time_ms=context.get_elapsed_time_ms(),
                error_message=f"Agent {self.agent_type} cannot handle task type {context.task.task_type}"
            )

        # Generate prompt
        prompt = self.generate_prompt(context.task, context.config, context.context_data)
        if not prompt or not prompt.strip():
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id=self._generate_agent_id(context),
                execution_time_ms=context.get_elapsed_time_ms(),
                error_message="Failed to generate prompt"
            )

        # Call LLM
        llm_output = self._call_llm(context, prompt)
        if llm_output is None:
            return AgentExecutionResult(
                success=False,
                output=None,
                agent_id=self._generate_agent_id(context),
                execution_time_ms=context.get_elapsed_time_ms(),
                error_message="LLM inference failed"
            )

        # Process output
        processed_output = self.process_output(llm_output, context.task)

        # Validate output
        validation_result = self.validate_output(processed_output, context.task)
        if not validation_result:
            return AgentExecutionResult(
                success=False,
                output=processed_output,
                agent_id=self._generate_agent_id(context),
                execution_time_ms=context.get_elapsed_time_ms(),
                error_message="Output validation failed"
            )

        # Calculate quality score
        quality_score = self.calculate_quality_score(processed_output, context.task)

        # Success
        return AgentExecutionResult(
            success=True,
            output=processed_output,
            agent_id=self._generate_agent_id(context),
            execution_time_ms=context.get_elapsed_time_ms(),
            quality_score=quality_score,
            metadata=self._gather_metadata(context, processed_output)
        )

    @abstractmethod
    def can_handle_task(self, task: TaskSpecification) -> bool:
        """
        Check if this agent can handle the given task.

        Args:
            task: Task specification

        Returns:
            True if agent can handle this task type
        """
        pass

    @abstractmethod
    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """
        Generate the prompt for LLM inference.

        This method must be implemented by each agent archetype to create
        task-specific prompts.

        Args:
            task: Task specification
            config: Agent configuration
            context_data: Additional context data

        Returns:
            Generated prompt string
        """
        pass

    @abstractmethod
    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """
        Process raw LLM output into structured result.

        Args:
            llm_output: Raw text from LLM
            task: Task specification

        Returns:
            Processed output in appropriate format
        """
        pass

    @abstractmethod
    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """
        Validate the processed output.

        Args:
            output: Processed output to validate
            task: Task specification

        Returns:
            True if output is valid
        """
        pass

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """
        Calculate quality score for the output.

        Default implementation provides basic scoring. Subclasses can override
        for task-specific quality metrics.

        Args:
            output: The output to score
            task: Task specification

        Returns:
            Quality score between 0.0 and 1.0
        """
        if output is None:
            return 0.0

        score = 0.5  # Base score

        # Bonus for non-empty output
        if output and str(output).strip():
            score += 0.2

        # Bonus for reasonable length
        output_str = str(output)
        if 10 <= len(output_str) <= task.max_lines * 100:
            score += 0.2

        # Bonus for no obvious issues
        if not self._has_obvious_issues(output_str):
            score += 0.1

        return min(1.0, max(0.0, score))

    def _call_llm(self, context: AgentExecutionContext, prompt: str) -> Optional[str]:
        """
        Call LLM with error handling.

        Args:
            context: Execution context
            prompt: Generated prompt

        Returns:
            LLM output or None if failed
        """
        # Validate inputs
        if not prompt or not prompt.strip():
            return None

        if not context.llm_pool:
            return None

        # Prepare parameters
        model_name = context.config.model_name
        temperature = context.config.temperature
        max_tokens = self.default_max_tokens

        # Validate parameters
        if not (0.0 <= temperature <= 2.0):
            temperature = self.default_temperature

        if max_tokens <= 0:
            max_tokens = 512

        # Call LLM (assumes LLM pool handles retries and error handling)
        output = context.llm_pool.generate(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return output if output else None

    def _generate_agent_id(self, context: AgentExecutionContext) -> str:
        """
        Generate unique agent ID.

        Args:
            context: Execution context

        Returns:
            Unique agent identifier
        """
        id_string = f"{self.agent_type}_{context.config.diversity_index}_{context.execution_id}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def _gather_metadata(
        self,
        context: AgentExecutionContext,
        output: Any
    ) -> Dict[str, Any]:
        """
        Gather execution metadata.

        Args:
            context: Execution context
            output: Generated output

        Returns:
            Metadata dictionary
        """
        return {
            'agent_type': self.agent_type,
            'task_id': context.task.id,
            'task_type': context.task.task_type.value,
            'model_name': context.config.model_name,
            'temperature': context.config.temperature,
            'diversity_index': context.config.diversity_index,
            'execution_id': context.execution_id,
            'output_length': len(str(output)) if output else 0
        }

    def _has_obvious_issues(self, text: str) -> bool:
        """
        Check for obvious issues in output.

        Args:
            text: Text to check

        Returns:
            True if obvious issues found
        """
        if not text:
            return True

        # Check for error patterns
        error_patterns = [
            'error:', 'exception:', 'failed:', 'cannot',
            'unable to', 'impossible', 'i don\'t know',
            'i cannot', 'i can\'t', 'todo', 'fixme'
        ]

        text_lower = text.lower()
        for pattern in error_patterns:
            if pattern in text_lower:
                return True

        return False

    def create_base_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig
    ) -> str:
        """
        Create base prompt with system instructions.

        This helper method creates the foundation of the prompt that all
        agents can build upon.

        Args:
            task: Task specification
            config: Agent configuration

        Returns:
            Base prompt string
        """
        # Start with system prompt
        prompt = config.system_prompt

        # Add task information
        prompt += f"\n\nTask: {task.name}\n"
        prompt += f"Description: {task.description}\n"

        # Add inputs if present
        if task.inputs:
            prompt += "\nInputs:\n"
            for inp in task.inputs:
                prompt += f"  - {inp.name}: {inp.type_annotation}\n"
                prompt += f"    {inp.description}\n"

        # Add outputs if present
        if task.outputs:
            prompt += "\nExpected Outputs:\n"
            for out in task.outputs:
                prompt += f"  - {out.name}: {out.type_annotation}\n"
                prompt += f"    {out.description}\n"

        # Add constraints
        prompt += f"\nConstraints:\n"
        prompt += f"  - Maximum {task.max_lines} lines of code\n"
        prompt += f"  - Maximum cyclomatic complexity: {task.max_complexity}\n"

        # Add preconditions if present
        if task.preconditions:
            prompt += "\nPreconditions:\n"
            for pre in task.preconditions:
                prompt += f"  - {pre.name}: {pre.expression}\n"

        # Add postconditions if present
        if task.postconditions:
            prompt += "\nPostconditions:\n"
            for post in task.postconditions:
                prompt += f"  - {post.name}: {post.expression}\n"

        return prompt
