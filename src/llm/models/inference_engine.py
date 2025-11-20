"""
Inference Engine Module.

This module provides the core inference engine for LLM models. It handles:
- Text generation with various sampling strategies
- Streaming generation support
- Batch inference for multiple prompts
- Token-level control and manipulation
- Generation parameters (temperature, top_p, top_k, etc.)
- Stop sequences and length constraints
- Token counting and budget management

The inference engine is backend-agnostic and works with models loaded
via the model_loader module.

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class SamplingStrategy(Enum):
    """
    Text generation sampling strategies.

    Attributes:
        GREEDY: Always select most likely token (deterministic)
        NUCLEUS: Nucleus sampling (top-p)
        TOP_K: Top-k sampling
        TEMPERATURE: Temperature-based sampling
        BEAM_SEARCH: Beam search for best sequences
    """
    GREEDY = "greedy"
    NUCLEUS = "nucleus"
    TOP_K = "top_k"
    TEMPERATURE = "temperature"
    BEAM_SEARCH = "beam_search"


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        min_new_tokens: Minimum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling threshold (0.0-1.0)
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
        length_penalty: Penalty for sequence length in beam search
        num_beams: Number of beams for beam search
        num_return_sequences: Number of sequences to return
        do_sample: Whether to use sampling (False = greedy)
        early_stopping: Stop generation when num_beams sentences are finished
        stop_sequences: List of sequences that stop generation
        pad_token_id: Token ID for padding
        eos_token_id: Token ID for end-of-sequence
        use_cache: Use KV cache for faster generation
    """
    max_new_tokens: int = 512
    min_new_tokens: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    num_return_sequences: int = 1
    do_sample: bool = True
    early_stopping: bool = False
    stop_sequences: List[str] = field(default_factory=list)
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True

    def validate(self) -> bool:
        """
        Validate generation configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate token counts
        if self.max_new_tokens <= 0:
            return False
        if self.min_new_tokens < 0:
            return False
        if self.min_new_tokens > self.max_new_tokens:
            return False

        # Validate temperature
        if self.temperature < 0.0:
            return False

        # Validate top_p
        if self.top_p < 0.0 or self.top_p > 1.0:
            return False

        # Validate top_k
        if self.top_k <= 0:
            return False

        # Validate penalties
        if self.repetition_penalty < 0.0:
            return False
        if self.length_penalty < 0.0:
            return False

        # Validate beam search parameters
        if self.num_beams <= 0:
            return False
        if self.num_return_sequences <= 0:
            return False
        if self.num_return_sequences > self.num_beams:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": self.min_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "stop_sequences": self.stop_sequences,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "use_cache": self.use_cache
        }


@dataclass
class GenerationResult:
    """
    Result of text generation.

    Attributes:
        text: Generated text
        tokens: Generated token IDs
        finish_reason: Reason generation stopped (length/eos/stop_sequence)
        num_tokens: Number of tokens generated
        generation_time_ms: Time taken for generation in milliseconds
        tokens_per_second: Generation speed in tokens/second
        prompt_tokens: Number of tokens in prompt
        total_tokens: Total tokens (prompt + generated)
        metadata: Additional metadata about generation
    """
    text: str
    tokens: List[int]
    finish_reason: str
    num_tokens: int
    generation_time_ms: float
    tokens_per_second: float
    prompt_tokens: int
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchGenerationResult:
    """
    Result of batch text generation.

    Attributes:
        results: List of generation results
        total_time_ms: Total time for batch
        average_tokens_per_second: Average generation speed
    """
    results: List[GenerationResult]
    total_time_ms: float
    average_tokens_per_second: float


class InferenceEngine:
    """
    Core inference engine for LLM text generation.

    Provides methods for single and batch text generation with
    various sampling strategies and configuration options.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        default_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize inference engine.

        Args:
            model: Loaded model object
            tokenizer: Tokenizer object
            default_config: Default generation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.default_config = default_config or GenerationConfig()

        # Statistics
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time_ms = 0.0

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Optional[GenerationResult]:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input prompt text
            config: Generation configuration (uses default if not provided)

        Returns:
            GenerationResult if successful, None if failed
        """
        # Validate prompt
        if not prompt:
            return None

        # Use provided config or default
        gen_config = config or self.default_config
        if not gen_config.validate():
            return None

        # Measure generation time
        start_time = time.time()

        # Simulate generation (in production, would call actual model)
        result = self._simulate_generation(prompt, gen_config)

        end_time = time.time()
        generation_time_ms = (end_time - start_time) * 1000

        if result is None:
            return None

        # Update statistics
        self.total_generations += 1
        self.total_tokens_generated += result.num_tokens
        self.total_generation_time_ms += generation_time_ms

        return result

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> Optional[BatchGenerationResult]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            BatchGenerationResult if successful, None if failed
        """
        # Validate prompts
        if not prompts:
            return None

        # Validate config
        gen_config = config or self.default_config
        if not gen_config.validate():
            return None

        # Measure total time
        start_time = time.time()

        # Generate for each prompt
        results: List[GenerationResult] = []
        for prompt in prompts:
            result = self.generate(prompt, gen_config)
            if result is None:
                # Failed generation - skip
                continue
            results.append(result)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # Calculate average speed
        total_tokens = sum(r.num_tokens for r in results)
        avg_tokens_per_sec = (total_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0.0

        return BatchGenerationResult(
            results=results,
            total_time_ms=total_time_ms,
            average_tokens_per_second=avg_tokens_per_sec
        )

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Optional[Iterator[str]]:
        """
        Generate text with streaming output (token by token).

        Args:
            prompt: Input prompt text
            config: Generation configuration

        Yields:
            Generated text chunks

        Returns:
            None if generation fails
        """
        # Validate
        if not prompt:
            return None

        gen_config = config or self.default_config
        if not gen_config.validate():
            return None

        # In production, would implement actual streaming
        # For now, simulate by yielding chunks
        def stream_generator() -> Iterator[str]:
            # Simulate streaming
            full_text = f"Generated response for: {prompt}"
            words = full_text.split()

            for word in words:
                yield word + " "
                time.sleep(0.01)  # Simulate token generation delay

        return stream_generator()

    def count_tokens(self, text: str) -> Optional[int]:
        """
        Count number of tokens in text.

        Args:
            text: Input text

        Returns:
            Token count, or None if failed
        """
        if not text:
            return 0

        # In production, would use actual tokenizer
        # For now, estimate based on words
        words = text.split()
        return len(words)

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int
    ) -> Optional[str]:
        """
        Truncate text to fit within token limit.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text, or None if failed
        """
        if not text or max_tokens <= 0:
            return None

        # Count tokens
        token_count = self.count_tokens(text)
        if token_count is None:
            return None

        # Already within limit
        if token_count <= max_tokens:
            return text

        # Truncate (rough approximation)
        words = text.split()
        ratio = max_tokens / token_count
        truncated_words = words[:int(len(words) * ratio)]

        return " ".join(truncated_words)

    def _simulate_generation(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> Optional[GenerationResult]:
        """
        Simulate text generation.

        In production, this would call the actual model.
        This placeholder demonstrates the interface.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            GenerationResult
        """
        # Simulate generation
        generated_text = f"[Generated response for prompt: {prompt[:50]}...]"

        # Simulate token IDs
        tokens = [i for i in range(10)]  # Placeholder

        # Calculate metrics
        prompt_tokens = self.count_tokens(prompt) or 0
        num_tokens = len(tokens)
        total_tokens = prompt_tokens + num_tokens

        # Simulate time
        generation_time_ms = num_tokens * 10.0  # 10ms per token
        tokens_per_second = (num_tokens / (generation_time_ms / 1000)) if generation_time_ms > 0 else 0.0

        return GenerationResult(
            text=generated_text,
            tokens=tokens,
            finish_reason="length",
            num_tokens=num_tokens,
            generation_time_ms=generation_time_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            metadata={
                "config": config.to_dict(),
                "model_name": "simulated"
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get inference engine statistics.

        Returns:
            Dictionary of statistics
        """
        avg_tokens_per_gen = (
            self.total_tokens_generated / self.total_generations
            if self.total_generations > 0 else 0.0
        )

        avg_time_per_gen = (
            self.total_generation_time_ms / self.total_generations
            if self.total_generations > 0 else 0.0
        )

        overall_tokens_per_sec = (
            self.total_tokens_generated / (self.total_generation_time_ms / 1000)
            if self.total_generation_time_ms > 0 else 0.0
        )

        return {
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time_ms": self.total_generation_time_ms,
            "average_tokens_per_generation": avg_tokens_per_gen,
            "average_time_per_generation_ms": avg_time_per_gen,
            "overall_tokens_per_second": overall_tokens_per_sec
        }

    def reset_statistics(self) -> None:
        """Reset inference engine statistics."""
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_generation_time_ms = 0.0


def create_default_generation_config(
    max_tokens: int = 512,
    temperature: float = 0.7
) -> GenerationConfig:
    """
    Create default generation configuration with common settings.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        GenerationConfig with defaults
    """
    return GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1
    )


def create_deterministic_config(max_tokens: int = 512) -> GenerationConfig:
    """
    Create deterministic generation configuration (greedy decoding).

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        GenerationConfig for deterministic generation
    """
    return GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=0.0,
        do_sample=False,
        num_beams=1
    )


def create_creative_config(max_tokens: int = 1024) -> GenerationConfig:
    """
    Create creative generation configuration (high temperature, nucleus sampling).

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        GenerationConfig for creative generation
    """
    return GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=1.0,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2
    )
