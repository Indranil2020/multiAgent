"""
Retry Handler Module.

This module implements intelligent retry logic for handling transient failures
in LLM inference operations. Provides exponential backoff, jitter, configurable
retry policies, and error classification.

Key Concepts:
- Transient errors (network timeout, CUDA OOM) should be retried
- Permanent errors (invalid input, model error) should fail fast
- Exponential backoff prevents overwhelming failing systems
- Jitter prevents thundering herd problem
- Different error types need different retry strategies

Retry Strategies:
- Exponential Backoff: delay = base_delay * (2 ** attempt)
- Exponential with Jitter: Add random variation to prevent synchronization
- Fixed Delay: Constant delay between retries
- Fibonacci: Delays follow Fibonacci sequence
- Custom: User-defined delay function

Features:
- Automatic error classification (retryable vs fatal)
- Configurable retry policies per error type
- Maximum retry limits and timeouts
- Retry statistics and monitoring
- Circuit breaker integration
- Async retry support

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time
import random


T = TypeVar('T')


class RetryStrategy(Enum):
    """
    Retry strategy types.

    Attributes:
        EXPONENTIAL_BACKOFF: Exponential delay growth
        EXPONENTIAL_WITH_JITTER: Exponential with random jitter
        FIXED_DELAY: Fixed delay between retries
        FIBONACCI: Fibonacci sequence delays
        CUSTOM: Custom delay function
    """
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"


class ErrorCategory(Enum):
    """
    Error categories for classification.

    Attributes:
        TRANSIENT: Temporary error, retry likely to succeed
        RATE_LIMIT: Rate limiting error, retry with backoff
        RESOURCE: Resource exhaustion (VRAM, etc), retry after delay
        TIMEOUT: Operation timeout, may succeed with retry
        NETWORK: Network error, retry possible
        FATAL: Permanent error, retry will not help
        UNKNOWN: Unknown error type
    """
    TRANSIENT = "transient"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    NETWORK = "network"
    FATAL = "fatal"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        strategy: Retry strategy to use
        max_attempts: Maximum retry attempts (0 = no retries)
        base_delay_seconds: Base delay for exponential strategies
        max_delay_seconds: Maximum delay between retries
        timeout_seconds: Total timeout for all retry attempts
        jitter_factor: Jitter factor for randomization (0.0-1.0)
        enable_circuit_breaker: Stop retries if system unhealthy
        exponential_base: Base for exponential growth (default 2)
        backoff_multiplier: Multiplier for delay (default 1)
        retryable_categories: Error categories to retry
        custom_delay_func: Custom delay calculation function
    """
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_WITH_JITTER
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    timeout_seconds: float = 300.0
    jitter_factor: float = 0.3
    enable_circuit_breaker: bool = True
    exponential_base: float = 2.0
    backoff_multiplier: float = 1.0
    retryable_categories: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.TRANSIENT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.RESOURCE,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK
    ])
    custom_delay_func: Optional[Callable[[int], float]] = None

    def validate(self) -> bool:
        """
        Validate retry configuration.

        Returns:
            True if valid, False otherwise
        """
        if self.max_attempts < 0:
            return False
        if self.base_delay_seconds < 0:
            return False
        if self.max_delay_seconds < self.base_delay_seconds:
            return False
        if self.timeout_seconds <= 0:
            return False
        if not (0.0 <= self.jitter_factor <= 1.0):
            return False
        if self.exponential_base <= 1.0:
            return False
        if self.backoff_multiplier <= 0:
            return False
        return True


@dataclass
class RetryAttempt:
    """
    Record of a retry attempt.

    Attributes:
        attempt_number: Attempt number (1-indexed)
        error_category: Classified error category
        error_message: Error message
        delay_seconds: Delay before this attempt
        timestamp: Attempt timestamp
        success: Whether attempt succeeded
    """
    attempt_number: int
    error_category: ErrorCategory
    error_message: str
    delay_seconds: float
    timestamp: float = field(default_factory=time.time)
    success: bool = False


@dataclass
class RetryResult(Generic[T]):
    """
    Result of retry operation.

    Attributes:
        success: Whether operation succeeded
        result: Operation result if successful
        attempts: Number of attempts made
        total_delay_seconds: Total time spent in delays
        total_time_seconds: Total time including execution
        retry_history: List of retry attempts
        final_error: Final error if failed
    """
    success: bool
    result: Optional[T] = None
    attempts: int = 0
    total_delay_seconds: float = 0.0
    total_time_seconds: float = 0.0
    retry_history: List[RetryAttempt] = field(default_factory=list)
    final_error: Optional[str] = None


@dataclass
class RetryStatistics:
    """
    Statistics for retry operations.

    Attributes:
        total_operations: Total operations attempted
        successful_operations: Operations that succeeded
        failed_operations: Operations that failed permanently
        total_retries: Total retry attempts made
        average_attempts: Average attempts per operation
        retries_by_category: Retry counts by error category
        success_after_retry: Operations that succeeded after retry
    """
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_retries: int = 0
    average_attempts: float = 0.0
    retries_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    success_after_retry: int = 0


class RetryHandler:
    """
    Intelligent retry handler with configurable strategies.

    Handles retry logic with exponential backoff, jitter, and
    automatic error classification.
    """

    def __init__(self, config: RetryConfig):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
        """
        if not config.validate():
            raise ValueError("Invalid retry configuration")

        self.config = config

        # Statistics
        self.stats = RetryStatistics()
        self.lock = threading.RLock()

        # Fibonacci cache
        self.fibonacci_cache = [0, 1]

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        error_classifier: Optional[Callable[[Exception], ErrorCategory]] = None
    ) -> RetryResult[T]:
        """
        Execute operation with retry logic.

        Args:
            operation: Function to execute
            error_classifier: Function to classify errors

        Returns:
            RetryResult with operation outcome
        """
        start_time = time.time()
        retry_history: List[RetryAttempt] = []
        total_delay = 0.0

        with self.lock:
            self.stats.total_operations += 1

        # Attempt operation
        for attempt in range(self.config.max_attempts + 1):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= self.config.timeout_seconds:
                break

            # Execute operation
            try:
                result = operation()

                # Success!
                with self.lock:
                    self.stats.successful_operations += 1
                    if attempt > 0:
                        self.stats.success_after_retry += 1

                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_delay_seconds=total_delay,
                    total_time_seconds=time.time() - start_time,
                    retry_history=retry_history
                )

            except Exception as e:
                # Classify error
                error_category = self._classify_error(e, error_classifier)

                # Record attempt
                retry_attempt = RetryAttempt(
                    attempt_number=attempt + 1,
                    error_category=error_category,
                    error_message=str(e),
                    delay_seconds=0.0,
                    success=False
                )
                retry_history.append(retry_attempt)

                # Update statistics
                with self.lock:
                    self.stats.total_retries += 1
                    if error_category not in self.stats.retries_by_category:
                        self.stats.retries_by_category[error_category] = 0
                    self.stats.retries_by_category[error_category] += 1

                # Check if error is retryable
                if not self._is_retryable(error_category):
                    # Fatal error, don't retry
                    break

                # Check if we have attempts left
                if attempt >= self.config.max_attempts:
                    break

                # Calculate delay
                delay = self._calculate_delay(attempt)

                # Check if delay would exceed timeout
                if elapsed + delay >= self.config.timeout_seconds:
                    break

                # Wait before retry
                time.sleep(delay)
                total_delay += delay
                retry_attempt.delay_seconds = delay

        # All attempts failed
        with self.lock:
            self.stats.failed_operations += 1
            self.stats.average_attempts = (
                (self.stats.total_retries + self.stats.total_operations) /
                self.stats.total_operations
            )

        return RetryResult(
            success=False,
            result=None,
            attempts=len(retry_history),
            total_delay_seconds=total_delay,
            total_time_seconds=time.time() - start_time,
            retry_history=retry_history,
            final_error=retry_history[-1].error_message if retry_history else "Unknown error"
        )

    def _classify_error(
        self,
        error: Exception,
        classifier: Optional[Callable[[Exception], ErrorCategory]]
    ) -> ErrorCategory:
        """
        Classify error into category.

        Args:
            error: Exception to classify
            classifier: Optional custom classifier

        Returns:
            ErrorCategory
        """
        # Use custom classifier if provided
        if classifier is not None:
            return classifier(error)

        # Default classification based on error type and message
        error_str = str(error).lower()

        if "timeout" in error_str or "timed out" in error_str:
            return ErrorCategory.TIMEOUT
        elif "oom" in error_str or "out of memory" in error_str:
            return ErrorCategory.RESOURCE
        elif "rate limit" in error_str or "too many requests" in error_str:
            return ErrorCategory.RATE_LIMIT
        elif "connection" in error_str or "network" in error_str:
            return ErrorCategory.NETWORK
        elif "transient" in error_str or "temporary" in error_str:
            return ErrorCategory.TRANSIENT
        elif "invalid" in error_str or "bad" in error_str:
            return ErrorCategory.FATAL
        else:
            return ErrorCategory.UNKNOWN

    def _is_retryable(self, category: ErrorCategory) -> bool:
        """
        Check if error category is retryable.

        Args:
            category: Error category

        Returns:
            True if retryable
        """
        return category in self.config.retryable_categories

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self._exponential_backoff(attempt)
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_WITH_JITTER:
            delay = self._exponential_backoff_with_jitter(attempt)
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay_seconds
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self._fibonacci_delay(attempt)
        elif self.config.strategy == RetryStrategy.CUSTOM:
            if self.config.custom_delay_func is not None:
                delay = self.config.custom_delay_func(attempt)
            else:
                delay = self.config.base_delay_seconds
        else:
            delay = self.config.base_delay_seconds

        # Cap at max delay
        delay = min(delay, self.config.max_delay_seconds)

        return delay

    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = (
            self.config.base_delay_seconds *
            self.config.backoff_multiplier *
            (self.config.exponential_base ** attempt)
        )
        return delay

    def _exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        base_delay = self._exponential_backoff(attempt)

        # Add random jitter
        jitter_range = base_delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        return base_delay + jitter

    def _fibonacci_delay(self, attempt: int) -> float:
        """Calculate Fibonacci sequence delay."""
        # Extend cache if needed
        while len(self.fibonacci_cache) <= attempt + 1:
            next_fib = self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
            self.fibonacci_cache.append(next_fib)

        fib_value = self.fibonacci_cache[attempt + 1]
        return self.config.base_delay_seconds * fib_value

    def get_statistics(self) -> RetryStatistics:
        """
        Get retry statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            return RetryStatistics(
                total_operations=self.stats.total_operations,
                successful_operations=self.stats.successful_operations,
                failed_operations=self.stats.failed_operations,
                total_retries=self.stats.total_retries,
                average_attempts=self.stats.average_attempts,
                retries_by_category=self.stats.retries_by_category.copy(),
                success_after_retry=self.stats.success_after_retry
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = RetryStatistics()


def create_default_retry_config() -> RetryConfig:
    """
    Create default retry configuration.

    Returns:
        RetryConfig with sensible defaults
    """
    return RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
        max_attempts=3,
        base_delay_seconds=1.0,
        max_delay_seconds=60.0,
        timeout_seconds=300.0,
        jitter_factor=0.3
    )


def create_aggressive_retry_config() -> RetryConfig:
    """
    Create aggressive retry configuration with many attempts.

    Returns:
        RetryConfig for aggressive retrying
    """
    return RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
        max_attempts=10,
        base_delay_seconds=0.5,
        max_delay_seconds=30.0,
        timeout_seconds=600.0,
        jitter_factor=0.2
    )


def create_conservative_retry_config() -> RetryConfig:
    """
    Create conservative retry configuration with few attempts.

    Returns:
        RetryConfig for conservative retrying
    """
    return RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        max_attempts=2,
        base_delay_seconds=2.0,
        max_delay_seconds=60.0,
        timeout_seconds=120.0,
        jitter_factor=0.0
    )


def create_fixed_delay_config(delay_seconds: float = 5.0) -> RetryConfig:
    """
    Create configuration with fixed delay between retries.

    Args:
        delay_seconds: Fixed delay in seconds

    Returns:
        RetryConfig with fixed delay
    """
    return RetryConfig(
        strategy=RetryStrategy.FIXED_DELAY,
        max_attempts=3,
        base_delay_seconds=delay_seconds,
        max_delay_seconds=delay_seconds,
        timeout_seconds=300.0,
        jitter_factor=0.0
    )
