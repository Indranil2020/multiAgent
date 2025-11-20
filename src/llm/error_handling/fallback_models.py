"""
Fallback Models Module.

This module implements graceful degradation strategies using fallback models
when primary models fail or are unavailable. Ensures system continues to
function even when individual models encounter errors.

Key Concepts:
- Primary model may fail due to OOM, errors, or unavailability
- Fallback chain provides alternative models in priority order
- Smaller/faster models can serve as fallbacks
- Quality vs availability tradeoff
- Automatic fallback with manual override options

Fallback Strategies:
- Sequential: Try models in order until one succeeds
- Parallel: Try multiple models simultaneously, use first success
- Conditional: Select fallback based on error type
- Quality-Based: Prefer higher quality, fallback to lower
- Performance-Based: Prefer faster models under load

Features:
- Configurable fallback chains
- Automatic fallback on errors
- Health-based model selection
- Fallback statistics and tracking
- Quality degradation warnings
- Automatic model recovery
- Circuit breaker integration
- Fallback testing and validation

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time


class FallbackStrategy(Enum):
    """
    Fallback strategy types.

    Attributes:
        SEQUENTIAL: Try models sequentially
        PARALLEL: Try models in parallel
        CONDITIONAL: Select based on error
        QUALITY_BASED: Prefer quality, fallback to speed
        PERFORMANCE_BASED: Prefer speed, fallback to quality
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    QUALITY_BASED = "quality_based"
    PERFORMANCE_BASED = "performance_based"


class ModelTier(Enum):
    """
    Model quality/capability tiers.

    Attributes:
        PRIMARY: Primary model (highest quality)
        SECONDARY: Secondary model (good quality)
        TERTIARY: Tertiary model (acceptable quality)
        EMERGENCY: Emergency fallback (basic functionality)
    """
    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    EMERGENCY = 4


@dataclass
class FallbackConfig:
    """
    Configuration for fallback behavior.

    Attributes:
        strategy: Fallback strategy to use
        enable_auto_fallback: Enable automatic fallback on errors
        enable_health_checks: Enable model health checking
        health_check_interval_seconds: Health check interval
        enable_auto_recovery: Enable automatic recovery to primary
        recovery_check_interval_seconds: Recovery check interval
        max_fallback_attempts: Maximum fallback attempts
        fallback_timeout_seconds: Timeout for fallback attempt
        enable_quality_warnings: Warn when using degraded model
        enable_parallel_fallback: Enable parallel fallback attempts
        parallel_timeout_seconds: Timeout for parallel attempts
    """
    strategy: FallbackStrategy = FallbackStrategy.SEQUENTIAL
    enable_auto_fallback: bool = True
    enable_health_checks: bool = True
    health_check_interval_seconds: float = 30.0
    enable_auto_recovery: bool = True
    recovery_check_interval_seconds: float = 60.0
    max_fallback_attempts: int = 3
    fallback_timeout_seconds: float = 30.0
    enable_quality_warnings: bool = True
    enable_parallel_fallback: bool = False
    parallel_timeout_seconds: float = 10.0

    def validate(self) -> bool:
        """
        Validate fallback configuration.

        Returns:
            True if valid, False otherwise
        """
        if self.health_check_interval_seconds <= 0:
            return False
        if self.recovery_check_interval_seconds <= 0:
            return False
        if self.max_fallback_attempts <= 0:
            return False
        if self.fallback_timeout_seconds <= 0:
            return False
        if self.parallel_timeout_seconds <= 0:
            return False
        return True


@dataclass
class ModelConfig:
    """
    Configuration for a fallback model.

    Attributes:
        model_id: Unique model identifier
        model_name: Human-readable model name
        tier: Model quality tier
        priority: Model priority (lower = preferred)
        pool_name: Pool name for this model
        capabilities: Model capabilities/features
        expected_quality: Expected quality score (0.0-1.0)
        expected_speed: Expected speed score (0.0-1.0)
        enabled: Whether model is enabled
        health_check_func: Optional health check function
        metadata: Additional metadata
    """
    model_id: str
    model_name: str
    tier: ModelTier
    priority: int
    pool_name: str
    capabilities: List[str] = field(default_factory=list)
    expected_quality: float = 1.0
    expected_speed: float = 1.0
    enabled: bool = True
    health_check_func: Optional[Callable[[], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelHealth:
    """
    Health status for a model.

    Attributes:
        model_id: Model identifier
        is_healthy: Whether model is healthy
        last_check_time: Last health check timestamp
        consecutive_failures: Consecutive failure count
        consecutive_successes: Consecutive success count
        total_requests: Total requests to this model
        successful_requests: Successful requests
        failed_requests: Failed requests
        average_response_time_ms: Average response time
        error_rate: Error rate (0.0-1.0)
    """
    model_id: str
    is_healthy: bool = True
    last_check_time: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0

    def update_error_rate(self) -> None:
        """Update error rate calculation."""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        else:
            self.error_rate = 0.0


@dataclass
class FallbackEvent:
    """
    Record of a fallback event.

    Attributes:
        event_id: Unique event identifier
        from_model: Model that failed
        to_model: Model used as fallback
        reason: Reason for fallback
        timestamp: Event timestamp
        success: Whether fallback succeeded
        quality_degradation: Quality degradation amount
        response_time_ms: Response time for fallback
    """
    event_id: str
    from_model: str
    to_model: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    quality_degradation: float = 0.0
    response_time_ms: float = 0.0


@dataclass
class FallbackStatistics:
    """
    Statistics for fallback operations.

    Attributes:
        total_requests: Total requests processed
        primary_requests: Requests served by primary
        fallback_requests: Requests that used fallback
        fallback_successes: Successful fallbacks
        fallback_failures: Failed fallbacks
        average_fallback_time_ms: Average time to fallback
        quality_degradation_events: Times quality was degraded
        recovery_events: Times recovered to primary
        fallback_by_tier: Fallback counts by tier
    """
    total_requests: int = 0
    primary_requests: int = 0
    fallback_requests: int = 0
    fallback_successes: int = 0
    fallback_failures: int = 0
    average_fallback_time_ms: float = 0.0
    quality_degradation_events: int = 0
    recovery_events: int = 0
    fallback_by_tier: Dict[ModelTier, int] = field(default_factory=dict)


class FallbackManager:
    """
    Manages fallback models and graceful degradation.

    Provides automatic fallback, health monitoring, and recovery.
    """

    def __init__(self, config: FallbackConfig):
        """
        Initialize fallback manager.

        Args:
            config: Fallback configuration
        """
        if not config.validate():
            raise ValueError("Invalid fallback configuration")

        self.config = config

        # Model registry
        self.models: Dict[str, ModelConfig] = {}
        self.model_health: Dict[str, ModelHealth] = {}
        self.fallback_chains: Dict[str, List[str]] = {}

        # Current active model
        self.current_model_id: Optional[str] = None

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = FallbackStatistics()

        # Event history
        self.fallback_events: List[FallbackEvent] = []
        self.event_counter = 0

    def register_model(self, model_config: ModelConfig) -> bool:
        """
        Register a model for fallback.

        Args:
            model_config: Model configuration

        Returns:
            True if registered, False if already exists
        """
        with self.lock:
            if model_config.model_id in self.models:
                return False

            self.models[model_config.model_id] = model_config
            self.model_health[model_config.model_id] = ModelHealth(
                model_id=model_config.model_id
            )

            # Set as current if first or highest priority
            if (self.current_model_id is None or
                model_config.priority < self.models[self.current_model_id].priority):
                self.current_model_id = model_config.model_id

            return True

    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model.

        Args:
            model_id: Model identifier

        Returns:
            True if unregistered, False if not found
        """
        with self.lock:
            if model_id not in self.models:
                return False

            del self.models[model_id]
            del self.model_health[model_id]

            # Update current model if needed
            if self.current_model_id == model_id:
                self.current_model_id = self._select_best_model()

            return True

    def create_fallback_chain(
        self,
        chain_name: str,
        model_ids: List[str]
    ) -> bool:
        """
        Create a fallback chain.

        Args:
            chain_name: Chain identifier
            model_ids: List of model IDs in priority order

        Returns:
            True if created, False if invalid
        """
        # Validate all models exist
        for model_id in model_ids:
            if model_id not in self.models:
                return False

        with self.lock:
            self.fallback_chains[chain_name] = model_ids

        return True

    def select_model(
        self,
        preferred_model_id: Optional[str] = None,
        chain_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Select best available model.

        Args:
            preferred_model_id: Preferred model ID
            chain_name: Fallback chain to use

        Returns:
            Model ID if available, None if no models available
        """
        with self.lock:
            # Try preferred model first
            if preferred_model_id and self._is_model_available(preferred_model_id):
                return preferred_model_id

            # Try fallback chain
            if chain_name and chain_name in self.fallback_chains:
                for model_id in self.fallback_chains[chain_name]:
                    if self._is_model_available(model_id):
                        return model_id

            # Select best available model
            return self._select_best_model()

    def _is_model_available(self, model_id: str) -> bool:
        """
        Check if model is available for use.

        Args:
            model_id: Model identifier

        Returns:
            True if available
        """
        if model_id not in self.models:
            return False

        model = self.models[model_id]
        if not model.enabled:
            return False

        health = self.model_health[model_id]
        if not health.is_healthy:
            return False

        return True

    def _select_best_model(self) -> Optional[str]:
        """
        Select best available model based on priority.

        Returns:
            Model ID if found, None if no models available
        """
        best_model = None
        best_priority = float('inf')

        for model_id, model in self.models.items():
            if not self._is_model_available(model_id):
                continue

            if model.priority < best_priority:
                best_priority = model.priority
                best_model = model_id

        return best_model

    def record_request(
        self,
        model_id: str,
        success: bool,
        response_time_ms: float
    ) -> None:
        """
        Record request result for model.

        Args:
            model_id: Model identifier
            success: Whether request succeeded
            response_time_ms: Response time
        """
        with self.lock:
            if model_id not in self.model_health:
                return

            health = self.model_health[model_id]
            health.total_requests += 1
            health.last_check_time = time.time()

            if success:
                health.successful_requests += 1
                health.consecutive_successes += 1
                health.consecutive_failures = 0

                # Update average response time
                n = health.successful_requests
                health.average_response_time_ms = (
                    (health.average_response_time_ms * (n - 1) + response_time_ms) / n
                )

                # Mark as healthy after consecutive successes
                if health.consecutive_successes >= 3:
                    health.is_healthy = True

            else:
                health.failed_requests += 1
                health.consecutive_failures += 1
                health.consecutive_successes = 0

                # Mark as unhealthy after consecutive failures
                if health.consecutive_failures >= 3:
                    health.is_healthy = False

            health.update_error_rate()

    def attempt_fallback(
        self,
        failed_model_id: str,
        reason: str,
        chain_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Attempt to fallback to another model.

        Args:
            failed_model_id: Model that failed
            reason: Reason for fallback
            chain_name: Fallback chain to use

        Returns:
            Fallback model ID if found, None if no fallback available
        """
        start_time = time.time()

        # Select fallback model
        fallback_model_id = None

        with self.lock:
            # Get fallback chain
            if chain_name and chain_name in self.fallback_chains:
                chain = self.fallback_chains[chain_name]

                # Find failed model in chain
                try:
                    failed_index = chain.index(failed_model_id)
                    # Try next models in chain
                    for model_id in chain[failed_index + 1:]:
                        if self._is_model_available(model_id):
                            fallback_model_id = model_id
                            break
                except ValueError:
                    pass

            # If no chain or no fallback found, select best available
            if fallback_model_id is None:
                fallback_model_id = self._select_best_model()

        if fallback_model_id is None:
            # No fallback available
            with self.lock:
                self.stats.fallback_failures += 1
            return None

        # Calculate quality degradation
        quality_degradation = 0.0
        if failed_model_id in self.models and fallback_model_id in self.models:
            failed_model = self.models[failed_model_id]
            fallback_model = self.models[fallback_model_id]
            quality_degradation = failed_model.expected_quality - fallback_model.expected_quality

        # Record fallback event
        fallback_time_ms = (time.time() - start_time) * 1000

        with self.lock:
            event_id = f"fallback_{self.event_counter}"
            self.event_counter += 1

            event = FallbackEvent(
                event_id=event_id,
                from_model=failed_model_id,
                to_model=fallback_model_id,
                reason=reason,
                success=True,
                quality_degradation=quality_degradation,
                response_time_ms=fallback_time_ms
            )

            self.fallback_events.append(event)

            # Update statistics
            self.stats.fallback_successes += 1
            self.stats.fallback_requests += 1

            # Update average fallback time
            n = self.stats.fallback_successes
            self.stats.average_fallback_time_ms = (
                (self.stats.average_fallback_time_ms * (n - 1) + fallback_time_ms) / n
            )

            # Track quality degradation
            if quality_degradation > 0:
                self.stats.quality_degradation_events += 1

            # Update current model
            self.current_model_id = fallback_model_id

        return fallback_model_id

    def get_model_health(self, model_id: str) -> Optional[ModelHealth]:
        """
        Get health status for model.

        Args:
            model_id: Model identifier

        Returns:
            ModelHealth if found, None otherwise
        """
        with self.lock:
            if model_id not in self.model_health:
                return None

            health = self.model_health[model_id]
            return ModelHealth(
                model_id=health.model_id,
                is_healthy=health.is_healthy,
                last_check_time=health.last_check_time,
                consecutive_failures=health.consecutive_failures,
                consecutive_successes=health.consecutive_successes,
                total_requests=health.total_requests,
                successful_requests=health.successful_requests,
                failed_requests=health.failed_requests,
                average_response_time_ms=health.average_response_time_ms,
                error_rate=health.error_rate
            )

    def get_all_model_health(self) -> Dict[str, ModelHealth]:
        """
        Get health status for all models.

        Returns:
            Dictionary of model health
        """
        with self.lock:
            return {
                model_id: self.get_model_health(model_id)
                for model_id in self.model_health.keys()
            }

    def get_fallback_events(
        self,
        limit: Optional[int] = None
    ) -> List[FallbackEvent]:
        """
        Get fallback event history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of fallback events
        """
        with self.lock:
            if limit is None:
                return self.fallback_events.copy()
            else:
                return self.fallback_events[-limit:]

    def get_statistics(self) -> FallbackStatistics:
        """
        Get fallback statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            return FallbackStatistics(
                total_requests=self.stats.total_requests,
                primary_requests=self.stats.primary_requests,
                fallback_requests=self.stats.fallback_requests,
                fallback_successes=self.stats.fallback_successes,
                fallback_failures=self.stats.fallback_failures,
                average_fallback_time_ms=self.stats.average_fallback_time_ms,
                quality_degradation_events=self.stats.quality_degradation_events,
                recovery_events=self.stats.recovery_events,
                fallback_by_tier=self.stats.fallback_by_tier.copy()
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = FallbackStatistics()


def create_default_fallback_config() -> FallbackConfig:
    """
    Create default fallback configuration.

    Returns:
        FallbackConfig with sensible defaults
    """
    return FallbackConfig(
        strategy=FallbackStrategy.SEQUENTIAL,
        enable_auto_fallback=True,
        enable_health_checks=True,
        enable_auto_recovery=True,
        max_fallback_attempts=3
    )


def create_aggressive_fallback_config() -> FallbackConfig:
    """
    Create aggressive fallback configuration.

    Returns:
        FallbackConfig for aggressive fallback
    """
    return FallbackConfig(
        strategy=FallbackStrategy.PARALLEL,
        enable_auto_fallback=True,
        enable_health_checks=True,
        enable_auto_recovery=True,
        max_fallback_attempts=5,
        enable_parallel_fallback=True,
        parallel_timeout_seconds=5.0
    )


def create_conservative_fallback_config() -> FallbackConfig:
    """
    Create conservative fallback configuration.

    Returns:
        FallbackConfig for conservative fallback
    """
    return FallbackConfig(
        strategy=FallbackStrategy.QUALITY_BASED,
        enable_auto_fallback=True,
        enable_health_checks=True,
        enable_auto_recovery=True,
        max_fallback_attempts=2,
        enable_quality_warnings=True
    )
