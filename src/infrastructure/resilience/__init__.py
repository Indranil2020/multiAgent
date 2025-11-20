"""
Resilience patterns for infrastructure.

This package provides resilience patterns including circuit breakers,
rate limiting, bulkheads, and timeout management.
"""

from .circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
)

from .rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    MultiKeyRateLimiter,
)

from .bulkhead import (
    BulkheadConfig,
    QueuedRequest,
    Bulkhead,
)

from .timeout_manager import (
    TimeoutConfig,
    TimeoutContext,
    TimeoutManager,
)

__all__ = [
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    # Rate Limiter
    "RateLimitConfig",
    "RateLimiter",
    "MultiKeyRateLimiter",
    # Bulkhead
    "BulkheadConfig",
    "QueuedRequest",
    "Bulkhead",
    # Timeout Manager
    "TimeoutConfig",
    "TimeoutContext",
    "TimeoutManager",
]
