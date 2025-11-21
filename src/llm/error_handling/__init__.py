"""
Error handling module for LLM operations.

This module provides comprehensive error handling, recovery, and fallback mechanisms
for LLM-based code generation:

1. CUDA Error Handling (cuda_handler.py):
   - GPU memory errors (OOM, allocation failures)
   - CUDA device errors and recovery strategies
   - Automatic memory cleanup and retry logic
   - Device statistics and monitoring

2. Fallback Models (fallback_models.py):
   - Multi-tier model fallback strategy
   - Automatic model selection based on health
   - Circuit breaker pattern for failing models
   - Model health monitoring and recovery

3. Retry Logic (retry_handler.py):
   - Exponential backoff with jitter
   - Per-error-type retry strategies
   - Maximum retry limits and timeout handling
   - Retry statistics and monitoring

4. Timeout Handling (timeout_handler.py):
   - Operation-level timeouts
   - Graceful cancellation
   - Timeout actions (retry, fallback, fail)
   - Timeout statistics and alerts

Usage:
    from llm.error_handling import (
        CUDAHandler,
        FallbackManager,
        RetryHandler,
        TimeoutHandler
    )

    # CUDA error handling
    cuda_handler = CUDAHandler()
    with cuda_handler.managed_cuda_context():
        # GPU operations

    # Fallback model management
    fallback_mgr = FallbackManager()
    result = fallback_mgr.execute_with_fallback(operation)

    # Retry with exponential backoff
    retry_handler = RetryHandler()
    result = retry_handler.retry(risky_operation, max_retries=3)

    # Timeout management
    timeout_handler = TimeoutHandler()
    result = timeout_handler.with_timeout(long_operation, timeout=30.0)
"""

from .cuda_handler import (
    CUDAErrorType,
    RecoveryStrategy,
    CUDAHandlerConfig,
    CUDAError,
    RecoveryAttempt,
    CUDAStatistics,
    CUDAHandler,
)

from .fallback_models import (
    FallbackStrategy,
    ModelTier,
    FallbackConfig,
    ModelConfig,
    ModelHealth,
    FallbackEvent,
    FallbackStatistics,
    FallbackManager,
)

from .retry_handler import (
    RetryStrategy,
    ErrorCategory,
    RetryConfig,
    RetryAttempt,
    RetryResult,
    RetryStatistics,
    RetryHandler,
)

from .timeout_handler import (
    TimeoutType,
    TimeoutAction,
    TimeoutConfig,
    TimeoutEvent,
    OperationTracker,
    TimeoutStatistics,
    TimeoutHandler,
)


__all__ = [
    # CUDA handler
    "CUDAErrorType",
    "RecoveryStrategy",
    "CUDAHandlerConfig",
    "CUDAError",
    "RecoveryAttempt",
    "CUDAStatistics",
    "CUDAHandler",

    # Fallback models
    "FallbackStrategy",
    "ModelTier",
    "FallbackConfig",
    "ModelConfig",
    "ModelHealth",
    "FallbackEvent",
    "FallbackStatistics",
    "FallbackManager",

    # Retry handler
    "RetryStrategy",
    "ErrorCategory",
    "RetryConfig",
    "RetryAttempt",
    "RetryResult",
    "RetryStatistics",
    "RetryHandler",

    # Timeout handler
    "TimeoutType",
    "TimeoutAction",
    "TimeoutConfig",
    "TimeoutEvent",
    "OperationTracker",
    "TimeoutStatistics",
    "TimeoutHandler",
]
