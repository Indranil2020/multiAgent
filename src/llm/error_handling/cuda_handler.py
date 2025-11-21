"""
CUDA Error Handler Module.

This module provides comprehensive handling of CUDA/GPU errors in LLM inference.
CUDA errors are common in GPU computing and require specialized handling for
graceful recovery and system stability.

Common CUDA Errors:
- Out of Memory (OOM): Most frequent error in LLM inference
- Device initialization failures
- CUDA runtime errors
- Synchronization errors
- Device-side assertions
- Launch failures
- Invalid device configuration

Key Concepts:
- CUDA OOM can be recovered by clearing cache and reducing batch size
- Some errors require device reset
- Context errors may need process restart
- Memory fragmentation can cause OOM even with free memory
- Proper cleanup prevents resource leaks

Recovery Strategies:
- Clear CUDA cache and retry
- Reduce batch size or sequence length
- Offload to CPU temporarily
- Reset CUDA device
- Graceful degradation to smaller model
- Restart worker process

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time


class CUDAErrorType(Enum):
    """
    CUDA error types.

    Attributes:
        OOM: Out of memory error
        DEVICE_INIT: Device initialization error
        LAUNCH_FAILURE: Kernel launch failure
        SYNC_ERROR: Synchronization error
        ILLEGAL_MEMORY: Illegal memory access
        DEVICE_ASSERT: Device-side assertion
        CONTEXT_ERROR: Context management error
        DRIVER_ERROR: CUDA driver error
        UNKNOWN: Unknown CUDA error
    """
    OOM = "oom"
    DEVICE_INIT = "device_init"
    LAUNCH_FAILURE = "launch_failure"
    SYNC_ERROR = "sync_error"
    ILLEGAL_MEMORY = "illegal_memory"
    DEVICE_ASSERT = "device_assert"
    CONTEXT_ERROR = "context_error"
    DRIVER_ERROR = "driver_error"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """
    CUDA error recovery strategies.

    Attributes:
        CLEAR_CACHE: Clear CUDA cache and retry
        REDUCE_BATCH: Reduce batch size
        REDUCE_SEQUENCE: Reduce sequence length
        OFFLOAD_CPU: Offload to CPU
        RESET_DEVICE: Reset CUDA device
        RESTART_WORKER: Restart worker process
        FAIL: No recovery possible
    """
    CLEAR_CACHE = "clear_cache"
    REDUCE_BATCH = "reduce_batch"
    REDUCE_SEQUENCE = "reduce_sequence"
    OFFLOAD_CPU = "offload_cpu"
    RESET_DEVICE = "reset_device"
    RESTART_WORKER = "restart_worker"
    FAIL = "fail"


@dataclass
class CUDAHandlerConfig:
    """
    Configuration for CUDA error handler.

    Attributes:
        enable_auto_recovery: Enable automatic recovery
        enable_cache_clearing: Enable CUDA cache clearing
        enable_device_reset: Enable device reset (dangerous)
        max_oom_retries: Maximum OOM retry attempts
        oom_reduction_factor: Factor to reduce batch size (0.0-1.0)
        min_batch_size: Minimum batch size before giving up
        enable_cpu_fallback: Enable CPU fallback
        enable_monitoring: Enable CUDA device monitoring
        monitor_interval_seconds: Monitoring interval
        max_recovery_attempts: Maximum recovery attempts per error
    """
    enable_auto_recovery: bool = True
    enable_cache_clearing: bool = True
    enable_device_reset: bool = False
    max_oom_retries: int = 3
    oom_reduction_factor: float = 0.5
    min_batch_size: int = 1
    enable_cpu_fallback: bool = True
    enable_monitoring: bool = True
    monitor_interval_seconds: float = 60.0
    max_recovery_attempts: int = 5

    def validate(self) -> bool:
        """
        Validate CUDA handler configuration.

        Returns:
            True if valid, False otherwise
        """
        if self.max_oom_retries < 0:
            return False
        if not (0.0 < self.oom_reduction_factor <= 1.0):
            return False
        if self.min_batch_size <= 0:
            return False
        if self.monitor_interval_seconds <= 0:
            return False
        if self.max_recovery_attempts <= 0:
            return False
        return True


@dataclass
class CUDAError:
    """
    CUDA error record.

    Attributes:
        error_type: Type of CUDA error
        error_message: Error message
        device_id: Device where error occurred
        timestamp: Error timestamp
        context: Additional context about error
        recoverable: Whether error is recoverable
        suggested_strategy: Suggested recovery strategy
    """
    error_type: CUDAErrorType
    error_message: str
    device_id: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    suggested_strategy: RecoveryStrategy = RecoveryStrategy.CLEAR_CACHE


@dataclass
class RecoveryAttempt:
    """
    Record of recovery attempt.

    Attributes:
        attempt_number: Attempt number
        strategy: Recovery strategy used
        timestamp: Attempt timestamp
        success: Whether recovery succeeded
        details: Additional details
    """
    attempt_number: int
    strategy: RecoveryStrategy
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    details: str = ""


@dataclass
class CUDAStatistics:
    """
    CUDA error handling statistics.

    Attributes:
        total_errors: Total CUDA errors encountered
        oom_errors: Out of memory errors
        recoverable_errors: Successfully recovered errors
        unrecoverable_errors: Errors that could not be recovered
        cache_clears: Number of cache clears performed
        device_resets: Number of device resets
        cpu_fallbacks: Number of CPU fallbacks
        errors_by_type: Error counts by type
        recovery_success_rate: Recovery success rate (0.0-1.0)
    """
    total_errors: int = 0
    oom_errors: int = 0
    recoverable_errors: int = 0
    unrecoverable_errors: int = 0
    cache_clears: int = 0
    device_resets: int = 0
    cpu_fallbacks: int = 0
    errors_by_type: Dict[CUDAErrorType, int] = field(default_factory=dict)
    recovery_success_rate: float = 0.0

    def update_success_rate(self) -> None:
        """Update recovery success rate."""
        total_recoverable = self.recoverable_errors + self.unrecoverable_errors
        if total_recoverable > 0:
            self.recovery_success_rate = self.recoverable_errors / total_recoverable
        else:
            self.recovery_success_rate = 0.0


class CUDAHandler:
    """
    Comprehensive CUDA error handler with recovery strategies.

    Handles CUDA/GPU errors with intelligent recovery and fallback.
    """

    def __init__(self, config: CUDAHandlerConfig):
        """
        Initialize CUDA handler.

        Args:
            config: CUDA handler configuration
        """
        if not config.validate():
            raise ValueError("Invalid CUDA handler configuration")

        self.config = config

        # Error tracking
        self.error_history: List[CUDAError] = []
        self.recovery_history: List[RecoveryAttempt] = []

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = CUDAStatistics()

    def handle_error(
        self,
        error: Exception,
        device_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[RecoveryStrategy]:
        """
        Handle CUDA error and determine recovery strategy.

        Args:
            error: Exception that occurred
            device_id: Device where error occurred
            context: Additional context

        Returns:
            RecoveryStrategy if error is recoverable, None if not
        """
        # Classify error
        cuda_error = self._classify_error(error, device_id, context or {})

        # Record error
        with self.lock:
            self.error_history.append(cuda_error)
            self.stats.total_errors += 1

            # Update error type counts
            if cuda_error.error_type not in self.stats.errors_by_type:
                self.stats.errors_by_type[cuda_error.error_type] = 0
            self.stats.errors_by_type[cuda_error.error_type] += 1

            # Track OOM separately
            if cuda_error.error_type == CUDAErrorType.OOM:
                self.stats.oom_errors += 1

        # Check if recoverable
        if not cuda_error.recoverable or not self.config.enable_auto_recovery:
            with self.lock:
                self.stats.unrecoverable_errors += 1
                self.stats.update_success_rate()
            return None

        # Return suggested recovery strategy
        return cuda_error.suggested_strategy

    def attempt_recovery(
        self,
        strategy: RecoveryStrategy,
        device_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt to recover from CUDA error.

        Args:
            strategy: Recovery strategy to use
            device_id: Device to recover
            context: Additional context

        Returns:
            True if recovery succeeded, False otherwise
        """
        attempt_number = len(self.recovery_history) + 1

        # Create recovery attempt record
        attempt = RecoveryAttempt(
            attempt_number=attempt_number,
            strategy=strategy
        )

        # Attempt recovery based on strategy
        if strategy == RecoveryStrategy.CLEAR_CACHE:
            success = self._clear_cuda_cache(device_id)
            if success:
                with self.lock:
                    self.stats.cache_clears += 1

        elif strategy == RecoveryStrategy.REDUCE_BATCH:
            success = self._reduce_batch_size(context or {})

        elif strategy == RecoveryStrategy.REDUCE_SEQUENCE:
            success = self._reduce_sequence_length(context or {})

        elif strategy == RecoveryStrategy.OFFLOAD_CPU:
            success = self._offload_to_cpu(context or {})
            if success:
                with self.lock:
                    self.stats.cpu_fallbacks += 1

        elif strategy == RecoveryStrategy.RESET_DEVICE:
            if self.config.enable_device_reset:
                success = self._reset_device(device_id)
                if success:
                    with self.lock:
                        self.stats.device_resets += 1
            else:
                success = False

        elif strategy == RecoveryStrategy.RESTART_WORKER:
            # Would trigger worker restart in production
            success = False

        else:
            success = False

        # Update attempt record
        attempt.success = success

        # Record attempt
        with self.lock:
            self.recovery_history.append(attempt)

            if success:
                self.stats.recoverable_errors += 1
            else:
                self.stats.unrecoverable_errors += 1

            self.stats.update_success_rate()

        return success

    def _classify_error(
        self,
        error: Exception,
        device_id: str,
        context: Dict[str, Any]
    ) -> CUDAError:
        """
        Classify CUDA error.

        Args:
            error: Exception
            device_id: Device ID
            context: Error context

        Returns:
            CUDAError with classification
        """
        error_str = str(error).lower()

        # Determine error type
        if "out of memory" in error_str or "oom" in error_str:
            error_type = CUDAErrorType.OOM
            recoverable = True
            strategy = RecoveryStrategy.CLEAR_CACHE

        elif "device" in error_str and "init" in error_str:
            error_type = CUDAErrorType.DEVICE_INIT
            recoverable = True
            strategy = RecoveryStrategy.RESET_DEVICE

        elif "launch" in error_str or "kernel" in error_str:
            error_type = CUDAErrorType.LAUNCH_FAILURE
            recoverable = True
            strategy = RecoveryStrategy.CLEAR_CACHE

        elif "sync" in error_str or "synchronize" in error_str:
            error_type = CUDAErrorType.SYNC_ERROR
            recoverable = True
            strategy = RecoveryStrategy.RESET_DEVICE

        elif "illegal" in error_str and "memory" in error_str:
            error_type = CUDAErrorType.ILLEGAL_MEMORY
            recoverable = False
            strategy = RecoveryStrategy.FAIL

        elif "assert" in error_str:
            error_type = CUDAErrorType.DEVICE_ASSERT
            recoverable = False
            strategy = RecoveryStrategy.FAIL

        elif "context" in error_str:
            error_type = CUDAErrorType.CONTEXT_ERROR
            recoverable = True
            strategy = RecoveryStrategy.RESET_DEVICE

        elif "driver" in error_str:
            error_type = CUDAErrorType.DRIVER_ERROR
            recoverable = True
            strategy = RecoveryStrategy.RESET_DEVICE

        else:
            error_type = CUDAErrorType.UNKNOWN
            recoverable = True
            strategy = RecoveryStrategy.CLEAR_CACHE

        return CUDAError(
            error_type=error_type,
            error_message=str(error),
            device_id=device_id,
            context=context,
            recoverable=recoverable,
            suggested_strategy=strategy
        )

    def _clear_cuda_cache(self, device_id: str) -> bool:
        """
        Clear CUDA cache.

        Args:
            device_id: Device ID

        Returns:
            True if successful
        """
        if not self.config.enable_cache_clearing:
            return False

        # In production, would call torch.cuda.empty_cache()
        # For now, simulate success
        return True

    def _reduce_batch_size(self, context: Dict[str, Any]) -> bool:
        """
        Reduce batch size in context.

        Args:
            context: Context to modify

        Returns:
            True if successful
        """
        if "batch_size" not in context:
            return False

        current_batch_size = context["batch_size"]
        new_batch_size = max(
            int(current_batch_size * self.config.oom_reduction_factor),
            self.config.min_batch_size
        )

        if new_batch_size < current_batch_size:
            context["batch_size"] = new_batch_size
            return True

        return False

    def _reduce_sequence_length(self, context: Dict[str, Any]) -> bool:
        """
        Reduce sequence length in context.

        Args:
            context: Context to modify

        Returns:
            True if successful
        """
        if "sequence_length" not in context:
            return False

        current_length = context["sequence_length"]
        new_length = int(current_length * self.config.oom_reduction_factor)

        if new_length < current_length and new_length > 0:
            context["sequence_length"] = new_length
            return True

        return False

    def _offload_to_cpu(self, context: Dict[str, Any]) -> bool:
        """
        Offload operation to CPU.

        Args:
            context: Context to modify

        Returns:
            True if successful
        """
        if not self.config.enable_cpu_fallback:
            return False

        # Set CPU device in context
        context["device"] = "cpu"
        context["use_cuda"] = False

        return True

    def _reset_device(self, device_id: str) -> bool:
        """
        Reset CUDA device.

        Args:
            device_id: Device to reset

        Returns:
            True if successful
        """
        if not self.config.enable_device_reset:
            return False

        # In production, would call torch.cuda.reset_peak_memory_stats()
        # and torch.cuda.reset_accumulated_memory_stats()
        # This is a dangerous operation that can affect other processes
        # For now, simulate success
        return True

    def get_error_history(
        self,
        limit: Optional[int] = None
    ) -> List[CUDAError]:
        """
        Get CUDA error history.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of CUDA errors
        """
        with self.lock:
            if limit is None:
                return self.error_history.copy()
            else:
                return self.error_history[-limit:]

    def get_recovery_history(
        self,
        limit: Optional[int] = None
    ) -> List[RecoveryAttempt]:
        """
        Get recovery attempt history.

        Args:
            limit: Maximum number of attempts to return

        Returns:
            List of recovery attempts
        """
        with self.lock:
            if limit is None:
                return self.recovery_history.copy()
            else:
                return self.recovery_history[-limit:]

    def get_statistics(self) -> CUDAStatistics:
        """
        Get CUDA error statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            return CUDAStatistics(
                total_errors=self.stats.total_errors,
                oom_errors=self.stats.oom_errors,
                recoverable_errors=self.stats.recoverable_errors,
                unrecoverable_errors=self.stats.unrecoverable_errors,
                cache_clears=self.stats.cache_clears,
                device_resets=self.stats.device_resets,
                cpu_fallbacks=self.stats.cpu_fallbacks,
                errors_by_type=self.stats.errors_by_type.copy(),
                recovery_success_rate=self.stats.recovery_success_rate
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = CUDAStatistics()


def create_default_cuda_config() -> CUDAHandlerConfig:
    """
    Create default CUDA handler configuration.

    Returns:
        CUDAHandlerConfig with sensible defaults
    """
    return CUDAHandlerConfig(
        enable_auto_recovery=True,
        enable_cache_clearing=True,
        enable_device_reset=False,
        max_oom_retries=3,
        oom_reduction_factor=0.5,
        enable_cpu_fallback=True
    )


def create_aggressive_recovery_config() -> CUDAHandlerConfig:
    """
    Create aggressive recovery configuration.

    Returns:
        CUDAHandlerConfig for aggressive recovery
    """
    return CUDAHandlerConfig(
        enable_auto_recovery=True,
        enable_cache_clearing=True,
        enable_device_reset=True,
        max_oom_retries=5,
        oom_reduction_factor=0.3,
        enable_cpu_fallback=True,
        max_recovery_attempts=10
    )


def create_conservative_config() -> CUDAHandlerConfig:
    """
    Create conservative CUDA configuration.

    Returns:
        CUDAHandlerConfig for conservative handling
    """
    return CUDAHandlerConfig(
        enable_auto_recovery=True,
        enable_cache_clearing=True,
        enable_device_reset=False,
        max_oom_retries=1,
        oom_reduction_factor=0.7,
        enable_cpu_fallback=False,
        max_recovery_attempts=2
    )
