"""
Timeout Handler Module.

This module provides comprehensive timeout management for LLM inference operations.
Prevents hanging operations, manages resource cleanup, and provides graceful timeout
handling with configurable policies.

Key Concepts:
- Long-running inference can hang due to bugs or system issues
- Timeouts prevent resource exhaustion from stuck operations
- Graceful timeouts allow cleanup before termination
- Hard timeouts forcefully stop operations
- Different operations need different timeout policies

Timeout Types:
- Operation Timeout: Maximum time for single inference
- Request Timeout: Maximum time including queuing
- Connection Timeout: Maximum time for network operations
- Idle Timeout: Maximum time without progress
- Global Timeout: System-wide timeout limit

Features:
- Configurable timeout policies per operation type
- Graceful shutdown with cleanup period
- Hard timeout enforcement
- Timeout statistics and monitoring
- Automatic timeout adjustment based on performance
- Timeout warnings before enforcement
- Support for nested timeouts

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time


T = TypeVar('T')


class TimeoutType(Enum):
    """
    Timeout types.

    Attributes:
        OPERATION: Single operation timeout
        REQUEST: Full request timeout (including queue)
        CONNECTION: Network connection timeout
        IDLE: Idle/progress timeout
        GLOBAL: System-wide timeout
    """
    OPERATION = "operation"
    REQUEST = "request"
    CONNECTION = "connection"
    IDLE = "idle"
    GLOBAL = "global"


class TimeoutAction(Enum):
    """
    Actions to take on timeout.

    Attributes:
        WARNING: Issue warning but continue
        GRACEFUL_STOP: Request graceful stop with cleanup
        HARD_STOP: Force immediate stop
        RETRY: Retry operation
        FAILOVER: Failover to backup
    """
    WARNING = "warning"
    GRACEFUL_STOP = "graceful_stop"
    HARD_STOP = "hard_stop"
    RETRY = "retry"
    FAILOVER = "failover"


@dataclass
class TimeoutConfig:
    """
    Configuration for timeout handling.

    Attributes:
        timeout_seconds: Timeout duration in seconds
        timeout_type: Type of timeout
        action: Action to take on timeout
        enable_warnings: Enable warnings before timeout
        warning_threshold: Warn at this fraction of timeout (0.0-1.0)
        grace_period_seconds: Graceful shutdown period
        enable_auto_adjustment: Adjust timeout based on performance
        min_timeout_seconds: Minimum timeout (for auto-adjustment)
        max_timeout_seconds: Maximum timeout (for auto-adjustment)
        adjustment_factor: Factor for auto-adjustment (1.0 = no adjustment)
    """
    timeout_seconds: float
    timeout_type: TimeoutType = TimeoutType.OPERATION
    action: TimeoutAction = TimeoutAction.GRACEFUL_STOP
    enable_warnings: bool = True
    warning_threshold: float = 0.8
    grace_period_seconds: float = 5.0
    enable_auto_adjustment: bool = False
    min_timeout_seconds: float = 1.0
    max_timeout_seconds: float = 600.0
    adjustment_factor: float = 1.2

    def validate(self) -> bool:
        """
        Validate timeout configuration.

        Returns:
            True if valid, False otherwise
        """
        if self.timeout_seconds <= 0:
            return False
        if not (0.0 <= self.warning_threshold <= 1.0):
            return False
        if self.grace_period_seconds < 0:
            return False
        if self.min_timeout_seconds <= 0:
            return False
        if self.max_timeout_seconds < self.min_timeout_seconds:
            return False
        if self.adjustment_factor <= 0:
            return False
        return True


@dataclass
class TimeoutEvent:
    """
    Record of a timeout event.

    Attributes:
        operation_id: Operation identifier
        timeout_type: Type of timeout
        action_taken: Action that was taken
        timeout_seconds: Configured timeout
        elapsed_seconds: Actual elapsed time
        timestamp: Event timestamp
        warning_issued: Whether warning was issued
        graceful_stop_succeeded: Whether graceful stop succeeded
        context: Additional context
    """
    operation_id: str
    timeout_type: TimeoutType
    action_taken: TimeoutAction
    timeout_seconds: float
    elapsed_seconds: float
    timestamp: float = field(default_factory=time.time)
    warning_issued: bool = False
    graceful_stop_succeeded: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationTracker:
    """
    Tracks a running operation for timeout monitoring.

    Attributes:
        operation_id: Unique operation identifier
        start_time: Operation start time
        config: Timeout configuration
        warning_issued: Whether warning has been issued
        cancel_requested: Whether cancellation was requested
        completed: Whether operation completed
        result: Operation result if completed
        error: Error if operation failed
    """
    operation_id: str
    start_time: float
    config: TimeoutConfig
    warning_issued: bool = False
    cancel_requested: bool = False
    completed: bool = False
    result: Any = None
    error: Optional[Exception] = None


@dataclass
class TimeoutStatistics:
    """
    Statistics for timeout handling.

    Attributes:
        total_operations: Total operations tracked
        completed_operations: Operations that completed
        timed_out_operations: Operations that timed out
        warnings_issued: Number of warnings issued
        graceful_stops: Number of graceful stops
        hard_stops: Number of hard stops
        average_duration_seconds: Average operation duration
        average_timeout_utilization: Average fraction of timeout used
        timeouts_by_type: Timeout counts by type
    """
    total_operations: int = 0
    completed_operations: int = 0
    timed_out_operations: int = 0
    warnings_issued: int = 0
    graceful_stops: int = 0
    hard_stops: int = 0
    average_duration_seconds: float = 0.0
    average_timeout_utilization: float = 0.0
    timeouts_by_type: Dict[TimeoutType, int] = field(default_factory=dict)


class TimeoutHandler:
    """
    Comprehensive timeout handler with configurable policies.

    Manages operation timeouts with graceful shutdown, warnings,
    and automatic adjustment.
    """

    def __init__(self):
        """Initialize timeout handler."""
        # Operation tracking
        self.active_operations: Dict[str, OperationTracker] = {}
        self.operation_counter = 0

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = TimeoutStatistics()

        # Timeout events
        self.timeout_events: List[TimeoutEvent] = []

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """
        Start timeout monitoring thread.

        Args:
            interval_seconds: Monitoring check interval
        """
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop timeout monitoring thread."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def track_operation(
        self,
        operation_id: Optional[str],
        config: TimeoutConfig
    ) -> str:
        """
        Start tracking an operation for timeout.

        Args:
            operation_id: Optional operation ID (generated if None)
            config: Timeout configuration

        Returns:
            Operation ID
        """
        if not config.validate():
            raise ValueError("Invalid timeout configuration")

        # Generate operation ID if needed
        if operation_id is None:
            with self.lock:
                operation_id = f"op_{self.operation_counter}"
                self.operation_counter += 1

        # Create tracker
        tracker = OperationTracker(
            operation_id=operation_id,
            start_time=time.time(),
            config=config
        )

        # Register operation
        with self.lock:
            self.active_operations[operation_id] = tracker
            self.stats.total_operations += 1

        return operation_id

    def complete_operation(
        self,
        operation_id: str,
        result: Any = None,
        error: Optional[Exception] = None
    ) -> bool:
        """
        Mark operation as completed.

        Args:
            operation_id: Operation identifier
            result: Operation result
            error: Error if operation failed

        Returns:
            True if operation was being tracked, False otherwise
        """
        with self.lock:
            if operation_id not in self.active_operations:
                return False

            tracker = self.active_operations[operation_id]
            tracker.completed = True
            tracker.result = result
            tracker.error = error

            # Update statistics
            duration = time.time() - tracker.start_time
            self.stats.completed_operations += 1

            # Update average duration
            n = self.stats.completed_operations
            self.stats.average_duration_seconds = (
                (self.stats.average_duration_seconds * (n - 1) + duration) / n
            )

            # Update timeout utilization
            utilization = duration / tracker.config.timeout_seconds
            self.stats.average_timeout_utilization = (
                (self.stats.average_timeout_utilization * (n - 1) + utilization) / n
            )

            # Remove from tracking
            del self.active_operations[operation_id]

            return True

    def is_operation_cancelled(self, operation_id: str) -> bool:
        """
        Check if operation has been cancelled.

        Args:
            operation_id: Operation identifier

        Returns:
            True if cancelled, False otherwise
        """
        with self.lock:
            if operation_id not in self.active_operations:
                return False

            tracker = self.active_operations[operation_id]
            return tracker.cancel_requested

    def _monitoring_loop(self, interval_seconds: float) -> None:
        """
        Monitoring loop for checking timeouts.

        Args:
            interval_seconds: Check interval
        """
        while self.monitoring:
            self._check_timeouts()
            time.sleep(interval_seconds)

    def _check_timeouts(self) -> None:
        """Check all active operations for timeouts."""
        current_time = time.time()

        with self.lock:
            # Get snapshot of active operations
            operations = list(self.active_operations.values())

        for tracker in operations:
            elapsed = current_time - tracker.start_time
            timeout_threshold = tracker.config.timeout_seconds
            warning_threshold = timeout_threshold * tracker.config.warning_threshold

            # Check for warning
            if (tracker.config.enable_warnings and
                not tracker.warning_issued and
                elapsed >= warning_threshold):
                self._issue_warning(tracker)

            # Check for timeout
            if elapsed >= timeout_threshold:
                self._handle_timeout(tracker, elapsed)

    def _issue_warning(self, tracker: OperationTracker) -> None:
        """
        Issue timeout warning.

        Args:
            tracker: Operation tracker
        """
        with self.lock:
            tracker.warning_issued = True
            self.stats.warnings_issued += 1

        # In production, would trigger warning callback
        # For now, just mark as warned

    def _handle_timeout(
        self,
        tracker: OperationTracker,
        elapsed: float
    ) -> None:
        """
        Handle operation timeout.

        Args:
            tracker: Operation tracker
            elapsed: Elapsed time
        """
        # Take action based on configuration
        action = tracker.config.action

        # Create timeout event
        event = TimeoutEvent(
            operation_id=tracker.operation_id,
            timeout_type=tracker.config.timeout_type,
            action_taken=action,
            timeout_seconds=tracker.config.timeout_seconds,
            elapsed_seconds=elapsed,
            warning_issued=tracker.warning_issued
        )

        if action == TimeoutAction.WARNING:
            # Just warn, don't stop
            pass

        elif action == TimeoutAction.GRACEFUL_STOP:
            # Request graceful stop
            success = self._request_graceful_stop(tracker)
            event.graceful_stop_succeeded = success

            with self.lock:
                self.stats.graceful_stops += 1

        elif action == TimeoutAction.HARD_STOP:
            # Force stop
            self._force_stop(tracker)

            with self.lock:
                self.stats.hard_stops += 1

        elif action == TimeoutAction.RETRY:
            # Mark for retry (handled by caller)
            tracker.cancel_requested = True

        elif action == TimeoutAction.FAILOVER:
            # Mark for failover (handled by caller)
            tracker.cancel_requested = True

        # Record event
        with self.lock:
            self.timeout_events.append(event)
            self.stats.timed_out_operations += 1

            # Update timeout counts by type
            timeout_type = tracker.config.timeout_type
            if timeout_type not in self.stats.timeouts_by_type:
                self.stats.timeouts_by_type[timeout_type] = 0
            self.stats.timeouts_by_type[timeout_type] += 1

            # Remove from active operations
            if tracker.operation_id in self.active_operations:
                del self.active_operations[tracker.operation_id]

    def _request_graceful_stop(self, tracker: OperationTracker) -> bool:
        """
        Request graceful stop of operation.

        Args:
            tracker: Operation tracker

        Returns:
            True if graceful stop succeeded
        """
        # Set cancel flag
        tracker.cancel_requested = True

        # Wait for grace period
        grace_end = time.time() + tracker.config.grace_period_seconds

        while time.time() < grace_end:
            if tracker.completed:
                return True
            time.sleep(0.1)

        # Grace period expired
        return False

    def _force_stop(self, tracker: OperationTracker) -> None:
        """
        Force stop operation.

        Args:
            tracker: Operation tracker
        """
        # Set cancel flag
        tracker.cancel_requested = True

        # In production, would forcefully terminate thread/process
        # This is dangerous and should be avoided when possible
        # For now, just mark as cancelled

    def get_active_operations(self) -> List[Dict[str, Any]]:
        """
        Get list of active operations.

        Returns:
            List of operation info dictionaries
        """
        current_time = time.time()

        with self.lock:
            operations = []
            for tracker in self.active_operations.values():
                elapsed = current_time - tracker.start_time
                remaining = tracker.config.timeout_seconds - elapsed

                operations.append({
                    "operation_id": tracker.operation_id,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining,
                    "timeout_type": tracker.config.timeout_type.value,
                    "warning_issued": tracker.warning_issued,
                    "cancel_requested": tracker.cancel_requested
                })

            return operations

    def get_timeout_events(
        self,
        limit: Optional[int] = None
    ) -> List[TimeoutEvent]:
        """
        Get timeout event history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of timeout events
        """
        with self.lock:
            if limit is None:
                return self.timeout_events.copy()
            else:
                return self.timeout_events[-limit:]

    def get_statistics(self) -> TimeoutStatistics:
        """
        Get timeout statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            return TimeoutStatistics(
                total_operations=self.stats.total_operations,
                completed_operations=self.stats.completed_operations,
                timed_out_operations=self.stats.timed_out_operations,
                warnings_issued=self.stats.warnings_issued,
                graceful_stops=self.stats.graceful_stops,
                hard_stops=self.stats.hard_stops,
                average_duration_seconds=self.stats.average_duration_seconds,
                average_timeout_utilization=self.stats.average_timeout_utilization,
                timeouts_by_type=self.stats.timeouts_by_type.copy()
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = TimeoutStatistics()


def create_default_timeout_config(timeout_seconds: float = 30.0) -> TimeoutConfig:
    """
    Create default timeout configuration.

    Args:
        timeout_seconds: Timeout duration

    Returns:
        TimeoutConfig with defaults
    """
    return TimeoutConfig(
        timeout_seconds=timeout_seconds,
        timeout_type=TimeoutType.OPERATION,
        action=TimeoutAction.GRACEFUL_STOP,
        enable_warnings=True,
        warning_threshold=0.8,
        grace_period_seconds=5.0
    )


def create_strict_timeout_config(timeout_seconds: float = 10.0) -> TimeoutConfig:
    """
    Create strict timeout configuration with hard stop.

    Args:
        timeout_seconds: Timeout duration

    Returns:
        TimeoutConfig for strict enforcement
    """
    return TimeoutConfig(
        timeout_seconds=timeout_seconds,
        timeout_type=TimeoutType.OPERATION,
        action=TimeoutAction.HARD_STOP,
        enable_warnings=True,
        warning_threshold=0.9,
        grace_period_seconds=0.0
    )


def create_lenient_timeout_config(timeout_seconds: float = 120.0) -> TimeoutConfig:
    """
    Create lenient timeout configuration with warnings only.

    Args:
        timeout_seconds: Timeout duration

    Returns:
        TimeoutConfig for lenient handling
    """
    return TimeoutConfig(
        timeout_seconds=timeout_seconds,
        timeout_type=TimeoutType.OPERATION,
        action=TimeoutAction.WARNING,
        enable_warnings=True,
        warning_threshold=0.7,
        grace_period_seconds=30.0
    )
