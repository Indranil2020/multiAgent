"""
Timeout management implementation.

This module provides comprehensive timeout tracking and management for operations,
including deadline calculation, remaining time tracking, and automatic expiration detection.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any, Callable
from enum import Enum
import time
import threading


class TimeoutStatus(Enum):
    """Status of a timeout context."""
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class TimeoutSeverity(Enum):
    """Severity levels for timeout violations."""
    WARNING = "warning"  # Approaching timeout
    CRITICAL = "critical"  # Timeout exceeded
    FATAL = "fatal"  # Severe timeout violation


@dataclass
class TimeoutConfig:
    """
    Configuration for timeout manager.
    
    Attributes:
        default_timeout_seconds: Default timeout duration
        warning_threshold_percent: Percentage of timeout to trigger warning
        enable_auto_cleanup: Enable automatic cleanup of expired contexts
        cleanup_interval_seconds: Interval for cleanup
        max_contexts: Maximum number of active contexts
    """
    default_timeout_seconds: float
    warning_threshold_percent: float = 80.0
    enable_auto_cleanup: bool = True
    cleanup_interval_seconds: float = 60.0
    max_contexts: int = 10000
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.default_timeout_seconds <= 0:
            return (False, "default_timeout_seconds must be positive")
        
        if not 0 < self.warning_threshold_percent < 100:
            return (False, "warning_threshold_percent must be between 0 and 100")
        
        if self.cleanup_interval_seconds <= 0:
            return (False, "cleanup_interval_seconds must be positive")
        
        if self.max_contexts < 1:
            return (False, "max_contexts must be at least 1")
        
        return (True, "")


@dataclass
class TimeoutContext:
    """
    A timeout context for tracking an operation.
    
    Attributes:
        context_id: Unique context identifier
        operation_name: Name of the operation
        start_time: When operation started
        deadline: When operation times out
        timeout_seconds: Timeout duration
        status: Current status
        metadata: Additional metadata
        warning_triggered: Whether warning has been triggered
        completion_time: When operation completed
    """
    context_id: str
    operation_name: str
    start_time: float
    deadline: float
    timeout_seconds: float
    status: TimeoutStatus = TimeoutStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    warning_triggered: bool = False
    completion_time: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if context has expired."""
        return time.time() >= self.deadline
    
    def remaining_seconds(self) -> float:
        """Get remaining time in seconds."""
        return max(0.0, self.deadline - time.time())
    
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def progress_percent(self) -> float:
        """Get progress percentage (0-100)."""
        elapsed = self.elapsed_seconds()
        return min(100.0, (elapsed / self.timeout_seconds) * 100.0)
    
    def duration_seconds(self) -> float:
        """Get total duration (for completed operations)."""
        if self.completion_time > 0:
            return self.completion_time - self.start_time
        return self.elapsed_seconds()


@dataclass
class TimeoutMetrics:
    """
    Comprehensive metrics for timeout manager.
    
    Attributes:
        total_contexts: Total contexts created
        active_contexts: Currently active contexts
        completed_contexts: Contexts completed successfully
        expired_contexts: Contexts that expired
        cancelled_contexts: Contexts that were cancelled
        warnings_triggered: Number of warnings triggered
        average_duration: Average operation duration
        timeout_rate: Percentage of operations that timed out
    """
    total_contexts: int = 0
    active_contexts: int = 0
    completed_contexts: int = 0
    expired_contexts: int = 0
    cancelled_contexts: int = 0
    warnings_triggered: int = 0
    average_duration: float = 0.0
    timeout_rate: float = 0.0


class TimeoutManager:
    """
    Timeout manager for tracking operation deadlines.
    
    Manages timeout contexts, tracks deadlines, detects expirations,
    and provides warnings for operations approaching timeout.
    """
    
    def __init__(
        self,
        config: TimeoutConfig,
        on_timeout: Optional[Callable[[TimeoutContext], None]] = None,
        on_warning: Optional[Callable[[TimeoutContext], None]] = None
    ):
        """
        Initialize timeout manager.
        
        Args:
            config: Timeout configuration
            on_timeout: Optional callback for timeouts
            on_warning: Optional callback for warnings
        """
        self.config = config
        self.contexts: Dict[str, TimeoutContext] = {}
        self.on_timeout = on_timeout
        self.on_warning = on_warning
        
        # Metrics
        self.metrics = TimeoutMetrics()
        
        # Duration tracking
        self.durations: List[float] = []
        
        # Context counter
        self.context_counter = 0
        
        # Cleanup tracking
        self.last_cleanup = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate timeout manager configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def start_timeout(
        self,
        operation_name: str,
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Start a timeout context for an operation.
        
        Args:
            operation_name: Name of the operation
            timeout_seconds: Timeout duration (uses default if None)
            metadata: Optional metadata
        
        Returns:
            Tuple of (success, context_id or None, message)
        """
        with self.lock:
            if not operation_name:
                return (False, None, "operation_name cannot be empty")
            
            # Check context limit
            if len(self.contexts) >= self.config.max_contexts:
                return (False, None, f"Maximum contexts ({self.config.max_contexts}) reached")
            
            # Use default timeout if not specified
            if timeout_seconds is None:
                timeout_seconds = self.config.default_timeout_seconds
            
            if timeout_seconds <= 0:
                return (False, None, "timeout_seconds must be positive")
            
            # Generate context ID
            context_id = self._generate_context_id(operation_name)
            
            # Create context
            current_time = time.time()
            context = TimeoutContext(
                context_id=context_id,
                operation_name=operation_name,
                start_time=current_time,
                deadline=current_time + timeout_seconds,
                timeout_seconds=timeout_seconds,
                metadata=metadata or {}
            )
            
            # Store context
            self.contexts[context_id] = context
            
            # Update metrics
            self.metrics.total_contexts += 1
            self.metrics.active_contexts = len([c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE])
            
            # Try cleanup if enabled
            if self.config.enable_auto_cleanup:
                self._try_cleanup()
            
            return (True, context_id, f"Timeout context started (deadline: {timeout_seconds}s)")
    
    def check_timeout(self, context_id: str) -> Tuple[bool, bool, str]:
        """
        Check if a context has timed out.
        
        Args:
            context_id: Context identifier
        
        Returns:
            Tuple of (success, is_expired, message)
        """
        with self.lock:
            if not context_id:
                return (False, False, "context_id cannot be empty")
            
            if context_id not in self.contexts:
                return (False, False, f"Context {context_id} not found")
            
            context = self.contexts[context_id]
            
            if context.status != TimeoutStatus.ACTIVE:
                return (True, context.status == TimeoutStatus.EXPIRED, f"Context status: {context.status.value}")
            
            # Check if expired
            if context.is_expired():
                # Mark as expired
                context.status = TimeoutStatus.EXPIRED
                context.completion_time = time.time()
                
                # Update metrics
                self.metrics.expired_contexts += 1
                self.metrics.active_contexts = len([c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE])
                self._update_timeout_rate()
                
                # Trigger callback
                if self.on_timeout:
                    self.on_timeout(context)
                
                return (True, True, f"Context expired (deadline: {context.deadline})")
            
            # Check for warning
            if not context.warning_triggered:
                progress = context.progress_percent()
                
                if progress >= self.config.warning_threshold_percent:
                    context.warning_triggered = True
                    self.metrics.warnings_triggered += 1
                    
                    # Trigger callback
                    if self.on_warning:
                        self.on_warning(context)
                    
                    return (True, False, f"Warning: {progress:.1f}% of timeout elapsed")
            
            return (True, False, f"Active ({context.remaining_seconds():.2f}s remaining)")
    
    def complete_timeout(self, context_id: str) -> Tuple[bool, str]:
        """
        Mark a timeout context as completed.
        
        Args:
            context_id: Context identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not context_id:
                return (False, "context_id cannot be empty")
            
            if context_id not in self.contexts:
                return (False, f"Context {context_id} not found")
            
            context = self.contexts[context_id]
            
            if context.status != TimeoutStatus.ACTIVE:
                return (False, f"Context is not active (status: {context.status.value})")
            
            # Mark as completed
            context.status = TimeoutStatus.COMPLETED
            context.completion_time = time.time()
            
            # Track duration
            duration = context.duration_seconds()
            self.durations.append(duration)
            self._update_average_duration()
            
            # Update metrics
            self.metrics.completed_contexts += 1
            self.metrics.active_contexts = len([c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE])
            self._update_timeout_rate()
            
            return (True, f"Context completed (duration: {duration:.2f}s)")
    
    def cancel_timeout(self, context_id: str) -> Tuple[bool, str]:
        """
        Cancel a timeout context.
        
        Args:
            context_id: Context identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not context_id:
                return (False, "context_id cannot be empty")
            
            if context_id not in self.contexts:
                return (False, f"Context {context_id} not found")
            
            context = self.contexts[context_id]
            
            if context.status != TimeoutStatus.ACTIVE:
                return (False, f"Context is not active (status: {context.status.value})")
            
            # Mark as cancelled
            context.status = TimeoutStatus.CANCELLED
            context.completion_time = time.time()
            
            # Update metrics
            self.metrics.cancelled_contexts += 1
            self.metrics.active_contexts = len([c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE])
            
            return (True, f"Context cancelled")
    
    def get_remaining_time(self, context_id: str) -> Tuple[bool, float, str]:
        """
        Get remaining time for a context.
        
        Args:
            context_id: Context identifier
        
        Returns:
            Tuple of (success, remaining_seconds, message)
        """
        with self.lock:
            if not context_id:
                return (False, 0.0, "context_id cannot be empty")
            
            if context_id not in self.contexts:
                return (False, 0.0, f"Context {context_id} not found")
            
            context = self.contexts[context_id]
            remaining = context.remaining_seconds()
            
            return (True, remaining, f"{remaining:.2f}s remaining")
    
    def get_context(self, context_id: str) -> Tuple[bool, Optional[TimeoutContext], str]:
        """
        Get a timeout context.
        
        Args:
            context_id: Context identifier
        
        Returns:
            Tuple of (success, context or None, message)
        """
        with self.lock:
            if not context_id:
                return (False, None, "context_id cannot be empty")
            
            if context_id not in self.contexts:
                return (False, None, f"Context {context_id} not found")
            
            return (True, self.contexts[context_id], "Context retrieved")
    
    def get_active_contexts(self) -> List[TimeoutContext]:
        """
        Get all active timeout contexts.
        
        Returns:
            List of active contexts
        """
        with self.lock:
            return [c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE]
    
    def get_expired_contexts(self) -> List[TimeoutContext]:
        """
        Get all expired timeout contexts.
        
        Returns:
            List of expired contexts
        """
        with self.lock:
            return [c for c in self.contexts.values() if c.status == TimeoutStatus.EXPIRED]
    
    def cleanup_expired(self) -> Tuple[bool, int, str]:
        """
        Cleanup expired timeout contexts.
        
        Returns:
            Tuple of (success, count_removed, message)
        """
        with self.lock:
            initial_count = len(self.contexts)
            
            # Remove expired and completed contexts
            to_remove = [
                cid for cid, ctx in self.contexts.items()
                if ctx.status in {TimeoutStatus.EXPIRED, TimeoutStatus.COMPLETED, TimeoutStatus.CANCELLED}
            ]
            
            for cid in to_remove:
                self.contexts.pop(cid)
            
            removed = len(to_remove)
            
            return (True, removed, f"Removed {removed} contexts")
    
    def _try_cleanup(self) -> None:
        """Try to cleanup if interval has elapsed."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.config.cleanup_interval_seconds:
            return
        
        self.cleanup_expired()
        self.last_cleanup = current_time
    
    def _update_average_duration(self) -> None:
        """Update average duration metric."""
        if self.durations:
            self.metrics.average_duration = sum(self.durations) / len(self.durations)
    
    def _update_timeout_rate(self) -> None:
        """Update timeout rate metric."""
        total_completed = self.metrics.completed_contexts + self.metrics.expired_contexts
        
        if total_completed > 0:
            self.metrics.timeout_rate = (self.metrics.expired_contexts / total_completed) * 100.0
    
    def _generate_context_id(self, operation_name: str) -> str:
        """Generate unique context ID."""
        self.context_counter += 1
        timestamp = int(time.time())
        safe_name = operation_name.replace(' ', '_')[:30]
        return f"timeout_{safe_name}_{timestamp}_{self.context_counter}"
    
    def get_metrics(self) -> TimeoutMetrics:
        """
        Get timeout manager metrics.
        
        Returns:
            TimeoutMetrics object
        """
        with self.lock:
            active_count = len([c for c in self.contexts.values() if c.status == TimeoutStatus.ACTIVE])
            
            return TimeoutMetrics(
                total_contexts=self.metrics.total_contexts,
                active_contexts=active_count,
                completed_contexts=self.metrics.completed_contexts,
                expired_contexts=self.metrics.expired_contexts,
                cancelled_contexts=self.metrics.cancelled_contexts,
                warnings_triggered=self.metrics.warnings_triggered,
                average_duration=self.metrics.average_duration,
                timeout_rate=self.metrics.timeout_rate
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get timeout manager statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            metrics = self.get_metrics()
            
            return {
                "default_timeout_seconds": self.config.default_timeout_seconds,
                "warning_threshold_percent": self.config.warning_threshold_percent,
                "max_contexts": self.config.max_contexts,
                "total_contexts": metrics.total_contexts,
                "active_contexts": metrics.active_contexts,
                "completed_contexts": metrics.completed_contexts,
                "expired_contexts": metrics.expired_contexts,
                "cancelled_contexts": metrics.cancelled_contexts,
                "warnings_triggered": metrics.warnings_triggered,
                "average_duration": f"{metrics.average_duration:.3f}s",
                "timeout_rate": f"{metrics.timeout_rate:.2f}%",
                "total_stored_contexts": len(self.contexts)
            }
    
    def reset(self) -> None:
        """Reset timeout manager to initial state."""
        with self.lock:
            self.contexts.clear()
            self.metrics = TimeoutMetrics()
            self.durations.clear()
            self.context_counter = 0
            self.last_cleanup = time.time()
