"""
Circuit breaker pattern implementation.

This module implements the circuit breaker pattern to prevent cascading failures
by detecting failures and temporarily blocking requests to failing services.
Includes sliding window tracking, failure rate calculation, and comprehensive metrics.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict, Any
from enum import Enum
from collections import deque
import time
import threading


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Types of failures."""
    TIMEOUT = "timeout"
    ERROR = "error"
    EXCEPTION = "exception"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.
    
    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close circuit from half-open
        timeout_seconds: Seconds to wait before trying half-open
        name: Name of this circuit breaker
        failure_rate_threshold: Failure rate percentage (0-100) to open circuit
        minimum_requests: Minimum requests before calculating failure rate
        sliding_window_size: Size of sliding window for tracking
        half_open_max_calls: Maximum calls allowed in half-open state
    """
    failure_threshold: int
    success_threshold: int
    timeout_seconds: float
    name: str
    failure_rate_threshold: float = 50.0
    minimum_requests: int = 10
    sliding_window_size: int = 100
    half_open_max_calls: int = 3
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.failure_threshold < 1:
            return (False, "failure_threshold must be at least 1")
        
        if self.success_threshold < 1:
            return (False, "success_threshold must be at least 1")
        
        if self.timeout_seconds <= 0:
            return (False, "timeout_seconds must be positive")
        
        if not self.name:
            return (False, "name cannot be empty")
        
        if not 0 <= self.failure_rate_threshold <= 100:
            return (False, "failure_rate_threshold must be between 0 and 100")
        
        if self.minimum_requests < 1:
            return (False, "minimum_requests must be at least 1")
        
        if self.sliding_window_size < 1:
            return (False, "sliding_window_size must be at least 1")
        
        if self.half_open_max_calls < 1:
            return (False, "half_open_max_calls must be at least 1")
        
        return (True, "")


@dataclass
class CallRecord:
    """
    Record of a single call.
    
    Attributes:
        timestamp: When call was made
        success: Whether call succeeded
        failure_type: Type of failure if failed
        duration_ms: Call duration in milliseconds
    """
    timestamp: float
    success: bool
    failure_type: Optional[FailureType] = None
    duration_ms: float = 0.0


@dataclass
class CircuitBreakerMetrics:
    """
    Comprehensive metrics for circuit breaker.
    
    Attributes:
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        rejected_calls: Number of rejected calls (circuit open)
        failure_rate: Current failure rate percentage
        average_duration_ms: Average call duration
        state_transitions: Number of state transitions
        time_in_open: Total time spent in open state
        time_in_half_open: Total time spent in half-open state
        last_failure_time: Timestamp of last failure
    """
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    failure_rate: float = 0.0
    average_duration_ms: float = 0.0
    state_transitions: int = 0
    time_in_open: float = 0.0
    time_in_half_open: float = 0.0
    last_failure_time: float = 0.0


class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.
    
    The circuit breaker monitors operation failures and automatically
    opens to prevent further requests when failure threshold is reached.
    Includes sliding window tracking, failure rate calculation, and
    comprehensive metrics collection.
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            on_state_change: Optional callback for state changes
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change_time = time.time()
        self.on_state_change = on_state_change
        
        # Sliding window for tracking calls
        self.call_window: deque = deque(maxlen=config.sliding_window_size)
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Half-open state tracking
        self.half_open_calls = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # State duration tracking
        self.state_durations: Dict[CircuitState, float] = {
            CircuitState.CLOSED: 0.0,
            CircuitState.OPEN: 0.0,
            CircuitState.HALF_OPEN: 0.0
        }
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate circuit breaker configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def can_execute(self) -> Tuple[bool, str]:
        """
        Check if operation can be executed.
        
        Returns:
            Tuple of (can_execute, reason)
        """
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.CLOSED:
                return (True, "Circuit is closed")
            
            if self.state == CircuitState.OPEN:
                # Check if timeout has elapsed
                time_since_open = current_time - self.last_failure_time
                
                if time_since_open >= self.config.timeout_seconds:
                    # Transition to half-open
                    self._transition_to_half_open()
                    return (True, "Circuit transitioning to half-open")
                
                # Increment rejected calls
                self.metrics.rejected_calls += 1
                
                return (False, f"Circuit is open (retry in {self.config.timeout_seconds - time_since_open:.1f}s)")
            
            # HALF_OPEN state
            if self.half_open_calls >= self.config.half_open_max_calls:
                return (False, "Half-open call limit reached")
            
            return (True, "Circuit is half-open (testing)")
    
    def record_success(self, duration_ms: float = 0.0) -> Tuple[bool, str]:
        """
        Record a successful operation.
        
        Args:
            duration_ms: Operation duration in milliseconds
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            # Record call
            call_record = CallRecord(
                timestamp=time.time(),
                success=True,
                duration_ms=duration_ms
            )
            self.call_window.append(call_record)
            
            # Update metrics
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self._update_metrics()
            
            if self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
                return (True, "Success recorded (circuit closed)")
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                self.half_open_calls += 1
                
                if self.success_count >= self.config.success_threshold:
                    # Transition back to closed
                    self._transition_to_closed()
                    return (True, "Circuit closed after successful recovery")
                
                return (True, f"Success recorded ({self.success_count}/{self.config.success_threshold})")
            
            # OPEN state - shouldn't reach here
            return (False, "Cannot record success when circuit is open")
    
    def record_failure(
        self,
        failure_type: FailureType = FailureType.UNKNOWN,
        duration_ms: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Record a failed operation.
        
        Args:
            failure_type: Type of failure
            duration_ms: Operation duration in milliseconds
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            current_time = time.time()
            self.last_failure_time = current_time
            
            # Record call
            call_record = CallRecord(
                timestamp=current_time,
                success=False,
                failure_type=failure_type,
                duration_ms=duration_ms
            )
            self.call_window.append(call_record)
            
            # Update metrics
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = current_time
            self._update_metrics()
            
            if self.state == CircuitState.CLOSED:
                self.failure_count += 1
                
                # Check failure threshold
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
                    return (True, "Circuit opened due to failure threshold")
                
                # Check failure rate
                if self._should_open_by_failure_rate():
                    self._transition_to_open()
                    return (True, "Circuit opened due to failure rate")
                
                return (True, f"Failure recorded ({self.failure_count}/{self.config.failure_threshold})")
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                # Any failure in half-open immediately opens circuit
                self._transition_to_open()
                return (True, "Circuit reopened after half-open failure")
            
            # OPEN state
            return (True, "Failure recorded (circuit already open)")
    
    def _should_open_by_failure_rate(self) -> bool:
        """Check if circuit should open based on failure rate."""
        if len(self.call_window) < self.config.minimum_requests:
            return False
        
        failure_rate = self._calculate_failure_rate()
        return failure_rate >= self.config.failure_rate_threshold
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate from sliding window."""
        if not self.call_window:
            return 0.0
        
        failed = sum(1 for call in self.call_window if not call.success)
        total = len(self.call_window)
        
        return (failed / total) * 100.0
    
    def _update_metrics(self) -> None:
        """Update comprehensive metrics."""
        # Calculate failure rate
        self.metrics.failure_rate = self._calculate_failure_rate()
        
        # Calculate average duration
        if self.call_window:
            total_duration = sum(call.duration_ms for call in self.call_window)
            self.metrics.average_duration_ms = total_duration / len(self.call_window)
    
    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        old_state = self.state
        self._update_state_duration(old_state)
        
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.half_open_calls = 0
        self.last_state_change_time = time.time()
        self.metrics.state_transitions += 1
        
        if self.on_state_change:
            self.on_state_change(old_state, CircuitState.OPEN)
    
    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        old_state = self.state
        self._update_state_duration(old_state)
        
        self.state = CircuitState.HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_state_change_time = time.time()
        self.metrics.state_transitions += 1
        
        if self.on_state_change:
            self.on_state_change(old_state, CircuitState.HALF_OPEN)
    
    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        old_state = self.state
        self._update_state_duration(old_state)
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_state_change_time = time.time()
        self.metrics.state_transitions += 1
        
        if self.on_state_change:
            self.on_state_change(old_state, CircuitState.CLOSED)
    
    def _update_state_duration(self, state: CircuitState) -> None:
        """Update duration spent in a state."""
        duration = time.time() - self.last_state_change_time
        self.state_durations[state] += duration
        
        if state == CircuitState.OPEN:
            self.metrics.time_in_open += duration
        elif state == CircuitState.HALF_OPEN:
            self.metrics.time_in_half_open += duration
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        with self.lock:
            return self.state
    
    def force_open(self) -> Tuple[bool, str]:
        """
        Manually force circuit to open state.
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                return (False, "Circuit is already open")
            
            self._transition_to_open()
            return (True, "Circuit manually opened")
    
    def force_close(self) -> Tuple[bool, str]:
        """
        Manually force circuit to closed state.
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return (False, "Circuit is already closed")
            
            self._transition_to_closed()
            return (True, "Circuit manually closed")
    
    def force_half_open(self) -> Tuple[bool, str]:
        """
        Manually force circuit to half-open state.
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                return (False, "Circuit is already half-open")
            
            self._transition_to_half_open()
            return (True, "Circuit manually set to half-open")
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """
        Get comprehensive circuit breaker metrics.
        
        Returns:
            CircuitBreakerMetrics object
        """
        with self.lock:
            # Update current state duration
            self._update_state_duration(self.state)
            self.last_state_change_time = time.time()
            
            return CircuitBreakerMetrics(
                total_calls=self.metrics.total_calls,
                successful_calls=self.metrics.successful_calls,
                failed_calls=self.metrics.failed_calls,
                rejected_calls=self.metrics.rejected_calls,
                failure_rate=self.metrics.failure_rate,
                average_duration_ms=self.metrics.average_duration_ms,
                state_transitions=self.metrics.state_transitions,
                time_in_open=self.state_durations[CircuitState.OPEN],
                time_in_half_open=self.state_durations[CircuitState.HALF_OPEN],
                last_failure_time=self.metrics.last_failure_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get circuit breaker statistics.
        
        Returns:
            Dictionary with current stats
        """
        with self.lock:
            metrics = self.get_metrics()
            
            return {
                "name": self.config.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "time_in_current_state": time.time() - self.last_state_change_time,
                "last_failure_time": self.last_failure_time,
                "total_calls": metrics.total_calls,
                "successful_calls": metrics.successful_calls,
                "failed_calls": metrics.failed_calls,
                "rejected_calls": metrics.rejected_calls,
                "failure_rate": f"{metrics.failure_rate:.2f}%",
                "average_duration_ms": f"{metrics.average_duration_ms:.2f}",
                "state_transitions": metrics.state_transitions,
                "time_in_open": f"{metrics.time_in_open:.2f}s",
                "time_in_half_open": f"{metrics.time_in_half_open:.2f}s",
                "window_size": len(self.call_window),
                "half_open_calls": self.half_open_calls
            }
    
    def get_failure_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of failures by type.
        
        Returns:
            Dictionary mapping failure type to count
        """
        with self.lock:
            breakdown = {ft.value: 0 for ft in FailureType}
            
            for call in self.call_window:
                if not call.success and call.failure_type:
                    breakdown[call.failure_type.value] += 1
            
            return breakdown
    
    def get_recent_calls(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent call records.
        
        Args:
            count: Number of recent calls to return
        
        Returns:
            List of call record dictionaries
        """
        with self.lock:
            recent = list(self.call_window)[-count:]
            
            return [
                {
                    "timestamp": call.timestamp,
                    "success": call.success,
                    "failure_type": call.failure_type.value if call.failure_type else None,
                    "duration_ms": call.duration_ms
                }
                for call in recent
            ]
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0
            self.last_state_change_time = time.time()
            self.half_open_calls = 0
            self.call_window.clear()
            self.metrics = CircuitBreakerMetrics()
            self.state_durations = {
                CircuitState.CLOSED: 0.0,
                CircuitState.OPEN: 0.0,
                CircuitState.HALF_OPEN: 0.0
            }
    
    def clear_history(self) -> None:
        """Clear call history but maintain current state."""
        with self.lock:
            self.call_window.clear()
            self.metrics.total_calls = 0
            self.metrics.successful_calls = 0
            self.metrics.failed_calls = 0
            self.metrics.rejected_calls = 0
            self.metrics.failure_rate = 0.0
            self.metrics.average_duration_ms = 0.0
