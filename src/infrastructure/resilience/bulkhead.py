"""
Bulkhead pattern implementation.

This module implements the bulkhead pattern for resource isolation,
preventing resource exhaustion by limiting concurrent operations and
providing request queuing with comprehensive monitoring.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Callable
from enum import Enum
from collections import deque
import time
import threading


class BulkheadState(Enum):
    """States of bulkhead."""
    NORMAL = "normal"  # Operating normally
    SATURATED = "saturated"  # All slots full, using queue
    OVERLOADED = "overloaded"  # Queue full, rejecting requests


class RequestPriority(Enum):
    """Priority levels for requests."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BulkheadConfig:
    """
    Configuration for bulkhead.
    
    Attributes:
        max_concurrent: Maximum concurrent operations
        max_queue_size: Maximum queue size
        queue_timeout_seconds: Timeout for queued requests
        name: Name of this bulkhead
        enable_priorities: Enable priority-based queuing
        fair_queuing: Use fair queuing (FIFO) instead of priorities
    """
    max_concurrent: int
    max_queue_size: int
    queue_timeout_seconds: float
    name: str
    enable_priorities: bool = False
    fair_queuing: bool = True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.max_concurrent < 1:
            return (False, "max_concurrent must be at least 1")
        
        if self.max_queue_size < 0:
            return (False, "max_queue_size cannot be negative")
        
        if self.queue_timeout_seconds <= 0:
            return (False, "queue_timeout_seconds must be positive")
        
        if not self.name:
            return (False, "name cannot be empty")
        
        return (True, "")


@dataclass
class QueuedRequest:
    """
    A queued request waiting for execution.
    
    Attributes:
        request_id: Unique request identifier
        enqueue_time: When request was queued
        timeout_time: When request times out
        priority: Request priority
        metadata: Additional metadata
    """
    request_id: str
    enqueue_time: float
    timeout_time: float
    priority: RequestPriority = RequestPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return time.time() >= self.timeout_time
    
    def wait_time(self) -> float:
        """Get current wait time in seconds."""
        return time.time() - self.enqueue_time


@dataclass
class BulkheadMetrics:
    """
    Comprehensive metrics for bulkhead.
    
    Attributes:
        total_requests: Total requests attempted
        accepted_requests: Requests accepted for execution
        queued_requests: Requests queued
        rejected_requests: Requests rejected
        completed_requests: Requests completed
        timed_out_requests: Requests that timed out in queue
        current_concurrent: Current concurrent operations
        current_queue_size: Current queue size
        peak_concurrent: Peak concurrent operations
        peak_queue_size: Peak queue size
        average_wait_time: Average queue wait time
        average_execution_time: Average execution time
    """
    total_requests: int = 0
    accepted_requests: int = 0
    queued_requests: int = 0
    rejected_requests: int = 0
    completed_requests: int = 0
    timed_out_requests: int = 0
    current_concurrent: int = 0
    current_queue_size: int = 0
    peak_concurrent: int = 0
    peak_queue_size: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0


class Bulkhead:
    """
    Bulkhead for resource isolation.
    
    Limits concurrent operations and queues excess requests,
    preventing resource exhaustion and providing isolation
    between different parts of the system.
    """
    
    def __init__(
        self,
        config: BulkheadConfig,
        on_state_change: Optional[Callable[[BulkheadState, BulkheadState], None]] = None
    ):
        """
        Initialize bulkhead.
        
        Args:
            config: Bulkhead configuration
            on_state_change: Optional callback for state changes
        """
        self.config = config
        self.current_concurrent = 0
        self.queue: deque = deque()
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.state = BulkheadState.NORMAL
        self.on_state_change = on_state_change
        
        # Metrics
        self.metrics = BulkheadMetrics()
        
        # Wait time tracking
        self.wait_times: List[float] = []
        self.execution_times: List[float] = []
        
        # Request counter
        self.request_counter = 0
        
        # Thread safety
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate bulkhead configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def try_acquire(
        self,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Try to acquire a slot for execution.
        
        Args:
            priority: Request priority
            metadata: Optional metadata
        
        Returns:
            Tuple of (acquired, request_id or None, message)
        """
        with self.lock:
            self.metrics.total_requests += 1
            
            # Check if slot available
            if self.current_concurrent < self.config.max_concurrent:
                # Acquire immediately
                request_id = self._generate_request_id()
                self.current_concurrent += 1
                self.active_requests[request_id] = time.time()
                
                # Update metrics
                self.metrics.accepted_requests += 1
                self.metrics.current_concurrent = self.current_concurrent
                
                if self.current_concurrent > self.metrics.peak_concurrent:
                    self.metrics.peak_concurrent = self.current_concurrent
                
                self._update_state()
                
                return (True, request_id, "Slot acquired")
            
            # No slot available - try to queue
            if len(self.queue) < self.config.max_queue_size:
                request_id = self._generate_request_id()
                
                queued_request = QueuedRequest(
                    request_id=request_id,
                    enqueue_time=time.time(),
                    timeout_time=time.time() + self.config.queue_timeout_seconds,
                    priority=priority,
                    metadata=metadata or {}
                )
                
                # Insert based on priority if enabled
                if self.config.enable_priorities and not self.config.fair_queuing:
                    self._insert_by_priority(queued_request)
                else:
                    self.queue.append(queued_request)
                
                # Update metrics
                self.metrics.queued_requests += 1
                self.metrics.current_queue_size = len(self.queue)
                
                if len(self.queue) > self.metrics.peak_queue_size:
                    self.metrics.peak_queue_size = len(self.queue)
                
                self._update_state()
                
                return (False, request_id, f"Queued at position {len(self.queue)}")
            
            # Queue full - reject
            self.metrics.rejected_requests += 1
            self._update_state()
            
            return (False, None, "Bulkhead overloaded (queue full)")
    
    def acquire_blocking(
        self,
        timeout_seconds: Optional[float] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Acquire a slot, blocking until available or timeout.
        
        Args:
            timeout_seconds: Maximum time to wait
            priority: Request priority
            metadata: Optional metadata
        
        Returns:
            Tuple of (acquired, request_id or None, message)
        """
        start_time = time.time()
        
        # Try immediate acquire
        success, request_id, msg = self.try_acquire(priority, metadata)
        
        if success:
            return (True, request_id, msg)
        
        # If queued, wait for slot
        if request_id:
            return self._wait_for_slot(request_id, timeout_seconds)
        
        # Not queued (rejected) - return failure
        return (False, None, msg)
    
    def _wait_for_slot(
        self,
        request_id: str,
        timeout_seconds: Optional[float]
    ) -> Tuple[bool, Optional[str], str]:
        """Wait for a queued request to get a slot."""
        start_time = time.time()
        
        with self.condition:
            while True:
                # Check if request got a slot
                if request_id in self.active_requests:
                    wait_time = time.time() - start_time
                    self.wait_times.append(wait_time)
                    self._update_average_wait_time()
                    return (True, request_id, f"Slot acquired after {wait_time:.2f}s")
                
                # Check if request is still in queue
                in_queue = any(req.request_id == request_id for req in self.queue)
                
                if not in_queue:
                    # Request was removed (timed out or cancelled)
                    return (False, None, "Request timed out or was cancelled")
                
                # Check timeout
                if timeout_seconds is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout_seconds:
                        self._remove_from_queue(request_id)
                        return (False, None, f"Timeout after {elapsed:.2f}s")
                
                # Wait for notification
                self.condition.wait(timeout=0.1)
    
    def release(self, request_id: str) -> Tuple[bool, str]:
        """
        Release a slot.
        
        Args:
            request_id: Request identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.condition:
            if request_id not in self.active_requests:
                return (False, f"Request {request_id} not active")
            
            # Calculate execution time
            start_time = self.active_requests.pop(request_id)
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            self._update_average_execution_time()
            
            self.current_concurrent -= 1
            self.metrics.completed_requests += 1
            self.metrics.current_concurrent = self.current_concurrent
            
            # Try to promote queued request
            self._promote_from_queue()
            
            self._update_state()
            
            # Notify waiting threads
            self.condition.notify_all()
            
            return (True, f"Slot released (execution time: {execution_time:.2f}s)")
    
    def _promote_from_queue(self) -> None:
        """Promote a request from queue to active."""
        # Clean up expired requests first
        self._cleanup_expired_requests()
        
        if not self.queue:
            return
        
        if self.current_concurrent >= self.config.max_concurrent:
            return
        
        # Get next request
        queued_request = self.queue.popleft()
        
        # Activate request
        self.current_concurrent += 1
        self.active_requests[queued_request.request_id] = time.time()
        
        # Update metrics
        self.metrics.current_concurrent = self.current_concurrent
        self.metrics.current_queue_size = len(self.queue)
        
        if self.current_concurrent > self.metrics.peak_concurrent:
            self.metrics.peak_concurrent = self.current_concurrent
    
    def _cleanup_expired_requests(self) -> None:
        """Remove expired requests from queue."""
        current_time = time.time()
        expired = []
        
        for req in self.queue:
            if req.is_expired():
                expired.append(req)
        
        for req in expired:
            self.queue.remove(req)
            self.metrics.timed_out_requests += 1
        
        if expired:
            self.metrics.current_queue_size = len(self.queue)
    
    def _insert_by_priority(self, request: QueuedRequest) -> None:
        """Insert request into queue based on priority."""
        # Find insertion point
        insert_index = len(self.queue)
        
        for i, queued in enumerate(self.queue):
            if request.priority.value > queued.priority.value:
                insert_index = i
                break
        
        self.queue.insert(insert_index, request)
    
    def _remove_from_queue(self, request_id: str) -> bool:
        """Remove a request from queue."""
        for req in self.queue:
            if req.request_id == request_id:
                self.queue.remove(req)
                self.metrics.current_queue_size = len(self.queue)
                return True
        
        return False
    
    def _update_state(self) -> None:
        """Update bulkhead state."""
        old_state = self.state
        
        if len(self.queue) >= self.config.max_queue_size:
            new_state = BulkheadState.OVERLOADED
        elif self.current_concurrent >= self.config.max_concurrent:
            new_state = BulkheadState.SATURATED
        else:
            new_state = BulkheadState.NORMAL
        
        if new_state != old_state:
            self.state = new_state
            
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    def _update_average_wait_time(self) -> None:
        """Update average wait time metric."""
        if self.wait_times:
            self.metrics.average_wait_time = sum(self.wait_times) / len(self.wait_times)
    
    def _update_average_execution_time(self) -> None:
        """Update average execution time metric."""
        if self.execution_times:
            self.metrics.average_execution_time = sum(self.execution_times) / len(self.execution_times)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self.request_counter += 1
        return f"{self.config.name}_req_{int(time.time())}_{self.request_counter}"
    
    def get_state(self) -> BulkheadState:
        """Get current bulkhead state."""
        with self.lock:
            return self.state
    
    def get_queue_position(self, request_id: str) -> Tuple[bool, Optional[int], str]:
        """
        Get position of request in queue.
        
        Args:
            request_id: Request identifier
        
        Returns:
            Tuple of (success, position or None, message)
        """
        with self.lock:
            for i, req in enumerate(self.queue):
                if req.request_id == request_id:
                    return (True, i + 1, f"Position {i + 1} of {len(self.queue)}")
            
            return (False, None, "Request not in queue")
    
    def cancel_request(self, request_id: str) -> Tuple[bool, str]:
        """
        Cancel a queued request.
        
        Args:
            request_id: Request identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.condition:
            if self._remove_from_queue(request_id):
                self.condition.notify_all()
                return (True, f"Request {request_id} cancelled")
            
            return (False, f"Request {request_id} not found in queue")
    
    def get_metrics(self) -> BulkheadMetrics:
        """
        Get bulkhead metrics.
        
        Returns:
            BulkheadMetrics object
        """
        with self.lock:
            return BulkheadMetrics(
                total_requests=self.metrics.total_requests,
                accepted_requests=self.metrics.accepted_requests,
                queued_requests=self.metrics.queued_requests,
                rejected_requests=self.metrics.rejected_requests,
                completed_requests=self.metrics.completed_requests,
                timed_out_requests=self.metrics.timed_out_requests,
                current_concurrent=self.current_concurrent,
                current_queue_size=len(self.queue),
                peak_concurrent=self.metrics.peak_concurrent,
                peak_queue_size=self.metrics.peak_queue_size,
                average_wait_time=self.metrics.average_wait_time,
                average_execution_time=self.metrics.average_execution_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get bulkhead statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            metrics = self.get_metrics()
            
            utilization = (self.current_concurrent / self.config.max_concurrent) * 100
            queue_utilization = (len(self.queue) / max(self.config.max_queue_size, 1)) * 100
            
            return {
                "name": self.config.name,
                "state": self.state.value,
                "max_concurrent": self.config.max_concurrent,
                "current_concurrent": self.current_concurrent,
                "utilization_percent": f"{utilization:.2f}%",
                "max_queue_size": self.config.max_queue_size,
                "current_queue_size": len(self.queue),
                "queue_utilization_percent": f"{queue_utilization:.2f}%",
                "total_requests": metrics.total_requests,
                "accepted_requests": metrics.accepted_requests,
                "queued_requests": metrics.queued_requests,
                "rejected_requests": metrics.rejected_requests,
                "completed_requests": metrics.completed_requests,
                "timed_out_requests": metrics.timed_out_requests,
                "rejection_rate": f"{(metrics.rejected_requests / max(metrics.total_requests, 1)) * 100:.2f}%",
                "peak_concurrent": metrics.peak_concurrent,
                "peak_queue_size": metrics.peak_queue_size,
                "average_wait_time": f"{metrics.average_wait_time:.3f}s",
                "average_execution_time": f"{metrics.average_execution_time:.3f}s"
            }
    
    def reset(self) -> None:
        """Reset bulkhead to initial state."""
        with self.condition:
            self.current_concurrent = 0
            self.queue.clear()
            self.active_requests.clear()
            self.state = BulkheadState.NORMAL
            self.metrics = BulkheadMetrics()
            self.wait_times.clear()
            self.execution_times.clear()
            self.request_counter = 0
            
            self.condition.notify_all()
