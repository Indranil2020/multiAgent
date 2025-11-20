"""
Rate limiting implementation.

This module provides comprehensive rate limiting using token bucket algorithm
with support for multiple keys, burst handling, and detailed statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any
from enum import Enum
import time
import threading


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiter.
    
    Attributes:
        max_requests: Maximum requests allowed
        time_window_seconds: Time window in seconds
        burst_size: Maximum burst size (tokens that can accumulate)
        refill_rate: Rate at which tokens refill per second
        strategy: Rate limiting strategy
        enable_metrics: Enable detailed metrics collection
    """
    max_requests: int
    time_window_seconds: float
    burst_size: Optional[int] = None
    refill_rate: Optional[float] = None
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Initialize derived values."""
        if self.burst_size is None:
            self.burst_size = self.max_requests
        
        if self.refill_rate is None:
            self.refill_rate = self.max_requests / self.time_window_seconds
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.max_requests < 1:
            return (False, "max_requests must be at least 1")
        
        if self.time_window_seconds <= 0:
            return (False, "time_window_seconds must be positive")
        
        if self.burst_size and self.burst_size < self.max_requests:
            return (False, "burst_size cannot be less than max_requests")
        
        if self.refill_rate and self.refill_rate <= 0:
            return (False, "refill_rate must be positive")
        
        return (True, "")


@dataclass
class RateLimitMetrics:
    """
    Metrics for rate limiter.
    
    Attributes:
        total_requests: Total requests attempted
        allowed_requests: Number of allowed requests
        rejected_requests: Number of rejected requests
        current_tokens: Current token count
        tokens_consumed: Total tokens consumed
        tokens_refilled: Total tokens refilled
        average_wait_time: Average wait time for rejected requests
        peak_usage: Peak usage percentage
    """
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_tokens: float = 0.0
    tokens_consumed: float = 0.0
    tokens_refilled: float = 0.0
    average_wait_time: float = 0.0
    peak_usage: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Implements token bucket algorithm for rate limiting with
    burst support, automatic refill, and comprehensive metrics.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiter configuration
        """
        self.config = config
        self.tokens = float(config.burst_size or config.max_requests)
        self.last_refill_time = time.time()
        self.metrics = RateLimitMetrics()
        self.metrics.current_tokens = self.tokens
        self.lock = threading.Lock()
        
        # Request history for sliding window
        self.request_history: List[float] = []
        
        # Wait time tracking
        self.wait_times: List[float] = []
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate rate limiter configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def try_acquire(self, tokens: int = 1) -> Tuple[bool, str]:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Tuple of (acquired, message)
        """
        with self.lock:
            if tokens < 1:
                return (False, "tokens must be at least 1")
            
            # Refill tokens
            self._refill_tokens()
            
            # Update metrics
            if self.config.enable_metrics:
                self.metrics.total_requests += 1
            
            # Check if enough tokens available
            if self.tokens >= tokens:
                self.tokens -= tokens
                
                if self.config.enable_metrics:
                    self.metrics.allowed_requests += 1
                    self.metrics.tokens_consumed += tokens
                    self.metrics.current_tokens = self.tokens
                    self._update_peak_usage()
                
                return (True, f"Acquired {tokens} tokens ({self.tokens:.2f} remaining)")
            
            # Not enough tokens
            if self.config.enable_metrics:
                self.metrics.rejected_requests += 1
                wait_time = self._calculate_wait_time(tokens)
                self.wait_times.append(wait_time)
                self._update_average_wait_time()
            
            return (False, f"Rate limit exceeded (need {tokens}, have {self.tokens:.2f})")
    
    def acquire_blocking(
        self,
        tokens: int = 1,
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Acquire tokens, blocking until available or timeout.
        
        Args:
            tokens: Number of tokens to acquire
            timeout_seconds: Maximum time to wait
        
        Returns:
            Tuple of (acquired, message)
        """
        start_time = time.time()
        
        while True:
            success, msg = self.try_acquire(tokens)
            
            if success:
                return (True, msg)
            
            # Check timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    return (False, f"Timeout after {elapsed:.2f}s")
            
            # Wait before retry
            wait_time = self._calculate_wait_time(tokens)
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill_time
        
        if elapsed <= 0:
            return
        
        # Calculate tokens to add
        refill_rate = self.config.refill_rate or (
            self.config.max_requests / self.config.time_window_seconds
        )
        tokens_to_add = elapsed * refill_rate
        
        # Add tokens up to burst size
        burst_size = self.config.burst_size or self.config.max_requests
        old_tokens = self.tokens
        self.tokens = min(self.tokens + tokens_to_add, burst_size)
        
        # Update metrics
        if self.config.enable_metrics:
            self.metrics.tokens_refilled += (self.tokens - old_tokens)
            self.metrics.current_tokens = self.tokens
        
        self.last_refill_time = current_time
    
    def _calculate_wait_time(self, tokens: int) -> float:
        """Calculate time to wait for tokens to be available."""
        tokens_needed = tokens - self.tokens
        
        if tokens_needed <= 0:
            return 0.0
        
        refill_rate = self.config.refill_rate or (
            self.config.max_requests / self.config.time_window_seconds
        )
        
        return tokens_needed / refill_rate
    
    def _update_peak_usage(self) -> None:
        """Update peak usage metric."""
        burst_size = self.config.burst_size or self.config.max_requests
        usage_percent = ((burst_size - self.tokens) / burst_size) * 100
        
        if usage_percent > self.metrics.peak_usage:
            self.metrics.peak_usage = usage_percent
    
    def _update_average_wait_time(self) -> None:
        """Update average wait time metric."""
        if self.wait_times:
            self.metrics.average_wait_time = sum(self.wait_times) / len(self.wait_times)
    
    def get_available_tokens(self) -> Tuple[bool, float, str]:
        """
        Get number of available tokens.
        
        Returns:
            Tuple of (success, token_count, message)
        """
        with self.lock:
            self._refill_tokens()
            return (True, self.tokens, f"{self.tokens:.2f} tokens available")
    
    def get_wait_time(self, tokens: int = 1) -> Tuple[bool, float, str]:
        """
        Get estimated wait time for tokens.
        
        Args:
            tokens: Number of tokens needed
        
        Returns:
            Tuple of (success, wait_seconds, message)
        """
        with self.lock:
            if tokens < 1:
                return (False, 0.0, "tokens must be at least 1")
            
            self._refill_tokens()
            wait_time = self._calculate_wait_time(tokens)
            
            return (True, wait_time, f"Wait time: {wait_time:.2f}s")
    
    def reset(self) -> None:
        """Reset rate limiter to initial state."""
        with self.lock:
            burst_size = self.config.burst_size or self.config.max_requests
            self.tokens = float(burst_size)
            self.last_refill_time = time.time()
            self.metrics = RateLimitMetrics()
            self.metrics.current_tokens = self.tokens
            self.request_history.clear()
            self.wait_times.clear()
    
    def get_metrics(self) -> RateLimitMetrics:
        """
        Get rate limiter metrics.
        
        Returns:
            RateLimitMetrics object
        """
        with self.lock:
            self._refill_tokens()
            return RateLimitMetrics(
                total_requests=self.metrics.total_requests,
                allowed_requests=self.metrics.allowed_requests,
                rejected_requests=self.metrics.rejected_requests,
                current_tokens=self.tokens,
                tokens_consumed=self.metrics.tokens_consumed,
                tokens_refilled=self.metrics.tokens_refilled,
                average_wait_time=self.metrics.average_wait_time,
                peak_usage=self.metrics.peak_usage
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            self._refill_tokens()
            metrics = self.get_metrics()
            
            burst_size = self.config.burst_size or self.config.max_requests
            usage_percent = ((burst_size - self.tokens) / burst_size) * 100
            
            return {
                "max_requests": self.config.max_requests,
                "time_window_seconds": self.config.time_window_seconds,
                "burst_size": burst_size,
                "refill_rate": self.config.refill_rate,
                "current_tokens": f"{self.tokens:.2f}",
                "current_usage_percent": f"{usage_percent:.2f}%",
                "total_requests": metrics.total_requests,
                "allowed_requests": metrics.allowed_requests,
                "rejected_requests": metrics.rejected_requests,
                "rejection_rate": f"{(metrics.rejected_requests / max(metrics.total_requests, 1)) * 100:.2f}%",
                "tokens_consumed": f"{metrics.tokens_consumed:.2f}",
                "tokens_refilled": f"{metrics.tokens_refilled:.2f}",
                "average_wait_time": f"{metrics.average_wait_time:.3f}s",
                "peak_usage": f"{metrics.peak_usage:.2f}%"
            }


class MultiKeyRateLimiter:
    """
    Rate limiter with per-key limits.
    
    Manages separate rate limiters for different keys (e.g., per-user,
    per-IP, per-resource) with automatic cleanup of inactive limiters.
    """
    
    def __init__(
        self,
        config: RateLimitConfig,
        cleanup_interval_seconds: float = 300.0,
        inactive_threshold_seconds: float = 600.0
    ):
        """
        Initialize multi-key rate limiter.
        
        Args:
            config: Rate limiter configuration (applied to all keys)
            cleanup_interval_seconds: Interval for cleanup
            inactive_threshold_seconds: Threshold for inactive limiters
        """
        self.config = config
        self.cleanup_interval = cleanup_interval_seconds
        self.inactive_threshold = inactive_threshold_seconds
        self.limiters: Dict[str, RateLimiter] = {}
        self.last_access: Dict[str, float] = {}
        self.last_cleanup = time.time()
        self.lock = threading.Lock()
        
        # Global metrics
        self.total_keys = 0
        self.active_keys = 0
        self.cleaned_keys = 0
    
    def try_acquire(self, key: str, tokens: int = 1) -> Tuple[bool, str]:
        """
        Try to acquire tokens for a specific key.
        
        Args:
            key: Key identifier
            tokens: Number of tokens to acquire
        
        Returns:
            Tuple of (acquired, message)
        """
        if not key:
            return (False, "key cannot be empty")
        
        with self.lock:
            # Get or create limiter for key
            if key not in self.limiters:
                self.limiters[key] = RateLimiter(self.config)
                self.total_keys += 1
            
            # Update last access
            self.last_access[key] = time.time()
            
            # Try cleanup if needed
            self._try_cleanup()
        
        # Acquire tokens (outside lock to avoid holding during acquire)
        return self.limiters[key].try_acquire(tokens)
    
    def acquire_blocking(
        self,
        key: str,
        tokens: int = 1,
        timeout_seconds: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Acquire tokens for a key, blocking until available.
        
        Args:
            key: Key identifier
            tokens: Number of tokens to acquire
            timeout_seconds: Maximum time to wait
        
        Returns:
            Tuple of (acquired, message)
        """
        if not key:
            return (False, "key cannot be empty")
        
        with self.lock:
            if key not in self.limiters:
                self.limiters[key] = RateLimiter(self.config)
                self.total_keys += 1
            
            self.last_access[key] = time.time()
        
        return self.limiters[key].acquire_blocking(tokens, timeout_seconds)
    
    def get_limiter(self, key: str) -> Tuple[bool, Optional[RateLimiter], str]:
        """
        Get rate limiter for a specific key.
        
        Args:
            key: Key identifier
        
        Returns:
            Tuple of (success, limiter or None, message)
        """
        if not key:
            return (False, None, "key cannot be empty")
        
        with self.lock:
            if key not in self.limiters:
                return (False, None, f"No limiter for key '{key}'")
            
            return (True, self.limiters[key], "Limiter retrieved")
    
    def remove_key(self, key: str) -> Tuple[bool, str]:
        """
        Remove rate limiter for a key.
        
        Args:
            key: Key identifier
        
        Returns:
            Tuple of (success, message)
        """
        if not key:
            return (False, "key cannot be empty")
        
        with self.lock:
            if key not in self.limiters:
                return (False, f"No limiter for key '{key}'")
            
            self.limiters.pop(key)
            self.last_access.pop(key, None)
            
            return (True, f"Removed limiter for key '{key}'")
    
    def _try_cleanup(self) -> None:
        """Try to cleanup inactive limiters."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        # Find inactive keys
        inactive_keys = [
            key for key, last_time in self.last_access.items()
            if current_time - last_time > self.inactive_threshold
        ]
        
        # Remove inactive limiters
        for key in inactive_keys:
            self.limiters.pop(key, None)
            self.last_access.pop(key, None)
            self.cleaned_keys += 1
        
        self.last_cleanup = current_time
        self.active_keys = len(self.limiters)
    
    def force_cleanup(self) -> Tuple[bool, int, str]:
        """
        Force cleanup of inactive limiters.
        
        Returns:
            Tuple of (success, count_removed, message)
        """
        with self.lock:
            initial_count = len(self.limiters)
            self._try_cleanup()
            removed = initial_count - len(self.limiters)
            
            return (True, removed, f"Removed {removed} inactive limiters")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all keys.
        
        Returns:
            Dictionary with all stats
        """
        with self.lock:
            return {
                "total_keys_created": self.total_keys,
                "active_keys": len(self.limiters),
                "cleaned_keys": self.cleaned_keys,
                "per_key_stats": {
                    key: limiter.get_stats()
                    for key, limiter in self.limiters.items()
                }
            }
    
    def reset_all(self) -> None:
        """Reset all rate limiters."""
        with self.lock:
            for limiter in self.limiters.values():
                limiter.reset()
    
    def clear_all(self) -> None:
        """Clear all rate limiters."""
        with self.lock:
            self.limiters.clear()
            self.last_access.clear()
            self.total_keys = 0
            self.active_keys = 0
            self.cleaned_keys = 0
