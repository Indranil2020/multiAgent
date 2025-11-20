"""
Rate Limiting Middleware.

This module provides rate limiting middleware to protect the API from abuse
and ensure fair resource allocation.

Features:
- Per-IP rate limiting
- Configurable rate windows
- Sliding window algorithm
- Exempt paths for health checks
"""

from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class RateLimiterConfig:
    """
    Configuration for rate limiter.

    Attributes:
        requests_per_minute: Maximum requests per minute per IP
        window_seconds: Time window in seconds
        exempt_paths: Paths exempt from rate limiting
        enable_rate_limiting: Whether rate limiting is enabled
    """
    requests_per_minute: int = 100
    window_seconds: int = 60
    exempt_paths: Set[str] = field(default_factory=set)
    enable_rate_limiting: bool = True

    def __post_init__(self):
        """Initialize default exempt paths."""
        if not self.exempt_paths:
            self.exempt_paths = {'/health', '/monitoring/health'}

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        if self.requests_per_minute <= 0:
            return False

        if self.window_seconds <= 0:
            return False

        return True


@dataclass
class RateLimitEntry:
    """
    Rate limit entry for an IP address.

    Attributes:
        ip_address: Client IP address
        request_times: List of request timestamps
        first_request: Time of first request in window
    """
    ip_address: str
    request_times: list = field(default_factory=list)
    first_request: float = 0.0


class RateLimiter:
    """
    Rate limiter implementation using sliding window algorithm.

    This class tracks requests per IP and enforces rate limits.
    """

    def __init__(self, config: RateLimiterConfig):
        """
        Initialize rate limiter.

        Args:
            config: Rate limiter configuration
        """
        if not config.validate():
            raise ValueError("Invalid rate limiter configuration")

        self.config = config
        self.rate_limits: Dict[str, RateLimitEntry] = {}

    def is_allowed(self, ip_address: str, path: str) -> Tuple[bool, int]:
        """
        Check if request is allowed.

        Args:
            ip_address: Client IP address
            path: Request path

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        if not self.config.enable_rate_limiting:
            return True, self.config.requests_per_minute

        # Check if path is exempt
        if path in self.config.exempt_paths:
            return True, self.config.requests_per_minute

        now = time.time()

        # Get or create rate limit entry
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = RateLimitEntry(
                ip_address=ip_address,
                request_times=[],
                first_request=now
            )

        entry = self.rate_limits[ip_address]

        # Clean up old requests outside window
        window_start = now - self.config.window_seconds
        entry.request_times = [
            t for t in entry.request_times
            if t > window_start
        ]

        # Check if limit exceeded
        current_count = len(entry.request_times)
        if current_count >= self.config.requests_per_minute:
            remaining = 0
            return False, remaining

        # Record this request
        entry.request_times.append(now)

        remaining = self.config.requests_per_minute - (current_count + 1)
        return True, remaining

    def reset(self, ip_address: str) -> bool:
        """
        Reset rate limit for an IP.

        Args:
            ip_address: IP address to reset

        Returns:
            True if reset
        """
        if ip_address in self.rate_limits:
            del self.rate_limits[ip_address]
            return True
        return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        now = time.time()
        window_start = now - self.config.window_seconds

        active_ips = 0
        total_requests = 0

        for entry in self.rate_limits.values():
            recent_requests = [
                t for t in entry.request_times
                if t > window_start
            ]

            if recent_requests:
                active_ips += 1
                total_requests += len(recent_requests)

        return {
            'active_ips': active_ips,
            'total_requests_in_window': total_requests,
            'tracked_ips': len(self.rate_limits)
        }


class RateLimitMiddleware:
    """
    FastAPI middleware for rate limiting.

    This middleware integrates the RateLimiter with FastAPI to enforce
    rate limits on incoming requests.

    In production, this would be a proper Starlette middleware:

    class RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Check rate limit
            # Return 429 if exceeded
            # Add headers
            # Call next
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        """
        Initialize rate limit middleware.

        Args:
            config: Rate limiter configuration
        """
        self.config = config or RateLimiterConfig()
        self.limiter = RateLimiter(self.config)

    def check_rate_limit(
        self,
        ip_address: str,
        path: str
    ) -> Tuple[bool, int, Optional[Dict[str, str]]]:
        """
        Check rate limit and return headers.

        Args:
            ip_address: Client IP address
            path: Request path

        Returns:
            Tuple of (allowed, status_code, headers)
        """
        allowed, remaining = self.limiter.is_allowed(ip_address, path)

        headers = {
            'X-RateLimit-Limit': str(self.config.requests_per_minute),
            'X-RateLimit-Remaining': str(remaining),
            'X-RateLimit-Window': str(self.config.window_seconds)
        }

        if not allowed:
            headers['Retry-After'] = str(self.config.window_seconds)
            return False, 429, headers

        return True, 200, headers

    def get_stats(self) -> Dict[str, int]:
        """
        Get middleware statistics.

        Returns:
            Statistics dictionary
        """
        return self.limiter.get_stats()
