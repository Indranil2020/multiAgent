"""
Request Logging Middleware.

This module provides middleware for logging API requests and responses
with detailed information for monitoring and debugging.

Features:
- Request/response logging
- Performance timing
- Error tracking
- Structured logging
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class RequestLoggerConfig:
    """
    Configuration for request logger.

    Attributes:
        log_requests: Whether to log requests
        log_responses: Whether to log responses
        log_request_body: Whether to log request body
        log_response_body: Whether to log response body
        max_body_length: Maximum body length to log
        log_headers: Whether to log headers
        exclude_paths: Paths to exclude from logging
    """
    log_requests: bool = True
    log_responses: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    max_body_length: int = 1000
    log_headers: bool = False
    exclude_paths: set = field(default_factory=set)

    def __post_init__(self):
        """Initialize default exclude paths."""
        if not self.exclude_paths:
            self.exclude_paths = {'/health', '/monitoring/health'}

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        if self.max_body_length < 0:
            return False

        return True


@dataclass
class RequestLogEntry:
    """
    Log entry for a request.

    Attributes:
        request_id: Unique request identifier
        method: HTTP method
        path: Request path
        ip_address: Client IP address
        status_code: Response status code
        request_time_ms: Request processing time
        timestamp: Request timestamp
        user_agent: User agent string
        error: Error message if any
    """
    request_id: str
    method: str
    path: str
    ip_address: str
    status_code: int = 0
    request_time_ms: int = 0
    timestamp: float = 0.0
    user_agent: str = ""
    error: str = ""


class RequestLogger:
    """
    Request logger implementation.

    This class maintains a log of API requests for monitoring and debugging.
    """

    def __init__(self, config: RequestLoggerConfig):
        """
        Initialize request logger.

        Args:
            config: Logger configuration
        """
        if not config.validate():
            raise ValueError("Invalid request logger configuration")

        self.config = config
        self.log_entries: List[RequestLogEntry] = []
        self.request_count = 0
        self.error_count = 0
        self.total_request_time_ms = 0

    def should_log(self, path: str) -> bool:
        """
        Check if path should be logged.

        Args:
            path: Request path

        Returns:
            True if should log
        """
        if not self.config.log_requests:
            return False

        if path in self.config.exclude_paths:
            return False

        return True

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        ip_address: str,
        user_agent: str = ""
    ) -> float:
        """
        Log a request start.

        Args:
            request_id: Unique request ID
            method: HTTP method
            path: Request path
            ip_address: Client IP
            user_agent: User agent

        Returns:
            Start time for timing
        """
        if not self.should_log(path):
            return time.time()

        start_time = time.time()

        # Store in temporary dict for completion
        # In production, would use proper logging framework

        self.request_count += 1

        return start_time

    def log_response(
        self,
        request_id: str,
        method: str,
        path: str,
        ip_address: str,
        status_code: int,
        start_time: float,
        user_agent: str = "",
        error: str = ""
    ) -> None:
        """
        Log a request completion.

        Args:
            request_id: Unique request ID
            method: HTTP method
            path: Request path
            ip_address: Client IP
            status_code: Response status code
            start_time: Request start time
            user_agent: User agent
            error: Error message if any
        """
        if not self.should_log(path):
            return

        # Calculate request time
        end_time = time.time()
        request_time_ms = int((end_time - start_time) * 1000)

        # Create log entry
        entry = RequestLogEntry(
            request_id=request_id,
            method=method,
            path=path,
            ip_address=ip_address,
            status_code=status_code,
            request_time_ms=request_time_ms,
            timestamp=start_time,
            user_agent=user_agent,
            error=error
        )

        # Store entry
        self.log_entries.append(entry)

        # Update stats
        self.total_request_time_ms += request_time_ms
        if status_code >= 400 or error:
            self.error_count += 1

        # In production, would write to logging framework
        # logger.info(f"{method} {path} {status_code} {request_time_ms}ms")

    def get_recent_logs(self, limit: int = 100) -> List[RequestLogEntry]:
        """
        Get recent log entries.

        Args:
            limit: Maximum number of entries

        Returns:
            List of log entries
        """
        if limit <= 0:
            limit = 100

        # Return most recent
        return self.log_entries[-limit:]

    def get_stats(self) -> Dict[str, any]:
        """
        Get logging statistics.

        Returns:
            Statistics dictionary
        """
        avg_request_time = 0.0
        if self.request_count > 0:
            avg_request_time = self.total_request_time_ms / self.request_count

        error_rate = 0.0
        if self.request_count > 0:
            error_rate = self.error_count / self.request_count

        return {
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': error_rate,
            'avg_request_time_ms': avg_request_time,
            'log_entries': len(self.log_entries)
        }

    def clear_old_logs(self, max_entries: int = 10000) -> int:
        """
        Clear old log entries to prevent memory growth.

        Args:
            max_entries: Maximum entries to keep

        Returns:
            Number of entries removed
        """
        if len(self.log_entries) <= max_entries:
            return 0

        entries_to_remove = len(self.log_entries) - max_entries
        self.log_entries = self.log_entries[entries_to_remove:]

        return entries_to_remove


class RequestLoggerMiddleware:
    """
    FastAPI middleware for request logging.

    This middleware integrates the RequestLogger with FastAPI to log
    all incoming requests and responses.

    In production, this would be a proper Starlette middleware:

    class RequestLoggerMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Generate request ID
            # Log request start
            # Call next
            # Log response
            # Add request ID header
    """

    def __init__(self, config: Optional[RequestLoggerConfig] = None):
        """
        Initialize request logger middleware.

        Args:
            config: Logger configuration
        """
        self.config = config or RequestLoggerConfig()
        self.logger = RequestLogger(self.config)

    def log_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        ip_address: str,
        user_agent: str = ""
    ) -> float:
        """
        Log request start.

        Args:
            request_id: Request ID
            method: HTTP method
            path: Request path
            ip_address: Client IP
            user_agent: User agent

        Returns:
            Start time
        """
        return self.logger.log_request(
            request_id,
            method,
            path,
            ip_address,
            user_agent
        )

    def log_request_end(
        self,
        request_id: str,
        method: str,
        path: str,
        ip_address: str,
        status_code: int,
        start_time: float,
        user_agent: str = "",
        error: str = ""
    ) -> None:
        """
        Log request end.

        Args:
            request_id: Request ID
            method: HTTP method
            path: Request path
            ip_address: Client IP
            status_code: Status code
            start_time: Start time
            user_agent: User agent
            error: Error message
        """
        self.logger.log_response(
            request_id,
            method,
            path,
            ip_address,
            status_code,
            start_time,
            user_agent,
            error
        )

    def get_recent_logs(self, limit: int = 100) -> List[RequestLogEntry]:
        """
        Get recent logs.

        Args:
            limit: Maximum entries

        Returns:
            Log entries
        """
        return self.logger.get_recent_logs(limit)

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics.

        Returns:
            Statistics dictionary
        """
        return self.logger.get_stats()
