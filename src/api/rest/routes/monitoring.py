"""
Monitoring API Routes.

This module provides REST endpoints for system monitoring, health checks,
and performance metrics in the zero-error system.

Endpoints:
- GET /monitoring/health       - System health check
- GET /monitoring/metrics      - System performance metrics
- GET /monitoring/red-flags    - Red flag events
- GET /monitoring/consensus    - Consensus statistics
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HealthCheckResponse:
    """
    Response for health check.

    Attributes:
        status: Overall status (healthy/degraded/unhealthy)
        uptime_seconds: System uptime
        version: System version
        components: Component health status
        timestamp: Check timestamp
    """
    status: str
    uptime_seconds: float
    version: str
    components: Dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class PerformanceMetrics:
    """
    Performance metrics.

    Attributes:
        requests_per_second: Current RPS
        avg_response_time_ms: Average response time
        active_tasks: Number of active tasks
        active_agents: Number of active agents
        memory_usage_mb: Memory usage in MB
        cpu_usage_percent: CPU usage percentage
    """
    requests_per_second: float
    avg_response_time_ms: float
    active_tasks: int
    active_agents: int
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class MetricsResponse:
    """
    Response containing system metrics.

    Attributes:
        timestamp: Metrics timestamp
        performance: Performance metrics
        task_metrics: Task-related metrics
        agent_metrics: Agent-related metrics
        verification_metrics: Verification metrics
    """
    timestamp: float
    performance: PerformanceMetrics
    task_metrics: Dict[str, Any] = field(default_factory=dict)
    agent_metrics: Dict[str, Any] = field(default_factory=dict)
    verification_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RedFlagEvent:
    """
    Red flag event information.

    Attributes:
        event_id: Unique event identifier
        task_id: Associated task ID
        agent_id: Agent that raised flag
        flag_type: Type of red flag
        severity: Severity level (1-5)
        description: Event description
        timestamp: When flag was raised
        resolved: Whether flag was resolved
    """
    event_id: str
    task_id: str
    agent_id: str
    flag_type: str
    severity: int
    description: str
    timestamp: float
    resolved: bool = False


@dataclass
class RedFlagsResponse:
    """
    Response containing red flag events.

    Attributes:
        total_flags: Total red flags raised
        active_flags: Currently active flags
        resolved_flags: Resolved flags
        flags_by_severity: Count by severity level
        recent_events: Recent red flag events
    """
    total_flags: int
    active_flags: int
    resolved_flags: int
    flags_by_severity: Dict[int, int] = field(default_factory=dict)
    recent_events: List[RedFlagEvent] = field(default_factory=list)


@dataclass
class ConsensusStats:
    """
    Consensus voting statistics.

    Attributes:
        total_votes: Total votes cast
        consensus_achieved: Number of times consensus achieved
        consensus_failed: Number of times consensus failed
        avg_confidence: Average confidence score
        avg_voting_time_ms: Average voting time
        votes_by_task_type: Votes grouped by task type
    """
    total_votes: int
    consensus_achieved: int
    consensus_failed: int
    avg_confidence: float
    avg_voting_time_ms: float
    votes_by_task_type: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class MonitoringRouteHandler:
    """
    Handler for monitoring-related API routes.

    This class implements the business logic for system monitoring
    endpoints, providing health checks and performance metrics.
    """

    def __init__(
        self,
        swarm_coordinator=None,
        task_distributor=None,
        verification_stack=None
    ):
        """
        Initialize monitoring route handler.

        Args:
            swarm_coordinator: Agent swarm coordinator
            task_distributor: Task distributor
            verification_stack: Verification stack
        """
        self.swarm_coordinator = swarm_coordinator
        self.task_distributor = task_distributor
        self.verification_stack = verification_stack

        self.start_time = datetime.now().timestamp()
        self.version = "1.0.0"

        # Tracking data
        self.red_flags: Dict[str, RedFlagEvent] = {}
        self.consensus_data = {
            'total_votes': 0,
            'consensus_achieved': 0,
            'consensus_failed': 0,
            'total_confidence': 0.0,
            'total_voting_time_ms': 0
        }
        self.request_count = 0
        self.total_response_time_ms = 0

    def get_health_check(self) -> HealthCheckResponse:
        """
        Get system health status.

        Returns:
            Health check response
        """
        now = datetime.now().timestamp()
        uptime = now - self.start_time

        # Check component health
        components = {
            'swarm_coordinator': 'healthy' if self.swarm_coordinator else 'not_configured',
            'task_distributor': 'healthy' if self.task_distributor else 'not_configured',
            'verification_stack': 'healthy' if self.verification_stack else 'not_configured'
        }

        # Determine overall status
        unhealthy_count = sum(
            1 for status in components.values()
            if status not in ['healthy', 'not_configured']
        )

        if unhealthy_count == 0:
            overall_status = 'healthy'
        elif unhealthy_count < len(components):
            overall_status = 'degraded'
        else:
            overall_status = 'unhealthy'

        return HealthCheckResponse(
            status=overall_status,
            uptime_seconds=uptime,
            version=self.version,
            components=components,
            timestamp=now
        )

    def get_metrics(self) -> MetricsResponse:
        """
        Get system performance metrics.

        Returns:
            Metrics response
        """
        now = datetime.now().timestamp()

        # Calculate performance metrics
        avg_response_time = 0.0
        if self.request_count > 0:
            avg_response_time = self.total_response_time_ms / self.request_count

        # Calculate RPS (simplified)
        uptime = now - self.start_time
        rps = 0.0
        if uptime > 0:
            rps = self.request_count / uptime

        performance = PerformanceMetrics(
            requests_per_second=rps,
            avg_response_time_ms=avg_response_time,
            active_tasks=self._get_active_task_count(),
            active_agents=self._get_active_agent_count(),
            memory_usage_mb=self._get_memory_usage(),
            cpu_usage_percent=self._get_cpu_usage()
        )

        # Task metrics
        task_metrics = {
            'pending': 0,
            'in_progress': 0,
            'completed': 0,
            'failed': 0
        }

        # Agent metrics
        agent_metrics = {
            'total_spawned': 0,
            'currently_active': self._get_active_agent_count(),
            'idle': 0,
            'busy': 0
        }

        # Verification metrics
        verification_metrics = {
            'total_verifications': 0,
            'passed': 0,
            'failed': 0,
            'avg_quality_score': 0.0
        }

        return MetricsResponse(
            timestamp=now,
            performance=performance,
            task_metrics=task_metrics,
            agent_metrics=agent_metrics,
            verification_metrics=verification_metrics
        )

    def get_red_flags(
        self,
        limit: int = 100,
        severity_filter: Optional[int] = None,
        resolved_filter: Optional[bool] = None
    ) -> RedFlagsResponse:
        """
        Get red flag events.

        Args:
            limit: Maximum number of events to return
            severity_filter: Optional severity filter
            resolved_filter: Optional resolved status filter

        Returns:
            Red flags response
        """
        if limit <= 0:
            limit = 100

        # Filter events
        filtered_events = []
        for event in self.red_flags.values():
            if severity_filter is not None and event.severity != severity_filter:
                continue

            if resolved_filter is not None and event.resolved != resolved_filter:
                continue

            filtered_events.append(event)

        # Sort by timestamp (most recent first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)

        # Limit results
        recent_events = filtered_events[:limit]

        # Count by severity
        flags_by_severity: Dict[int, int] = {}
        active_flags = 0
        resolved_flags = 0

        for event in self.red_flags.values():
            severity = event.severity
            flags_by_severity[severity] = flags_by_severity.get(severity, 0) + 1

            if event.resolved:
                resolved_flags += 1
            else:
                active_flags += 1

        return RedFlagsResponse(
            total_flags=len(self.red_flags),
            active_flags=active_flags,
            resolved_flags=resolved_flags,
            flags_by_severity=flags_by_severity,
            recent_events=recent_events
        )

    def get_consensus_stats(self) -> ConsensusStats:
        """
        Get consensus voting statistics.

        Returns:
            Consensus statistics
        """
        avg_confidence = 0.0
        if self.consensus_data['total_votes'] > 0:
            avg_confidence = (
                self.consensus_data['total_confidence'] /
                self.consensus_data['total_votes']
            )

        avg_voting_time = 0.0
        if self.consensus_data['total_votes'] > 0:
            avg_voting_time = (
                self.consensus_data['total_voting_time_ms'] /
                self.consensus_data['total_votes']
            )

        return ConsensusStats(
            total_votes=self.consensus_data['total_votes'],
            consensus_achieved=self.consensus_data['consensus_achieved'],
            consensus_failed=self.consensus_data['consensus_failed'],
            avg_confidence=avg_confidence,
            avg_voting_time_ms=avg_voting_time,
            votes_by_task_type={}
        )

    def record_red_flag(
        self,
        task_id: str,
        agent_id: str,
        flag_type: str,
        severity: int,
        description: str
    ) -> str:
        """
        Record a red flag event.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            flag_type: Type of red flag
            severity: Severity level (1-5)
            description: Event description

        Returns:
            Event ID
        """
        import hashlib
        event_id = hashlib.sha256(
            f"{task_id}_{agent_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        event = RedFlagEvent(
            event_id=event_id,
            task_id=task_id,
            agent_id=agent_id,
            flag_type=flag_type,
            severity=severity,
            description=description,
            timestamp=datetime.now().timestamp(),
            resolved=False
        )

        self.red_flags[event_id] = event

        return event_id

    def resolve_red_flag(self, event_id: str) -> bool:
        """
        Mark a red flag as resolved.

        Args:
            event_id: Event identifier

        Returns:
            True if resolved
        """
        if event_id not in self.red_flags:
            return False

        self.red_flags[event_id].resolved = True
        return True

    def record_request(self, response_time_ms: int) -> None:
        """
        Record an API request for metrics.

        Args:
            response_time_ms: Request response time
        """
        self.request_count += 1
        self.total_response_time_ms += response_time_ms

    def record_consensus_vote(
        self,
        achieved: bool,
        confidence: float,
        voting_time_ms: int
    ) -> None:
        """
        Record a consensus vote for statistics.

        Args:
            achieved: Whether consensus was achieved
            confidence: Confidence score
            voting_time_ms: Voting time
        """
        self.consensus_data['total_votes'] += 1
        if achieved:
            self.consensus_data['consensus_achieved'] += 1
        else:
            self.consensus_data['consensus_failed'] += 1

        self.consensus_data['total_confidence'] += confidence
        self.consensus_data['total_voting_time_ms'] += voting_time_ms

    def _get_active_task_count(self) -> int:
        """Get count of active tasks."""
        # In production, would query task distributor
        return 0

    def _get_active_agent_count(self) -> int:
        """Get count of active agents."""
        # In production, would query pool manager
        return 0

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        # In production, would use psutil or similar
        return 0.0

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        # In production, would use psutil or similar
        return 0.0
