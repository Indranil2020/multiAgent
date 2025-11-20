"""
Monitor Command Module.

This module implements the 'monitor' CLI command for displaying real-time
system status, metrics, and health information.

Command: zero-error monitor [OPTIONS]

Options:
    --follow: Continuously follow updates (like tail -f)
    --interval: Update interval in seconds
    --format: Output format (text/json)
    --api-url: API server URL
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import time

from ..utils import (
    CLIResult,
    ConsoleFormatter,
    CLIConfig,
    create_default_formatter
)


@dataclass
class MonitorOptions:
    """
    Options for system monitoring.

    Attributes:
        follow: Continuously follow updates
        interval: Update interval in seconds
        format_type: Output format
        api_url: API server URL
    """
    follow: bool = False
    interval: int = 2
    format_type: str = "text"
    api_url: Optional[str] = None


@dataclass
class SystemMetrics:
    """
    System metrics snapshot.

    Attributes:
        timestamp: Metrics timestamp
        active_agents: Number of active agents
        idle_agents: Number of idle agents
        busy_agents: Number of busy agents
        active_tasks: Number of active tasks
        completed_tasks: Number of completed tasks
        failed_tasks: Number of failed tasks
        avg_task_time_ms: Average task completion time
        verification_pass_rate: Verification pass rate
        uptime_seconds: System uptime
    """
    timestamp: str
    active_agents: int = 0
    idle_agents: int = 0
    busy_agents: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_time_ms: float = 0.0
    verification_pass_rate: float = 0.0
    uptime_seconds: float = 0.0


@dataclass
class AgentStatus:
    """
    Agent status information.

    Attributes:
        agent_id: Agent identifier
        agent_type: Agent type (coder, verifier, etc.)
        status: Agent status (idle, busy, failed)
        current_task: Current task ID if busy
        tasks_completed: Number of tasks completed
        avg_quality_score: Average quality score
    """
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    tasks_completed: int = 0
    avg_quality_score: float = 0.0


@dataclass
class TaskStatus:
    """
    Task status information.

    Attributes:
        task_id: Task identifier
        name: Task name
        status: Task status (pending, in_progress, completed, failed)
        progress: Progress percentage
        agents_assigned: Number of agents assigned
        started_at: Start timestamp
        estimated_completion: Estimated completion time
    """
    task_id: str
    name: str
    status: str
    progress: int = 0
    agents_assigned: int = 0
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None


@dataclass
class HealthStatus:
    """
    System health status.

    Attributes:
        status: Overall health status (healthy, degraded, unhealthy)
        components: Component health status
        issues: List of health issues
        last_check: Last health check timestamp
    """
    status: str
    components: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    last_check: Optional[str] = None


class MonitorCommand:
    """
    Monitor command implementation.

    Displays real-time system metrics, agent status, task progress,
    and health information.
    """

    def __init__(self, formatter: Optional[ConsoleFormatter] = None):
        """
        Initialize command handler.

        Args:
            formatter: Console formatter for output
        """
        self.formatter = formatter or create_default_formatter()

    def execute(self, options: MonitorOptions) -> CLIResult:
        """
        Execute monitor command.

        Args:
            options: Monitor options

        Returns:
            CLIResult with operation status
        """
        if options.follow:
            # Continuous monitoring mode
            return self._follow_mode(options)
        else:
            # Single snapshot mode
            return self._snapshot_mode(options)

    def _snapshot_mode(self, options: MonitorOptions) -> CLIResult:
        """
        Display a single metrics snapshot.

        Args:
            options: Monitor options

        Returns:
            CLIResult with operation status
        """
        # Fetch current metrics
        metrics = self._fetch_metrics(options.api_url)
        health = self._fetch_health(options.api_url)
        agents = self._fetch_agent_status(options.api_url)
        tasks = self._fetch_task_status(options.api_url)

        # Display results
        if options.format_type == "json":
            self._display_json(metrics, health, agents, tasks)
        else:
            self._display_text(metrics, health, agents, tasks)

        return CLIResult(
            success=True,
            message="Metrics retrieved",
            data={
                "timestamp": metrics.timestamp,
                "active_agents": metrics.active_agents,
                "active_tasks": metrics.active_tasks
            }
        )

    def _follow_mode(self, options: MonitorOptions) -> CLIResult:
        """
        Continuous monitoring mode.

        Args:
            options: Monitor options

        Returns:
            CLIResult with operation status
        """
        self.formatter.print_info(
            f"Monitoring system (updating every {options.interval}s)..."
        )
        self.formatter.print_info("Press Ctrl+C to stop")
        print()

        try:
            while True:
                # Fetch metrics
                metrics = self._fetch_metrics(options.api_url)
                health = self._fetch_health(options.api_url)
                agents = self._fetch_agent_status(options.api_url)
                tasks = self._fetch_task_status(options.api_url)

                # Clear screen (simplified - in production would use proper terminal control)
                print("\033[2J\033[H", end="")

                # Display
                if options.format_type == "json":
                    self._display_json(metrics, health, agents, tasks)
                else:
                    self._display_text(metrics, health, agents, tasks)

                # Wait for next update
                time.sleep(options.interval)

        except KeyboardInterrupt:
            print("\n")
            self.formatter.print_info("Monitoring stopped")
            return CLIResult(success=True, message="Monitoring stopped")

    def _fetch_metrics(self, api_url: Optional[str]) -> SystemMetrics:
        """
        Fetch system metrics.

        Args:
            api_url: API server URL

        Returns:
            SystemMetrics snapshot
        """
        # In production, this would make HTTP request to API
        # For now, return simulated data

        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            active_agents=8,
            idle_agents=3,
            busy_agents=5,
            active_tasks=3,
            completed_tasks=45,
            failed_tasks=2,
            avg_task_time_ms=1250.5,
            verification_pass_rate=0.96,
            uptime_seconds=3600.0
        )

    def _fetch_health(self, api_url: Optional[str]) -> HealthStatus:
        """
        Fetch health status.

        Args:
            api_url: API server URL

        Returns:
            HealthStatus
        """
        # In production, would query API
        return HealthStatus(
            status="healthy",
            components={
                "swarm_coordinator": "healthy",
                "task_distributor": "healthy",
                "verification_stack": "healthy",
                "voting_engine": "healthy"
            },
            issues=[],
            last_check=datetime.now().isoformat()
        )

    def _fetch_agent_status(
        self,
        api_url: Optional[str]
    ) -> List[AgentStatus]:
        """
        Fetch agent status list.

        Args:
            api_url: API server URL

        Returns:
            List of AgentStatus
        """
        # In production, would query API
        return [
            AgentStatus(
                agent_id="agent_001",
                agent_type="coder",
                status="busy",
                current_task="task_123",
                tasks_completed=10,
                avg_quality_score=0.95
            ),
            AgentStatus(
                agent_id="agent_002",
                agent_type="verifier",
                status="busy",
                current_task="task_124",
                tasks_completed=15,
                avg_quality_score=0.92
            ),
            AgentStatus(
                agent_id="agent_003",
                agent_type="coder",
                status="idle",
                tasks_completed=8,
                avg_quality_score=0.89
            )
        ]

    def _fetch_task_status(
        self,
        api_url: Optional[str]
    ) -> List[TaskStatus]:
        """
        Fetch task status list.

        Args:
            api_url: API server URL

        Returns:
            List of TaskStatus
        """
        # In production, would query API
        return [
            TaskStatus(
                task_id="task_123",
                name="Implement user authentication",
                status="in_progress",
                progress=65,
                agents_assigned=3,
                started_at=datetime.now().isoformat()
            ),
            TaskStatus(
                task_id="task_124",
                name="Add database caching",
                status="in_progress",
                progress=30,
                agents_assigned=2,
                started_at=datetime.now().isoformat()
            )
        ]

    def _display_text(
        self,
        metrics: SystemMetrics,
        health: HealthStatus,
        agents: List[AgentStatus],
        tasks: List[TaskStatus]
    ) -> None:
        """
        Display metrics in text format.

        Args:
            metrics: System metrics
            health: Health status
            agents: Agent status list
            tasks: Task status list
        """
        # System overview
        uptime_hours = metrics.uptime_seconds / 3600
        self.formatter.print_panel(
            f"Timestamp: {metrics.timestamp}\n"
            f"Health: {health.status.upper()}\n"
            f"Uptime: {uptime_hours:.2f} hours\n"
            f"Active Agents: {metrics.active_agents} "
            f"(Idle: {metrics.idle_agents}, Busy: {metrics.busy_agents})\n"
            f"Active Tasks: {metrics.active_tasks}\n"
            f"Completed Tasks: {metrics.completed_tasks}\n"
            f"Failed Tasks: {metrics.failed_tasks}\n"
            f"Avg Task Time: {metrics.avg_task_time_ms:.2f}ms\n"
            f"Verification Pass Rate: {metrics.verification_pass_rate * 100:.1f}%",
            title="System Overview"
        )

        # Component health
        if health.components:
            columns = ["Component", "Status"]
            rows = [
                [name, status.upper()]
                for name, status in health.components.items()
            ]
            self.formatter.print_table("Component Health", columns, rows)

        # Active agents
        if agents:
            agent_columns = ["Agent ID", "Type", "Status", "Tasks Done", "Quality"]
            agent_rows = []
            for agent in agents[:10]:  # Show top 10
                status_symbol = "[FAIL]" if agent.status == "failed" else (
                    "[IDLE]" if agent.status == "idle" else "[BUSY]"
                )
                agent_rows.append([
                    agent.agent_id,
                    agent.agent_type,
                    f"{status_symbol} {agent.status}",
                    str(agent.tasks_completed),
                    f"{agent.avg_quality_score:.2f}"
                ])
            self.formatter.print_table("Active Agents", agent_columns, agent_rows)

        # Active tasks
        if tasks:
            task_columns = ["Task ID", "Name", "Status", "Progress", "Agents"]
            task_rows = []
            for task in tasks:
                progress_bar = self._create_progress_bar(task.progress)
                task_rows.append([
                    task.task_id,
                    task.name[:30],
                    task.status,
                    f"{progress_bar} {task.progress}%",
                    str(task.agents_assigned)
                ])
            self.formatter.print_table("Active Tasks", task_columns, task_rows)

        # Health issues
        if health.issues:
            self.formatter.print_warning("Health Issues:")
            for issue in health.issues:
                print(f"  - {issue}")

    def _display_json(
        self,
        metrics: SystemMetrics,
        health: HealthStatus,
        agents: List[AgentStatus],
        tasks: List[TaskStatus]
    ) -> None:
        """
        Display metrics in JSON format.

        Args:
            metrics: System metrics
            health: Health status
            agents: Agent status list
            tasks: Task status list
        """
        data = {
            "metrics": {
                "timestamp": metrics.timestamp,
                "active_agents": metrics.active_agents,
                "idle_agents": metrics.idle_agents,
                "busy_agents": metrics.busy_agents,
                "active_tasks": metrics.active_tasks,
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "avg_task_time_ms": metrics.avg_task_time_ms,
                "verification_pass_rate": metrics.verification_pass_rate,
                "uptime_seconds": metrics.uptime_seconds
            },
            "health": {
                "status": health.status,
                "components": health.components,
                "issues": health.issues,
                "last_check": health.last_check
            },
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "status": a.status,
                    "current_task": a.current_task,
                    "tasks_completed": a.tasks_completed,
                    "avg_quality_score": a.avg_quality_score
                }
                for a in agents
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "status": t.status,
                    "progress": t.progress,
                    "agents_assigned": t.agents_assigned,
                    "started_at": t.started_at,
                    "estimated_completion": t.estimated_completion
                }
                for t in tasks
            ]
        }

        self.formatter.print_json(data)

    def _create_progress_bar(self, progress: int, width: int = 10) -> str:
        """
        Create text progress bar.

        Args:
            progress: Progress percentage (0-100)
            width: Bar width in characters

        Returns:
            Progress bar string
        """
        filled = int((progress / 100) * width)
        empty = width - filled
        return f"[{'=' * filled}{'-' * empty}]"


def run_monitor(
    follow: bool = False,
    interval: int = 2,
    format_type: str = "text",
    api_url: Optional[str] = None,
    formatter: Optional[ConsoleFormatter] = None
) -> CLIResult:
    """
    Run monitor command with given options.

    Args:
        follow: Continuously follow updates
        interval: Update interval in seconds
        format_type: Output format
        api_url: API server URL
        formatter: Console formatter

    Returns:
        CLIResult with operation status
    """
    options = MonitorOptions(
        follow=follow,
        interval=interval,
        format_type=format_type,
        api_url=api_url
    )

    command = MonitorCommand(formatter)
    return command.execute(options)
