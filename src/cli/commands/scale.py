"""
Scale Command Module.

This module implements the 'scale' CLI command for managing agent pool
sizing and auto-scaling.

Command: zero-error scale <operation> [OPTIONS]

Operations:
    up: Scale up agent pool
    down: Scale down agent pool
    auto: Enable/configure auto-scaling
    status: Show current scaling status

Options:
    --agent-type: Agent type to scale (coder/verifier/tester/all)
    --count: Number of agents to add/remove
    --min: Minimum pool size (for auto-scaling)
    --max: Maximum pool size (for auto-scaling)
    --target-utilization: Target utilization percentage (for auto-scaling)
"""

from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from ..utils import (
    CLIResult,
    ConsoleFormatter,
    create_default_formatter
)


@dataclass
class ScaleOptions:
    """
    Options for scaling operations.

    Attributes:
        operation: Scaling operation (up/down/auto/status)
        agent_type: Agent type to scale
        count: Number of agents
        min_size: Minimum pool size
        max_size: Maximum pool size
        target_utilization: Target utilization percentage
    """
    operation: str
    agent_type: str = "all"
    count: int = 1
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    target_utilization: float = 0.8


@dataclass
class PoolStatus:
    """
    Agent pool status.

    Attributes:
        agent_type: Agent type
        current_size: Current pool size
        active_agents: Number of active agents
        utilization: Current utilization rate
        min_size: Minimum pool size (if auto-scaling)
        max_size: Maximum pool size (if auto-scaling)
        auto_scaling_enabled: Whether auto-scaling is enabled
    """
    agent_type: str
    current_size: int
    active_agents: int
    utilization: float
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    auto_scaling_enabled: bool = False


# Valid agent types
VALID_AGENT_TYPES = [
    "coder",
    "verifier",
    "tester",
    "reviewer",
    "documenter",
    "optimizer",
    "all"
]

# Valid operations
VALID_OPERATIONS = ["up", "down", "auto", "status"]


class ScaleCommand:
    """
    Scale command implementation.

    Manages agent pool sizing and auto-scaling configuration.
    """

    def __init__(self, formatter: Optional[ConsoleFormatter] = None):
        """
        Initialize command handler.

        Args:
            formatter: Console formatter for output
        """
        self.formatter = formatter or create_default_formatter()

    def execute(self, options: ScaleOptions) -> CLIResult:
        """
        Execute scale command.

        Args:
            options: Scale options

        Returns:
            CLIResult with operation status
        """
        # Validate operation
        if options.operation not in VALID_OPERATIONS:
            return CLIResult(
                success=False,
                message=f"Invalid operation '{options.operation}'. "
                        f"Valid: {', '.join(VALID_OPERATIONS)}",
                error_code=1
            )

        # Validate agent type
        if options.agent_type not in VALID_AGENT_TYPES:
            return CLIResult(
                success=False,
                message=f"Invalid agent type '{options.agent_type}'. "
                        f"Valid: {', '.join(VALID_AGENT_TYPES)}",
                error_code=2
            )

        # Route to appropriate handler
        if options.operation == "up":
            return self._scale_up(options)
        elif options.operation == "down":
            return self._scale_down(options)
        elif options.operation == "auto":
            return self._configure_autoscaling(options)
        elif options.operation == "status":
            return self._show_status(options)

        return CLIResult(
            success=False,
            message="Unknown operation",
            error_code=99
        )

    def _scale_up(self, options: ScaleOptions) -> CLIResult:
        """
        Scale up agent pool.

        Args:
            options: Scale options

        Returns:
            CLIResult with operation status
        """
        self.formatter.print_info(
            f"Scaling up {options.agent_type} agents by {options.count}..."
        )

        # In production, this would:
        # 1. Validate current pool status
        # 2. Check resource availability
        # 3. Spawn new agents
        # 4. Register agents with pool manager
        # 5. Verify agents are healthy

        result = self._simulate_scale_up(options)

        if result.success:
            self.formatter.print_success(
                f"Successfully scaled up {options.agent_type} pool"
            )
        else:
            self.formatter.print_error(
                f"Failed to scale up: {result.message}"
            )

        return result

    def _scale_down(self, options: ScaleOptions) -> CLIResult:
        """
        Scale down agent pool.

        Args:
            options: Scale options

        Returns:
            CLIResult with operation status
        """
        self.formatter.print_info(
            f"Scaling down {options.agent_type} agents by {options.count}..."
        )

        # In production, this would:
        # 1. Identify idle agents
        # 2. Wait for busy agents to complete tasks
        # 3. Gracefully shutdown agents
        # 4. Remove from pool manager
        # 5. Verify cleanup

        result = self._simulate_scale_down(options)

        if result.success:
            self.formatter.print_success(
                f"Successfully scaled down {options.agent_type} pool"
            )
        else:
            self.formatter.print_error(
                f"Failed to scale down: {result.message}"
            )

        return result

    def _configure_autoscaling(self, options: ScaleOptions) -> CLIResult:
        """
        Configure auto-scaling.

        Args:
            options: Scale options

        Returns:
            CLIResult with operation status
        """
        # Validate auto-scaling parameters
        if options.min_size is not None and options.max_size is not None:
            if options.min_size > options.max_size:
                return CLIResult(
                    success=False,
                    message="Minimum size cannot exceed maximum size",
                    error_code=10
                )

        if options.target_utilization <= 0.0 or options.target_utilization > 1.0:
            return CLIResult(
                success=False,
                message="Target utilization must be between 0.0 and 1.0",
                error_code=11
            )

        self.formatter.print_info(
            f"Configuring auto-scaling for {options.agent_type} agents..."
        )

        # In production, this would:
        # 1. Update pool manager configuration
        # 2. Enable auto-scaling monitor
        # 3. Set scaling policies
        # 4. Configure scaling cooldown periods

        result = self._simulate_configure_autoscaling(options)

        if result.success:
            self.formatter.print_success(
                "Auto-scaling configured successfully"
            )
            if options.min_size is not None and options.max_size is not None:
                self.formatter.print_info(
                    f"Pool will scale between {options.min_size} and "
                    f"{options.max_size} agents"
                )
            self.formatter.print_info(
                f"Target utilization: {options.target_utilization * 100:.0f}%"
            )
        else:
            self.formatter.print_error(
                f"Failed to configure auto-scaling: {result.message}"
            )

        return result

    def _show_status(self, options: ScaleOptions) -> CLIResult:
        """
        Show scaling status.

        Args:
            options: Scale options

        Returns:
            CLIResult with operation status
        """
        # Fetch current pool status
        pool_statuses = self._fetch_pool_status(options.agent_type)

        if not pool_statuses:
            return CLIResult(
                success=False,
                message="No pool status available",
                error_code=20
            )

        # Display status
        self._display_pool_status(pool_statuses)

        return CLIResult(
            success=True,
            message="Status retrieved",
            data={
                "pools": [
                    {
                        "agent_type": s.agent_type,
                        "current_size": s.current_size,
                        "utilization": s.utilization
                    }
                    for s in pool_statuses
                ]
            }
        )

    def _simulate_scale_up(self, options: ScaleOptions) -> CLIResult:
        """
        Simulate scale up operation.

        Args:
            options: Scale options

        Returns:
            CLIResult
        """
        # Simulate success
        return CLIResult(
            success=True,
            message=f"Scaled up {options.count} {options.agent_type} agents",
            data={
                "agent_type": options.agent_type,
                "added_count": options.count,
                "new_pool_size": 10 + options.count
            }
        )

    def _simulate_scale_down(self, options: ScaleOptions) -> CLIResult:
        """
        Simulate scale down operation.

        Args:
            options: Scale options

        Returns:
            CLIResult
        """
        # Check if we can scale down
        current_size = 10
        if current_size - options.count < 1:
            return CLIResult(
                success=False,
                message="Cannot scale below minimum pool size (1)",
                error_code=30
            )

        # Simulate success
        return CLIResult(
            success=True,
            message=f"Scaled down {options.count} {options.agent_type} agents",
            data={
                "agent_type": options.agent_type,
                "removed_count": options.count,
                "new_pool_size": current_size - options.count
            }
        )

    def _simulate_configure_autoscaling(
        self,
        options: ScaleOptions
    ) -> CLIResult:
        """
        Simulate auto-scaling configuration.

        Args:
            options: Scale options

        Returns:
            CLIResult
        """
        return CLIResult(
            success=True,
            message="Auto-scaling configured",
            data={
                "agent_type": options.agent_type,
                "min_size": options.min_size,
                "max_size": options.max_size,
                "target_utilization": options.target_utilization
            }
        )

    def _fetch_pool_status(
        self,
        agent_type: str
    ) -> list[PoolStatus]:
        """
        Fetch pool status information.

        Args:
            agent_type: Agent type or 'all'

        Returns:
            List of PoolStatus
        """
        # In production, would query pool manager
        # For now, return simulated data

        if agent_type == "all":
            return [
                PoolStatus(
                    agent_type="coder",
                    current_size=10,
                    active_agents=7,
                    utilization=0.7,
                    min_size=5,
                    max_size=20,
                    auto_scaling_enabled=True
                ),
                PoolStatus(
                    agent_type="verifier",
                    current_size=6,
                    active_agents=4,
                    utilization=0.67,
                    min_size=3,
                    max_size=15,
                    auto_scaling_enabled=True
                ),
                PoolStatus(
                    agent_type="tester",
                    current_size=4,
                    active_agents=2,
                    utilization=0.5,
                    min_size=2,
                    max_size=10,
                    auto_scaling_enabled=False
                )
            ]
        else:
            return [
                PoolStatus(
                    agent_type=agent_type,
                    current_size=10,
                    active_agents=7,
                    utilization=0.7,
                    min_size=5,
                    max_size=20,
                    auto_scaling_enabled=True
                )
            ]

    def _display_pool_status(self, statuses: list[PoolStatus]) -> None:
        """
        Display pool status information.

        Args:
            statuses: List of pool statuses
        """
        self.formatter.print_panel(
            f"Agent Pool Status\n"
            f"Timestamp: {datetime.now().isoformat()}",
            title="Scaling Status"
        )

        # Create table
        columns = [
            "Agent Type",
            "Size",
            "Active",
            "Utilization",
            "Auto-Scale",
            "Min/Max"
        ]
        rows = []

        for status in statuses:
            utilization_pct = f"{status.utilization * 100:.0f}%"
            auto_scale = "Yes" if status.auto_scaling_enabled else "No"

            if status.auto_scaling_enabled and status.min_size and status.max_size:
                min_max = f"{status.min_size}/{status.max_size}"
            else:
                min_max = "-"

            rows.append([
                status.agent_type,
                str(status.current_size),
                str(status.active_agents),
                utilization_pct,
                auto_scale,
                min_max
            ])

        self.formatter.print_table("Pool Details", columns, rows)

        # Show recommendations
        for status in statuses:
            if status.utilization > 0.9:
                self.formatter.print_warning(
                    f"{status.agent_type} pool is highly utilized "
                    f"({status.utilization * 100:.0f}%) - consider scaling up"
                )
            elif status.utilization < 0.3:
                self.formatter.print_info(
                    f"{status.agent_type} pool has low utilization "
                    f"({status.utilization * 100:.0f}%) - consider scaling down"
                )


def run_scale(
    operation: str,
    agent_type: str = "all",
    count: int = 1,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    target_utilization: float = 0.8,
    formatter: Optional[ConsoleFormatter] = None
) -> CLIResult:
    """
    Run scale command with given options.

    Args:
        operation: Scaling operation
        agent_type: Agent type to scale
        count: Number of agents
        min_size: Minimum pool size
        max_size: Maximum pool size
        target_utilization: Target utilization
        formatter: Console formatter

    Returns:
        CLIResult with operation status
    """
    options = ScaleOptions(
        operation=operation,
        agent_type=agent_type,
        count=count,
        min_size=min_size,
        max_size=max_size,
        target_utilization=target_utilization
    )

    command = ScaleCommand(formatter)
    return command.execute(options)
