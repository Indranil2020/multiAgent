"""
CLI Module.

This module provides the command-line interface for the zero-error system.

Components:
- main: Main CLI application and entry point
- utils: Shared utilities for CLI operations
- commands: Individual command implementations

Commands:
- init: Initialize new projects
- run: Execute task specifications
- verify: Run code verification
- monitor: Monitor system status
- scale: Manage agent scaling

Usage:
    from cli import main
    main.run_cli()

Or use the command-line interface:
    zero-error <command> [options]
"""

from .main import (
    CLIApplication,
    main,
    run_cli
)

from .utils import (
    CLIConfig,
    CLIResult,
    ConsoleFormatter,
    ConfigLoader,
    PathValidator,
    OutputSerializer,
    create_default_formatter,
    load_or_create_config
)

from .commands.init import (
    InitCommand,
    InitOptions,
    run_init
)

from .commands.run import (
    RunCommand,
    RunOptions,
    TaskSpecification,
    ExecutionResult,
    run_task
)

from .commands.verify import (
    VerifyCommand,
    VerifyOptions,
    LayerResult,
    VerificationReport,
    run_verify
)

from .commands.monitor import (
    MonitorCommand,
    MonitorOptions,
    SystemMetrics,
    AgentStatus,
    TaskStatus,
    HealthStatus,
    run_monitor
)

from .commands.scale import (
    ScaleCommand,
    ScaleOptions,
    PoolStatus,
    run_scale
)


__all__ = [
    # Main application
    'CLIApplication',
    'main',
    'run_cli',

    # Utils
    'CLIConfig',
    'CLIResult',
    'ConsoleFormatter',
    'ConfigLoader',
    'PathValidator',
    'OutputSerializer',
    'create_default_formatter',
    'load_or_create_config',

    # Init command
    'InitCommand',
    'InitOptions',
    'run_init',

    # Run command
    'RunCommand',
    'RunOptions',
    'TaskSpecification',
    'ExecutionResult',
    'run_task',

    # Verify command
    'VerifyCommand',
    'VerifyOptions',
    'LayerResult',
    'VerificationReport',
    'run_verify',

    # Monitor command
    'MonitorCommand',
    'MonitorOptions',
    'SystemMetrics',
    'AgentStatus',
    'TaskStatus',
    'HealthStatus',
    'run_monitor',

    # Scale command
    'ScaleCommand',
    'ScaleOptions',
    'PoolStatus',
    'run_scale',
]


__version__ = "1.0.0"
__author__ = "Zero-Error System"
__description__ = "CLI for the Ultimate Zero-Error Software Development System"
