"""
CLI Main Entry Point.

This module provides the main CLI application using Typer framework.
It registers all commands and provides the primary user interface.

Usage:
    zero-error init <path> [options]
    zero-error run <spec-file> [options]
    zero-error verify <code-path> [options]
    zero-error monitor [options]
    zero-error scale <operation> [options]
"""

from pathlib import Path
from typing import Optional, List
import sys

# Note: In production, would use typer/click
# For now, provide a simplified interface that demonstrates the structure

from .utils import (
    CLIConfig,
    ConsoleFormatter,
    ConfigLoader,
    create_default_formatter,
    load_or_create_config
)

from .commands.init import run_init
from .commands.run import run_task
from .commands.verify import run_verify
from .commands.monitor import run_monitor
from .commands.scale import run_scale


class CLIApplication:
    """
    Main CLI application.

    Coordinates command routing and global configuration.
    """

    def __init__(self):
        """Initialize CLI application."""
        self.formatter = create_default_formatter()
        self.config: Optional[CLIConfig] = None

    def run(self, args: List[str]) -> int:
        """
        Run CLI application.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        if len(args) < 1:
            self._show_help()
            return 0

        command = args[0]

        # Global options
        config_path = self._extract_option(args, "--config")
        verbose = "--verbose" in args or "-v" in args

        # Load configuration
        if config_path:
            self.config = load_or_create_config(Path(config_path))
        else:
            self.config = load_or_create_config()

        if verbose:
            self.config.verbose = True

        # Route to command handler
        if command == "init":
            return self._handle_init(args[1:])
        elif command == "run":
            return self._handle_run(args[1:])
        elif command == "verify":
            return self._handle_verify(args[1:])
        elif command == "monitor":
            return self._handle_monitor(args[1:])
        elif command == "scale":
            return self._handle_scale(args[1:])
        elif command in ["help", "--help", "-h"]:
            self._show_help()
            return 0
        elif command in ["version", "--version", "-V"]:
            self._show_version()
            return 0
        else:
            self.formatter.print_error(f"Unknown command: {command}")
            self.formatter.print_info("Run 'zero-error help' for usage information")
            return 1

    def _handle_init(self, args: List[str]) -> int:
        """
        Handle init command.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        if len(args) < 1:
            self.formatter.print_error("Missing project path")
            self.formatter.print_info("Usage: zero-error init <path> [options]")
            return 1

        project_path = Path(args[0])
        template = self._extract_option(args, "--template") or "api"
        with_examples = "--no-examples" not in args

        result = run_init(
            project_path=project_path,
            template=template,
            with_examples=with_examples,
            formatter=self.formatter
        )

        return 0 if result.success else (result.error_code or 1)

    def _handle_run(self, args: List[str]) -> int:
        """
        Handle run command.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        if len(args) < 1:
            self.formatter.print_error("Missing spec file")
            self.formatter.print_info("Usage: zero-error run <spec-file> [options]")
            return 1

        spec_file = Path(args[0])

        # Parse options
        num_agents = int(self._extract_option(args, "--agents") or "5")
        layers_str = self._extract_option(args, "--layers")
        layers = layers_str.split(",") if layers_str else None
        output_dir_str = self._extract_option(args, "--output-dir")
        output_dir = Path(output_dir_str) if output_dir_str else None
        config_file_str = self._extract_option(args, "--config")
        config_file = Path(config_file_str) if config_file_str else None
        verbose = "--verbose" in args or "-v" in args

        result = run_task(
            spec_file=spec_file,
            num_agents=num_agents,
            enabled_layers=layers,
            output_dir=output_dir,
            config_file=config_file,
            verbose=verbose,
            formatter=self.formatter
        )

        return 0 if result.success else (result.error_code or 1)

    def _handle_verify(self, args: List[str]) -> int:
        """
        Handle verify command.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        if len(args) < 1:
            self.formatter.print_error("Missing code path")
            self.formatter.print_info("Usage: zero-error verify <code-path> [options]")
            return 1

        code_path = Path(args[0])

        # Parse options
        layers_str = self._extract_option(args, "--layers")
        layers = layers_str.split(",") if layers_str else None
        strict = "--strict" in args
        report_format = self._extract_option(args, "--format") or "text"
        output_str = self._extract_option(args, "--output")
        output_path = Path(output_str) if output_str else None

        result = run_verify(
            code_path=code_path,
            layers=layers,
            strict=strict,
            report_format=report_format,
            output_path=output_path,
            formatter=self.formatter
        )

        return 0 if result.success else (result.error_code or 1)

    def _handle_monitor(self, args: List[str]) -> int:
        """
        Handle monitor command.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        follow = "--follow" in args or "-f" in args
        interval = int(self._extract_option(args, "--interval") or "2")
        format_type = self._extract_option(args, "--format") or "text"
        api_url = self._extract_option(args, "--api-url")

        result = run_monitor(
            follow=follow,
            interval=interval,
            format_type=format_type,
            api_url=api_url,
            formatter=self.formatter
        )

        return 0 if result.success else (result.error_code or 1)

    def _handle_scale(self, args: List[str]) -> int:
        """
        Handle scale command.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        if len(args) < 1:
            self.formatter.print_error("Missing operation")
            self.formatter.print_info("Usage: zero-error scale <operation> [options]")
            self.formatter.print_info("Operations: up, down, auto, status")
            return 1

        operation = args[0]
        agent_type = self._extract_option(args, "--agent-type") or "all"
        count = int(self._extract_option(args, "--count") or "1")

        min_size_str = self._extract_option(args, "--min")
        min_size = int(min_size_str) if min_size_str else None

        max_size_str = self._extract_option(args, "--max")
        max_size = int(max_size_str) if max_size_str else None

        target_util = float(
            self._extract_option(args, "--target-utilization") or "0.8"
        )

        result = run_scale(
            operation=operation,
            agent_type=agent_type,
            count=count,
            min_size=min_size,
            max_size=max_size,
            target_utilization=target_util,
            formatter=self.formatter
        )

        return 0 if result.success else (result.error_code or 1)

    def _extract_option(self, args: List[str], option: str) -> Optional[str]:
        """
        Extract option value from arguments.

        Args:
            args: Argument list
            option: Option name (e.g., '--config')

        Returns:
            Option value or None
        """
        # Check if option exists in args
        if option not in args:
            return None
        
        idx = args.index(option)
        if idx + 1 < len(args):
            return args[idx + 1]
        return None

    def _show_help(self) -> None:
        """Show help message."""
        help_text = """
Zero-Error Software Development System CLI

USAGE:
    zero-error <command> [options]

COMMANDS:
    init        Initialize a new zero-error project
    run         Run the zero-error system on a task specification
    verify      Verify code through the verification stack
    monitor     Monitor system status and metrics
    scale       Manage agent pool scaling
    help        Show this help message
    version     Show version information

GLOBAL OPTIONS:
    --config <path>    Configuration file path
    --verbose, -v      Enable verbose output
    --help, -h         Show help for a command

EXAMPLES:
    # Initialize a new API project
    zero-error init my-project --template api --with-examples

    # Run a task specification
    zero-error run specs/tasks/my_task.json --agents 10

    # Verify code
    zero-error verify src/module.py --layers syntax,type_checking,security

    # Monitor system (continuously)
    zero-error monitor --follow --interval 5

    # Scale agent pool
    zero-error scale up --agent-type coder --count 5
    zero-error scale auto --min 5 --max 20 --target-utilization 0.8

For more information, see the documentation.
"""
        print(help_text)

    def _show_version(self) -> None:
        """Show version information."""
        version_text = """
Zero-Error System CLI
Version: 1.0.0
Python: 3.10+

Architecture: Ultimate Zero-Error Software Development
Features:
  - 7-layer hierarchical decomposition
  - 8-layer verification stack
  - MAKER-style voting
  - Agent swarm coordination
  - Distributed execution
"""
        print(version_text)


def main() -> int:
    """
    Main entry point for CLI.

    Returns:
        Exit code
    """
    app = CLIApplication()
    return app.run(sys.argv[1:])


def run_cli() -> None:
    """
    Run CLI and exit with appropriate code.

    This is the entry point used by setup.py console_scripts.
    """
    sys.exit(main())


if __name__ == "__main__":
    run_cli()
