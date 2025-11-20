"""
CLI Utilities Module.

This module provides shared utility functions for the CLI including:
- Configuration loading and validation
- Rich console formatting
- Path validation and resolution
- Output serialization
- Error message formatting

All utilities follow the zero-error philosophy with explicit validation
and no exception-based control flow.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
import json
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class CLIConfig:
    """
    CLI configuration.

    Attributes:
        config_path: Path to configuration file
        log_level: Logging level (debug, info, warning, error)
        verbose: Enable verbose output
        api_host: API server host
        api_port: API server port
    """
    config_path: Optional[Path] = None
    log_level: str = "info"
    verbose: bool = False
    api_host: str = "localhost"
    api_port: int = 8000


@dataclass
class CLIResult:
    """
    Result of a CLI operation.

    Attributes:
        success: Whether operation succeeded
        message: Human-readable message
        data: Optional result data
        error_code: Optional error code for failures
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[int] = None


class ConsoleFormatter:
    """
    Console output formatter using Rich library.

    Provides formatted output for tables, progress bars, panels, and syntax
    highlighting. Falls back to plain text if Rich is not available.
    """

    def __init__(self, use_rich: bool = True):
        """
        Initialize console formatter.

        Args:
            use_rich: Whether to use Rich formatting (if available)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

    def print_success(self, message: str) -> None:
        """
        Print success message.

        Args:
            message: Success message to display
        """
        if self.use_rich and self.console:
            self.console.print(f"[green][OK][/green] {message}")
        else:
            print(f"[OK] {message}")

    def print_error(self, message: str) -> None:
        """
        Print error message.

        Args:
            message: Error message to display
        """
        if self.use_rich and self.console:
            self.console.print(f"[red][ERROR][/red] {message}", file=sys.stderr)
        else:
            print(f"[ERROR] {message}", file=sys.stderr)

    def print_warning(self, message: str) -> None:
        """
        Print warning message.

        Args:
            message: Warning message to display
        """
        if self.use_rich and self.console:
            self.console.print(f"[yellow][WARNING][/yellow] {message}")
        else:
            print(f"[WARNING] {message}")

    def print_info(self, message: str) -> None:
        """
        Print info message.

        Args:
            message: Info message to display
        """
        if self.use_rich and self.console:
            self.console.print(f"[blue][INFO][/blue] {message}")
        else:
            print(f"[INFO] {message}")

    def print_table(
        self,
        title: str,
        columns: List[str],
        rows: List[List[str]]
    ) -> None:
        """
        Print formatted table.

        Args:
            title: Table title
            columns: Column headers
            rows: Table rows
        """
        if self.use_rich and self.console:
            table = Table(title=title)
            for column in columns:
                table.add_column(column)
            for row in rows:
                table.add_row(*row)
            self.console.print(table)
        else:
            # Fallback: plain text table
            print(f"\n{title}")
            print("-" * len(title))
            print(" | ".join(columns))
            print("-" * (sum(len(c) for c in columns) + 3 * (len(columns) - 1)))
            for row in rows:
                print(" | ".join(row))
            print()

    def print_panel(self, content: str, title: Optional[str] = None) -> None:
        """
        Print content in a panel.

        Args:
            content: Panel content
            title: Optional panel title
        """
        if self.use_rich and self.console:
            panel = Panel(content, title=title)
            self.console.print(panel)
        else:
            if title:
                print(f"\n=== {title} ===")
            print(content)
            print()

    def print_json(self, data: Dict[str, Any], indent: int = 2) -> None:
        """
        Print JSON data with syntax highlighting.

        Args:
            data: Data to print as JSON
            indent: Indentation level
        """
        json_str = json.dumps(data, indent=indent)

        if self.use_rich and self.console:
            syntax = Syntax(json_str, "json", theme="monokai")
            self.console.print(syntax)
        else:
            print(json_str)


class ConfigLoader:
    """
    Configuration file loader and validator.

    Loads CLI configuration from JSON or Python files with validation.
    """

    @staticmethod
    def load_config(config_path: Path) -> Optional[CLIConfig]:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            CLIConfig if successful, None if failed
        """
        if not config_path.exists():
            return None

        if not config_path.is_file():
            return None

        # Read file content
        try:
            content = config_path.read_text()
        except (OSError, UnicodeDecodeError):
            return None

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        # Validate and create config
        return ConfigLoader._create_config_from_dict(data)

    @staticmethod
    def _create_config_from_dict(data: Dict[str, Any]) -> Optional[CLIConfig]:
        """
        Create CLIConfig from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            CLIConfig if valid, None if invalid
        """
        if not isinstance(data, dict):
            return None

        # Extract and validate fields
        log_level = data.get("log_level", "info")
        if log_level not in ["debug", "info", "warning", "error"]:
            return None

        verbose = data.get("verbose", False)
        if not isinstance(verbose, bool):
            return None

        api_host = data.get("api_host", "localhost")
        if not isinstance(api_host, str) or not api_host:
            return None

        api_port = data.get("api_port", 8000)
        if not isinstance(api_port, int) or api_port <= 0 or api_port > 65535:
            return None

        return CLIConfig(
            log_level=log_level,
            verbose=verbose,
            api_host=api_host,
            api_port=api_port
        )

    @staticmethod
    def save_config(config: CLIConfig, config_path: Path) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_path: Path to save configuration

        Returns:
            True if successful, False otherwise
        """
        # Create parent directory if needed
        if not config_path.parent.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                return False

        # Convert to dictionary
        data = {
            "log_level": config.log_level,
            "verbose": config.verbose,
            "api_host": config.api_host,
            "api_port": config.api_port
        }

        # Write to file
        try:
            config_path.write_text(json.dumps(data, indent=2))
            return True
        except (OSError, TypeError):
            return False


class PathValidator:
    """
    Path validation and resolution utilities.

    Provides safe path operations with explicit validation.
    """

    @staticmethod
    def validate_file_path(path: Union[str, Path]) -> Optional[Path]:
        """
        Validate that path exists and is a file.

        Args:
            path: Path to validate

        Returns:
            Resolved Path if valid file, None otherwise
        """
        try:
            resolved_path = Path(path).resolve()
        except (OSError, RuntimeError):
            return None

        if not resolved_path.exists():
            return None

        if not resolved_path.is_file():
            return None

        return resolved_path

    @staticmethod
    def validate_directory_path(path: Union[str, Path]) -> Optional[Path]:
        """
        Validate that path exists and is a directory.

        Args:
            path: Path to validate

        Returns:
            Resolved Path if valid directory, None otherwise
        """
        try:
            resolved_path = Path(path).resolve()
        except (OSError, RuntimeError):
            return None

        if not resolved_path.exists():
            return None

        if not resolved_path.is_dir():
            return None

        return resolved_path

    @staticmethod
    def validate_creatable_path(path: Union[str, Path]) -> Optional[Path]:
        """
        Validate that path can be created.

        Args:
            path: Path to validate

        Returns:
            Resolved Path if creatable, None otherwise
        """
        try:
            resolved_path = Path(path).resolve()
        except (OSError, RuntimeError):
            return None

        # Check if parent directory exists or can be created
        parent = resolved_path.parent
        if not parent.exists():
            # Check if we can traverse to an existing parent
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None

        return resolved_path

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Optional[Path]:
        """
        Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Resolved Path if successful, None otherwise
        """
        try:
            resolved_path = Path(path).resolve()
        except (OSError, RuntimeError):
            return None

        if resolved_path.exists() and not resolved_path.is_dir():
            return None

        if not resolved_path.exists():
            try:
                resolved_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None

        return resolved_path


class OutputSerializer:
    """
    Output serialization utilities.

    Handles serialization of data to various formats (JSON, plain text).
    """

    @staticmethod
    def to_json(data: Any, indent: int = 2) -> Optional[str]:
        """
        Serialize data to JSON string.

        Args:
            data: Data to serialize
            indent: Indentation level

        Returns:
            JSON string if successful, None otherwise
        """
        try:
            return json.dumps(data, indent=indent)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def from_json(json_str: str) -> Optional[Any]:
        """
        Deserialize JSON string to data.

        Args:
            json_str: JSON string

        Returns:
            Deserialized data if successful, None otherwise
        """
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def format_result(result: CLIResult, format_type: str = "text") -> Optional[str]:
        """
        Format CLI result for output.

        Args:
            result: CLI result to format
            format_type: Output format (text or json)

        Returns:
            Formatted string if successful, None otherwise
        """
        if format_type == "json":
            data = {
                "success": result.success,
                "message": result.message
            }
            if result.data is not None:
                data["data"] = result.data
            if result.error_code is not None:
                data["error_code"] = result.error_code

            return OutputSerializer.to_json(data)

        elif format_type == "text":
            status = "SUCCESS" if result.success else "FAILED"
            output = f"[{status}] {result.message}"

            if result.error_code is not None:
                output += f" (Error code: {result.error_code})"

            return output

        return None


def create_default_formatter() -> ConsoleFormatter:
    """
    Create default console formatter.

    Returns:
        ConsoleFormatter instance
    """
    return ConsoleFormatter(use_rich=RICH_AVAILABLE)


def load_or_create_config(config_path: Optional[Path] = None) -> CLIConfig:
    """
    Load configuration or create default.

    Args:
        config_path: Optional path to configuration file

    Returns:
        CLIConfig instance
    """
    if config_path is not None:
        config = ConfigLoader.load_config(config_path)
        if config is not None:
            return config

    return CLIConfig()
