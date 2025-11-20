"""
Logging configuration implementation.

This module provides comprehensive logging configuration with structured logging,
multiple handlers, formatters, and log level management.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time
import sys


class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """Log output formats."""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogRecord:
    """
    A log record.
    
    Attributes:
        level: Log level
        message: Log message
        timestamp: When log was created
        logger_name: Name of logger
        module: Module name
        function: Function name
        line_number: Line number
        extra: Extra fields
    """
    level: LogLevel
    message: str
    timestamp: float
    logger_name: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    extra: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    def is_valid(self) -> bool:
        """Check if log record is valid."""
        return bool(self.message and self.logger_name)


@dataclass
class LoggingConfig:
    """
    Configuration for logging.
    
    Attributes:
        default_level: Default log level
        format_type: Log format type
        enable_console: Enable console logging
        enable_file: Enable file logging
        log_file_path: Path to log file
        max_file_size_mb: Maximum log file size
        backup_count: Number of backup files
        include_timestamp: Include timestamp in logs
        include_level: Include level in logs
        include_logger_name: Include logger name
    """
    default_level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.TEXT
    enable_console: bool = True
    enable_file: bool = False
    log_file_path: str = "app.log"
    max_file_size_mb: float = 10.0
    backup_count: int = 5
    include_timestamp: bool = True
    include_level: bool = True
    include_logger_name: bool = True
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if self.enable_file and not self.log_file_path:
            return (False, "log_file_path cannot be empty when file logging is enabled")
        
        if self.max_file_size_mb <= 0:
            return (False, "max_file_size_mb must be positive")
        
        if self.backup_count < 0:
            return (False, "backup_count cannot be negative")
        
        return (True, "")


class Logger:
    """
    Logger for structured logging.
    
    Provides logging with levels, structured fields, and multiple outputs.
    """
    
    def __init__(self, name: str, config: LoggingConfig):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            config: Logging configuration
        """
        self.name = name
        self.config = config
        self.level = config.default_level
        self.records: List[LogRecord] = []
        self.handlers: List[Any] = []
    
    def set_level(self, level: LogLevel) -> Tuple[bool, str]:
        """
        Set log level.
        
        Args:
            level: New log level
        
        Returns:
            Tuple of (success, message)
        """
        self.level = level
        return (True, f"Log level set to {level.name}")
    
    def debug(self, message: str, **kwargs) -> Tuple[bool, str]:
        """Log debug message."""
        return self._log(LogLevel.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs) -> Tuple[bool, str]:
        """Log info message."""
        return self._log(LogLevel.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs) -> Tuple[bool, str]:
        """Log warning message."""
        return self._log(LogLevel.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs) -> Tuple[bool, str]:
        """Log error message."""
        return self._log(LogLevel.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs) -> Tuple[bool, str]:
        """Log critical message."""
        return self._log(LogLevel.CRITICAL, message, kwargs)
    
    def _log(self, level: LogLevel, message: str, extra: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Internal logging method.
        
        Args:
            level: Log level
            message: Log message
            extra: Extra fields
        
        Returns:
            Tuple of (success, message)
        """
        if not message:
            return (False, "message cannot be empty")
        
        # Check if we should log this level
        if level.value < self.level.value:
            return (True, "Log level below threshold")
        
        # Create log record
        record = LogRecord(
            level=level,
            message=message,
            timestamp=time.time(),
            logger_name=self.name,
            extra=extra
        )
        
        if not record.is_valid():
            return (False, "Invalid log record")
        
        # Store record
        self.records.append(record)
        
        # Format and output
        formatted = self._format_record(record)
        
        if self.config.enable_console:
            print(formatted, file=sys.stderr)
        
        return (True, "Logged")
    
    def _format_record(self, record: LogRecord) -> str:
        """Format log record."""
        parts = []
        
        if self.config.include_timestamp:
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.timestamp))
            parts.append(timestamp_str)
        
        if self.config.include_level:
            parts.append(f"[{record.level.name}]")
        
        if self.config.include_logger_name:
            parts.append(f"{record.logger_name}:")
        
        parts.append(record.message)
        
        # Add extra fields
        if record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            parts.append(f"({extra_str})")
        
        return " ".join(parts)
    
    def get_records(
        self,
        level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[LogRecord]:
        """
        Get log records.
        
        Args:
            level: Filter by level
            limit: Maximum records to return
        
        Returns:
            List of log records
        """
        records = self.records
        
        if level:
            records = [r for r in records if r.level == level]
        
        # Return most recent first
        records = list(reversed(records[-limit:]))
        
        return records
    
    def clear_records(self) -> None:
        """Clear all log records."""
        self.records.clear()


class LoggingManager:
    """
    Central logging manager.
    
    Manages multiple loggers and provides centralized configuration.
    """
    
    def __init__(self, config: LoggingConfig):
        """
        Initialize logging manager.
        
        Args:
            config: Logging configuration
        """
        self.config = config
        self.loggers: Dict[str, Logger] = {}
    
    def validate_config(self) -> Tuple[bool, str]:
        """Validate configuration."""
        return self.config.validate()
    
    def get_logger(self, name: str) -> Tuple[bool, Optional[Logger], str]:
        """
        Get or create a logger.
        
        Args:
            name: Logger name
        
        Returns:
            Tuple of (success, logger or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if name in self.loggers:
            return (True, self.loggers[name], "Existing logger retrieved")
        
        # Create new logger
        logger = Logger(name, self.config)
        self.loggers[name] = logger
        
        return (True, logger, f"Logger '{name}' created")
    
    def set_global_level(self, level: LogLevel) -> Tuple[bool, str]:
        """
        Set log level for all loggers.
        
        Args:
            level: New log level
        
        Returns:
            Tuple of (success, message)
        """
        self.config.default_level = level
        
        for logger in self.loggers.values():
            logger.set_level(level)
        
        return (True, f"Global log level set to {level.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        total_records = sum(len(logger.records) for logger in self.loggers.values())
        
        return {
            "total_loggers": len(self.loggers),
            "total_records": total_records,
            "default_level": self.config.default_level.name,
            "console_enabled": self.config.enable_console,
            "file_enabled": self.config.enable_file
        }
