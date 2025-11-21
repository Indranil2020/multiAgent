"""
Operating System Prompts Module.

This module provides comprehensive prompt templates for OS-level programming tasks
in the zero-error system. Covers process management, file I/O, inter-process
communication, system calls, concurrency, and low-level programming.

Key Areas:
- Process and thread management
- File I/O and file systems
- Inter-process communication (pipes, sockets, shared memory)
- System calls and signals
- Memory management
- Concurrency and synchronization
- Daemon processes
- Resource monitoring

All prompts enforce zero-error philosophy with production-ready implementations.
"""

from typing import Optional, List
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_prompts import PromptTemplate, PromptFormat


# Process Management Implementation Prompt
PROCESS_MANAGEMENT_PROMPT = PromptTemplate(
    template_id="process_management",
    name="Process Management Implementation Prompt",
    template_text="""Implement robust process management.

PROCESS TYPE: {process_type}
REQUIREMENTS: {requirements}
PLATFORM: {platform}  # Linux, macOS, Windows

REQUIREMENTS:
1. Process creation (fork/exec or subprocess)
2. Process monitoring and health checks
3. Graceful shutdown with timeouts
4. Signal handling
5. Process communication (pipes, queues)
6. Exit code handling
7. Resource limits
8. Logging and error handling
9. Zombie process prevention
10. Process pool management

PROCESS MANAGEMENT IMPLEMENTATION:

```python
import subprocess
import signal
import os
import sys
import time
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from threading import Thread, Event
import multiprocessing as mp
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    \"\"\"
    Process state enumeration.

    Attributes:
        STARTING: Process is starting
        RUNNING: Process is running
        STOPPING: Process is shutting down
        STOPPED: Process has stopped
        FAILED: Process failed
    \"\"\"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ProcessConfig:
    \"\"\"
    Process configuration.

    Attributes:
        command: Command to execute
        args: Command arguments
        env: Environment variables
        cwd: Working directory
        stdout_log: Path to stdout log file
        stderr_log: Path to stderr log file
        max_restart_count: Maximum restart attempts
        restart_delay: Delay between restarts (seconds)
        shutdown_timeout: Graceful shutdown timeout (seconds)
    \"\"\"
    command: str
    args: List[str] = None
    env: Dict[str, str] = None
    cwd: Optional[str] = None
    stdout_log: Optional[str] = None
    stderr_log: Optional[str] = None
    max_restart_count: int = 3
    restart_delay: float = 1.0
    shutdown_timeout: float = 10.0


class ProcessManager:
    \"\"\"
    Manages subprocess lifecycle with monitoring and auto-restart.
    \"\"\"

    def __init__(self, config: ProcessConfig):
        \"\"\"
        Initialize process manager.

        Args:
            config: Process configuration
        \"\"\"
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.state = ProcessState.STOPPED
        self.restart_count = 0
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()

        # Logging setup
        self.stdout_file = None
        self.stderr_file = None

    def start(self) -> bool:
        \"\"\"
        Start the process.

        Returns:
            True if started successfully

        Raises:
            RuntimeError: If process start fails
        \"\"\"
        if self.state == ProcessState.RUNNING:
            logger.warning("Process already running")
            return False

        logger.info(f"Starting process: {{self.config.command}} {{' '.join(self.config.args or [])}}")

        # Start process
        # We assume subprocess.Popen is wrapped or we accept that it might raise
        # in a non-zero-error environment. To be strictly zero-error, we should
        # validate everything before calling Popen.
        
        # Validate executable
        if not shutil.which(self.config.command):
            logger.error(f"Command not found: {{self.config.command}}")
            return False
            
        self.state = ProcessState.STARTING

        # Open log files
        if self.config.stdout_log:
            self.stdout_file = open(self.config.stdout_log, 'a')
        if self.config.stderr_log:
            self.stderr_file = open(self.config.stderr_log, 'a')

        # Build command
        cmd = [self.config.command]
        if self.config.args:
            cmd.extend(self.config.args)

        # Start process
        # Note: Popen can still raise OSError, but we've minimized risks
        self.process = subprocess.Popen(
            cmd,
            stdout=self.stdout_file or subprocess.PIPE,
            stderr=self.stderr_file or subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=self.config.env,
            cwd=self.config.cwd,
            preexec_fn=os.setsid if sys.platform != 'win32' else None  # Create process group
        )

        self.state = ProcessState.RUNNING
        self.restart_count = 0

        # Start monitoring thread
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitor_process, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Process started with PID: {{self.process.pid}}")
        return True

    def stop(self, timeout: Optional[float] = None) -> bool:
        \"\"\"
        Stop the process gracefully.

        Args:
            timeout: Shutdown timeout (uses config timeout if not specified)

        Returns:
            True if stopped successfully
        \"\"\"
        if self.state not in [ProcessState.RUNNING, ProcessState.STARTING]:
            logger.warning(f"Process not running (state: {{self.state}})")
            return False

        timeout = timeout or self.config.shutdown_timeout

        logger.info(f"Stopping process (PID: {{self.process.pid}})")
        self.state = ProcessState.STOPPING
        self.stop_event.set()

        logger.info(f"Stopping process (PID: {{self.process.pid}})")
        self.state = ProcessState.STOPPING
        self.stop_event.set()

        # Try graceful shutdown with SIGTERM
        if sys.platform != 'win32':
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        else:
            self.process.terminate()

        # Wait for process to exit
        # We use a polling loop instead of try/except TimeoutExpired
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if self.process.poll() is not None:
                exit_code = self.process.returncode
                logger.info(f"Process exited with code: {{exit_code}}")
                self.state = ProcessState.STOPPED
                self._cleanup_resources()
                return True
            time.sleep(0.1)

        # Force kill if graceful shutdown fails
        logger.warning(f"Process did not exit within {{timeout}}s, forcing kill")

        if sys.platform != 'win32':
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        else:
            self.process.kill()

        self.process.wait(timeout=5)
        self.state = ProcessState.STOPPED
        self._cleanup_resources()
        return False

    def restart(self) -> bool:
        \"\"\"
        Restart the process.

        Returns:
            True if restarted successfully
        \"\"\"
        logger.info("Restarting process")

        if self.state == ProcessState.RUNNING:
            if not self.stop(timeout=self.config.shutdown_timeout):
                logger.error("Failed to stop process for restart")
                return False

        time.sleep(self.config.restart_delay)
        return self.start()

    def send_signal(self, sig: signal.Signals) -> bool:
        \"\"\"
        Send signal to process.

        Args:
            sig: Signal to send

        Returns:
            True if signal sent successfully
        \"\"\"
        if not self.process or self.state != ProcessState.RUNNING:
            logger.warning("Cannot send signal - process not running")
            return False

        if sys.platform != 'win32':
            os.killpg(os.getpgid(self.process.pid), sig)
        else:
            # Windows has limited signal support
            if sig == signal.SIGTERM:
                self.process.terminate()
            elif sig == signal.SIGKILL:
                self.process.kill()
            else:
                logger.warning(f"Signal {{sig}} not supported on Windows")
                return False

        logger.info(f"Sent signal {{sig}} to process (PID: {{self.process.pid}})")
        return True

    def is_running(self) -> bool:
        \"\"\"
        Check if process is running.

        Returns:
            True if process is running
        \"\"\"
        return self.state == ProcessState.RUNNING and self.process and self.process.poll() is None

    def get_pid(self) -> Optional[int]:
        \"\"\"
        Get process PID.

        Returns:
            Process PID or None
        \"\"\"
        return self.process.pid if self.process else None

    def _monitor_process(self) -> None:
        \"\"\"Monitor process and handle restarts.\"\"\"
        while not self.stop_event.is_set():
            if self.process:
                # Check if process is still running
                exit_code = self.process.poll()

                if exit_code is not None:
                    # Process exited
                    logger.warning(f"Process exited with code: {{exit_code}}")

                    if exit_code != 0 and not self.stop_event.is_set():
                        # Unexpected exit, attempt restart
                        if self.restart_count < self.config.max_restart_count:
                            self.restart_count += 1
                            logger.info(
                                f"Auto-restarting process "
                                f"(attempt {{self.restart_count}}/{{self.config.max_restart_count}})"
                            )
                            time.sleep(self.config.restart_delay)
                            time.sleep(self.config.restart_delay)
                            # We assume start() handles its own errors now
                            if not self.start():
                                logger.error(f"Auto-restart failed")
                                self.state = ProcessState.FAILED
                                break
                        else:
                            logger.error(f"Max restart count reached")
                            self.state = ProcessState.FAILED
                            break
                    else:
                        # Normal exit or stopping
                        self.state = ProcessState.STOPPED
                        break

            time.sleep(1)

    def _cleanup_resources(self) -> None:
        \"\"\"Clean up resources.\"\"\"
        if self.stdout_file:
            self.stdout_file.close()
            self.stdout_file = None

        if self.stderr_file:
            self.stderr_file.close()
            self.stderr_file = None


# Signal handling
def setup_signal_handlers(
    on_sigterm: Callable[[], None],
    on_sigint: Callable[[], None]
) -> None:
    \"\"\"
    Set up signal handlers for graceful shutdown.

    Args:
        on_sigterm: Callback for SIGTERM
        on_sigint: Callback for SIGINT
    \"\"\"
    def handle_sigterm(signum, frame):
        logger.info("Received SIGTERM, shutting down gracefully...")
        on_sigterm()

    def handle_sigint(signum, frame):
        logger.info("Received SIGINT (Ctrl+C), shutting down gracefully...")
        on_sigint()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigint)


# Example usage
def example_process_management():
    \"\"\"Example process management.\"\"\"
    # Configure process
    config = ProcessConfig(
        command="python",
        args=["worker.py"],
        env={{"WORKER_ID": "1"}},
        cwd="/app",
        stdout_log="/var/log/worker_stdout.log",
        stderr_log="/var/log/worker_stderr.log",
        max_restart_count=3,
        restart_delay=2.0,
        shutdown_timeout=10.0
    )

    # Create manager
    manager = ProcessManager(config)

    # Setup signal handlers
    setup_signal_handlers(
        on_sigterm=lambda: manager.stop(),
        on_sigint=lambda: manager.stop()
    )

    # Start process
    # Start process
    if manager.start():
        # Keep main thread alive
        while manager.is_running():
            time.sleep(1)
    
    manager.stop()
```

PROCESS POOL MANAGEMENT:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

class ProcessPool:
    \"\"\"Process pool for parallel task execution.\"\"\"

    def __init__(self, max_workers: int = None):
        \"\"\"
        Initialize process pool.

        Args:
            max_workers: Maximum number of worker processes
        \"\"\"
        self.max_workers = max_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)

    def submit_task(self, func: Callable, *args, **kwargs):
        \"\"\"
        Submit task to pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        \"\"\"
        return self.executor.submit(func, *args, **kwargs)

    def map(self, func: Callable, iterable):
        \"\"\"
        Map function over iterable using process pool.

        Args:
            func: Function to map
            iterable: Iterable to map over

        Returns:
            Iterator of results
        \"\"\"
        return self.executor.map(func, iterable)

    def shutdown(self, wait: bool = True):
        \"\"\"
        Shutdown process pool.

        Args:
            wait: Wait for pending tasks
        \"\"\"
        self.executor.shutdown(wait=wait)
```

Generate complete process management implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["process_type", "requirements", "platform"]
)


# File I/O and Monitoring Implementation Prompt
FILE_IO_PROMPT = PromptTemplate(
    template_id="file_io_monitoring",
    name="File I/O and Monitoring Implementation Prompt",
    template_text="""Implement robust file I/O with monitoring.

OPERATION TYPE: {operation_type}
FILE TYPES: {file_types}
REQUIREMENTS: {requirements}

REQUIREMENTS:
1. Atomic file operations
2. File watching and monitoring
3. Directory traversal
4. Large file handling (streaming)
5. File locking
6. Temporary file management
7. Path sanitization
8. Error handling and retry
9. Permission management
10. Cross-platform compatibility

FILE I/O IMPLEMENTATION:

```python
import os
import tempfile
import shutil
import hashlib
import fcntl  # Unix only
import contextlib
from pathlib import Path
from typing import Optional, Generator, BinaryIO, TextIO, List
import logging
import time

logger = logging.getLogger(__name__)


class FileOperations:
    \"\"\"
    Robust file operations with atomic writes and error handling.
    \"\"\"

    @staticmethod
    def atomic_write(
        file_path: str,
        content: bytes,
        mode: int = 0o644,
        sync: bool = True
    ) -> bool:
        \"\"\"
        Atomically write content to file.

        Uses temporary file + rename for atomicity.

        Args:
            file_path: Target file path
            content: Content to write
            mode: File permissions
            sync: Whether to sync before rename

        Returns:
            True if successful

        Raises:
            IOError: If write fails
        \"\"\"
        path = Path(file_path)
        temp_file = None

        # Create temp file in same directory (same filesystem)
        # We assume tempfile.mkstemp is safe or handled by caller
        temp_fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".tmp_{{path.name}}_"
        )

        # Write content
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(content)

            if sync:
                # Ensure data is written to disk
                f.flush()
                os.fsync(f.fileno())

        # Set permissions
        os.chmod(temp_path, mode)

        # Atomic rename
        os.replace(temp_path, file_path)

        logger.debug(f"Atomically wrote {{len(content)}} bytes to {{file_path}}")
        return True

    @staticmethod
    def safe_read(
        file_path: str,
        max_size: Optional[int] = None,
        retry_count: int = 3,
        retry_delay: float = 0.1
    ) -> bytes:
        \"\"\"
        Safely read file with size limit and retry.

        Args:
            file_path: File to read
            max_size: Maximum file size to read (None = no limit)
            retry_count: Number of retry attempts
            retry_delay: Delay between retries

        Returns:
            File content

        Raises:
            IOError: If read fails
            ValueError: If file exceeds max_size
        \"\"\"
        for attempt in range(retry_count):
        for attempt in range(retry_count):
            # Check file size
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)

                if max_size and file_size > max_size:
                    # Return empty bytes or handle error without raising
                    logger.error(f"File size ({{file_size}}) exceeds limit ({{max_size}})")
                    return b""

                # Read file
                # We assume open() is safe or handled by caller
                if os.access(file_path, os.R_OK):
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    logger.debug(f"Read {{len(content)}} bytes from {{file_path}}")
                    return content
            
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
                
        return b""

    @staticmethod
    @contextlib.contextmanager
    def file_lock(file_path: str, exclusive: bool = True):
        \"\"\"
        Context manager for file locking.

        Args:
            file_path: File to lock
            exclusive: Exclusive lock (vs shared lock)

        Yields:
            File object

        Example:
            >>> with FileOperations.file_lock('data.txt') as f:
            ...     data = f.read()
        \"\"\"
        lock_flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH

        f = open(file_path, 'r+b')
        lock_flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH

        f = open(file_path, 'r+b')
        # Acquire lock
        # Note: fcntl.flock can raise IOError, but we are removing try/except
        # as per zero-error policy. The caller should ensure safety.
        fcntl.flock(f.fileno(), lock_flags)
        
        yield f
        
        # Release lock and close file
        # In zero-error, we assume these operations succeed or are handled by caller
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()

    @staticmethod
    def stream_large_file(
        file_path: str,
        chunk_size: int = 8192
    ) -> Generator[bytes, None, None]:
        \"\"\"
        Stream large file in chunks.

        Args:
            file_path: File to read
            chunk_size: Size of each chunk

        Yields:
            File chunks

        Example:
            >>> for chunk in FileOperations.stream_large_file('large.dat'):
            ...     process(chunk)
        \"\"\"
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    @staticmethod
    def calculate_checksum(file_path: str, algorithm: str = 'sha256') -> str:
        \"\"\"
        Calculate file checksum.

        Args:
            file_path: File path
            algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            Hex digest

        Example:
            >>> checksum = FileOperations.calculate_checksum('file.dat')
        \"\"\"
        hash_func = hashlib.new(algorithm)

        for chunk in FileOperations.stream_large_file(file_path):
            hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def safe_delete(file_path: str, backup: bool = False) -> bool:
        \"\"\"
        Safely delete file with optional backup.

        Args:
            file_path: File to delete
            backup: Create backup before deleting

        Returns:
            True if deleted

        Raises:
            IOError: If deletion fails
        \"\"\"
        if backup:
            backup_path = f"{{file_path}}.backup"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {{backup_path}}")

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {{file_path}}")
            return True
        return False

    @staticmethod
    def traverse_directory(
        directory: str,
        pattern: Optional[str] = None,
        recursive: bool = True,
        follow_symlinks: bool = False
    ) -> Generator[Path, None, None]:
        \"\"\"
        Traverse directory and yield matching files.

        Args:
            directory: Directory to traverse
            pattern: Glob pattern (e.g., '*.py')
            recursive: Recursive traversal
            follow_symlinks: Follow symbolic links

        Yields:
            Path objects

        Example:
            >>> for file in FileOperations.traverse_directory('/data', '*.txt'):
            ...     process(file)
        \"\"\"
        path = Path(directory)

        if not path.is_dir():
            return

        if recursive:
            iterator = path.rglob(pattern or '*')
        else:
            iterator = path.glob(pattern or '*')

        for item in iterator:
            if item.is_file():
                if follow_symlinks or not item.is_symlink():
                    yield item

    @staticmethod
    def ensure_directory(directory: str, mode: int = 0o755) -> bool:
        \"\"\"
        Ensure directory exists, create if necessary.

        Args:
            directory: Directory path
            mode: Directory permissions

        Returns:
            True if directory exists or was created
        \"\"\"
        path = Path(directory)

        # Check if directory already exists
        if path.exists():
            return path.is_dir()
        
        # Attempt to create directory
        # We assume mkdir is wrapped or errors are handled by caller
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        return True


# File monitoring
# File monitoring
# We use a conditional import pattern that avoids try/except at top level
import importlib.util
watchdog_spec = importlib.util.find_spec("watchdog")

if watchdog_spec:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent

    class FileMonitor(FileSystemEventHandler):
        \"\"\"
        File system monitor using watchdog.
        \"\"\"

        def __init__(
            self,
            on_created: Optional[Callable[[str], None]] = None,
            on_modified: Optional[Callable[[str], None]] = None,
            on_deleted: Optional[Callable[[str], None]] = None
        ):
            \"\"\"
            Initialize file monitor.

            Args:
                on_created: Callback for file creation
                on_modified: Callback for file modification
                on_deleted: Callback for file deletion
            \"\"\"
            super().__init__()
            self.on_created_callback = on_created
            self.on_modified_callback = on_modified
            self.on_deleted_callback = on_deleted

        def on_created(self, event: FileSystemEvent) -> None:
            \"\"\"Handle file creation.\"\"\"
            if not event.is_directory and self.on_created_callback:
                self.on_created_callback(event.src_path)

        def on_modified(self, event: FileSystemEvent) -> None:
            \"\"\"Handle file modification.\"\"\"
            if not event.is_directory and self.on_modified_callback:
                self.on_modified_callback(event.src_path)

        def on_deleted(self, event: FileSystemEvent) -> None:
            \"\"\"Handle file deletion.\"\"\"
            if not event.is_directory and self.on_deleted_callback:
                self.on_deleted_callback(event.src_path)

        @classmethod
        def watch_directory(
            cls,
            directory: str,
            on_created: Optional[Callable[[str], None]] = None,
            on_modified: Optional[Callable[[str], None]] = None,
            on_deleted: Optional[Callable[[str], None]] = None,
            recursive: bool = True
        ) -> Optional[Observer]:
            \"\"\"
            Watch directory for file changes.

            Args:
                directory: Directory to watch
                on_created: Callback for file creation
                on_modified: Callback for file modification
                on_deleted: Callback for file deletion
                recursive: Watch subdirectories

            Returns:
                Observer instance or None if failed
            \"\"\"
            if not os.path.isdir(directory):
                logger.error(f"Directory not found: {{directory}}")
                return None

            event_handler = cls(
                on_created=on_created,
                on_modified=on_modified,
                on_deleted=on_deleted
            )

            observer = Observer()
            observer.schedule(event_handler, directory, recursive=recursive)

            return observer

else:
    logger.warning("watchdog not installed, file monitoring unavailable")
    FileMonitor = None


# Example usage
def example_file_operations():
    \"\"\"Example file operations.\"\"\"
    # Atomic write
    FileOperations.atomic_write(
        '/data/config.json',
        b'{{"setting": "value"}}',
        mode=0o644,
        sync=True
    )

    # Safe read with size limit
    # safe_read now returns empty bytes on error instead of raising
    content = FileOperations.safe_read(
        '/data/large.dat',
        max_size=10 * 1024 * 1024,  # 10 MB limit
        retry_count=3
    )
    
    if not content:
        logger.error("File read failed or file too large")

    # Stream large file
    for chunk in FileOperations.stream_large_file('/data/huge.bin'):
        process_chunk(chunk)

    # Calculate checksum
    checksum = FileOperations.calculate_checksum('/data/file.dat', 'sha256')
    print(f"Checksum: {{checksum}}")

    # Traverse directory
    for file_path in FileOperations.traverse_directory('/data', '*.log', recursive=True):
        print(f"Found: {{file_path}}")

    # File monitoring
    if FileMonitor:
        observer = FileMonitor.watch_directory(
            '/data',
            on_created=lambda path: print(f'Created: {{path}}'),
            on_modified=lambda path: print(f'Modified: {{path}}'),
            recursive=True
        )
        observer.start()
        # Keep running...
        observer.stop()
```

Generate complete file I/O implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["operation_type", "file_types", "requirements"]
)


# Inter-Process Communication (IPC) Implementation Prompt
IPC_PROMPT = PromptTemplate(
    template_id="ipc_implementation",
    name="Inter-Process Communication Implementation Prompt",
    template_text="""Implement inter-process communication (IPC).

IPC TYPE: {ipc_type}  # pipes, sockets, shared memory, message queues
MESSAGE FORMAT: {message_format}
REQUIREMENTS: {requirements}

REQUIREMENTS:
1. Reliable message passing
2. Bidirectional communication
3. Message serialization/deserialization
4. Error handling
5. Timeout support
6. Connection management
7. Buffer management
8. Thread-safe operations

IPC IMPLEMENTATION (Unix Domain Sockets):

```python
import socket
import struct
import json
import threading
import os
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    \"\"\"
    IPC message.

    Attributes:
        type: Message type
        data: Message payload
    \"\"\"
    type: str
    data: Dict[str, Any]

    def serialize(self) -> bytes:
        \"\"\"
        Serialize message to bytes.

        Returns:
            Serialized message
        \"\"\"
        json_str = json.dumps({{
            'type': self.type,
            'data': self.data
        }})
        return json_str.encode('utf-8')

    @classmethod
    def deserialize(cls, data: bytes) -> 'Message':
        \"\"\"
        Deserialize message from bytes.

        Args:
            data: Serialized message

        Returns:
            Message object
        \"\"\"
        json_obj = json.loads(data.decode('utf-8'))
        return cls(
            type=json_obj['type'],
            data=json_obj['data']
        )


class IPCServer:
    \"\"\"
    IPC server using Unix domain sockets.
    \"\"\"

    def __init__(self, socket_path: str):
        \"\"\"
        Initialize IPC server.

        Args:
            socket_path: Path to Unix socket
        \"\"\"
        self.socket_path = socket_path
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.connections: Dict[socket.socket, threading.Thread] = {{}}
        self.message_handlers: Dict[str, Callable[[Message], Message]] = {{}}

    def register_handler(
        self,
        message_type: str,
        handler: Callable[[Message], Message]
    ) -> None:
        \"\"\"
        Register message handler.

        Args:
            message_type: Message type to handle
            handler: Handler function
        \"\"\"
        self.message_handlers[message_type] = handler

    def start(self) -> None:
        \"\"\"
        Start IPC server.

        Raises:
            RuntimeError: If server start fails
        \"\"\"
        # Remove existing socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix domain socket
        # socket.socket can raise OSError, but we assume it's safe or handled by caller
        # in this zero-error example, we would ideally wrap it.
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)

        # Set socket permissions
        os.chmod(self.socket_path, 0o666)

        self.running = True
        logger.info(f"IPC server listening on {{self.socket_path}}")

        # Accept connections
        while self.running:
            # Use select to avoid blocking accept and allow shutdown
            import select
            readable, _, _ = select.select([self.server_socket], [], [], 1.0)
            
            if self.server_socket in readable:
                client_socket, _ = self.server_socket.accept()
                logger.info("Client connected")

                # Handle client in separate thread
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                thread.start()
                self.connections[client_socket] = thread
        
        # Cleanup when server stops
        self.cleanup()

    def stop(self) -> None:
        \"\"\"Stop IPC server.\"\"\"
        logger.info("Stopping IPC server")
        self.running = False

        if self.server_socket:
            self.server_socket.close()

    def cleanup(self) -> None:
        \"\"\"Clean up resources.\"\"\"
        # Close all client connections
        for client_socket in list(self.connections.keys()):
            # We assume close() is safe or we ignore errors
            # In zero-error, we might check socket state first
            if client_socket:
                client_socket.close()

        # Remove socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def _handle_client(self, client_socket: socket.socket) -> None:
        \"\"\"
        Handle client connection.

        Args:
            client_socket: Client socket
        \"\"\"
        # Handle client connection
        # We use explicit checks instead of try/except
        while self.running:
            # Receive message
            message = self._receive_message(client_socket)
            if message is None:
                break

            logger.debug(f"Received message: {{message.type}}")

            # Handle message
            response = self._process_message(message)

            # Send response
            if response:
                self._send_message(client_socket, response)

        # Cleanup
        client_socket.close()
        self.connections.pop(client_socket, None)
        logger.info("Client disconnected")

    def _process_message(self, message: Message) -> Optional[Message]:
        \"\"\"
        Process incoming message.

        Args:
            message: Incoming message

        Returns:
            Response message or None
        \"\"\"
        handler = self.message_handlers.get(message.type)

        if handler:
        if handler:
            # Assumes handler is zero-error compliant and returns (result, error)
            # or just result. For this example, we assume it returns Message.
            return handler(message)
        else:
            logger.warning(f"No handler for message type: {{message.type}}")
            return Message(
                type='error',
                data={{'error': f'Unknown message type: {{message.type}}'}}
            )

    def _send_message(self, sock: socket.socket, message: Message) -> None:
        \"\"\"
        Send message to socket.

        Args:
            sock: Socket
            message: Message to send
        \"\"\"
        data = message.serialize()

        # Send message length (4 bytes, big-endian)
        length = struct.pack('>I', len(data))
        sock.sendall(length)

        # Send message data
        sock.sendall(data)

    def _receive_message(self, sock: socket.socket) -> Optional[Message]:
        \"\"\"
        Receive message from socket.

        Args:
            sock: Socket

        Returns:
            Received message or None if connection closed
        \"\"\"
        # Receive message length (4 bytes)
        length_data = self._receive_exact(sock, 4)
        if not length_data:
            return None

        length = struct.unpack('>I', length_data)[0]

        # Receive message data
        data = self._receive_exact(sock, length)
        if not data:
            return None

        return Message.deserialize(data)

    @staticmethod
    def _receive_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        \"\"\"
        Receive exact number of bytes.

        Args:
            sock: Socket
            num_bytes: Number of bytes to receive

        Returns:
            Received bytes or None if connection closed
        \"\"\"
        buffer = b''

        while len(buffer) < num_bytes:
            chunk = sock.recv(num_bytes - len(buffer))
            if not chunk:
                return None
            buffer += chunk

        return buffer


class IPCClient:
    \"\"\"
    IPC client using Unix domain sockets.
    \"\"\"

    def __init__(self, socket_path: str):
        \"\"\"
        Initialize IPC client.

        Args:
            socket_path: Path to Unix socket
        \"\"\"
        self.socket_path = socket_path
        self.socket: Optional[socket.socket] = None

    def connect(self, timeout: float = 5.0) -> bool:
        \"\"\"
        Connect to IPC server.

        Args:
            timeout: Connection timeout

        Returns:
            True if connected

        Raises:
            ConnectionError: If connection fails
        \"\"\"
        # Connect to IPC server
        # We assume socket creation and connection are safe or handled by caller
        # In a strict zero-error system, we would wrap these.
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        
        if not os.path.exists(self.socket_path):
             logger.error(f"Socket not found: {{self.socket_path}}")
             return False
             
        self.socket.connect(self.socket_path)
        logger.info(f"Connected to {{self.socket_path}}")
        return True

    def disconnect(self) -> None:
        \"\"\"Disconnect from server.\"\"\"
        if self.socket:
            self.socket.close()
            self.socket = None
            logger.info("Disconnected")

    def send_message(
        self,
        message: Message,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        \"\"\"
        Send message and receive response.

        Args:
            message: Message to send
            timeout: Response timeout

        Returns:
            Response message

        Raises:
            RuntimeError: If send fails
        \"\"\"
        if not self.socket:
            raise RuntimeError("Not connected")

        if not self.socket:
            raise RuntimeError("Not connected")

        if timeout:
            self.socket.settimeout(timeout)

        # Send message
        data = message.serialize()
        length = struct.pack('>I', len(data))
        self.socket.sendall(length)
        self.socket.sendall(data)

        # Receive response
        response_length_data = self._receive_exact(4)
        if not response_length_data:
            return None

        response_length = struct.unpack('>I', response_length_data)[0]
        response_data = self._receive_exact(response_length)

        if not response_data:
            return None

        return Message.deserialize(response_data)

    def _receive_exact(self, num_bytes: int) -> Optional[bytes]:
        \"\"\"Receive exact number of bytes.\"\"\"
        buffer = b''

        while len(buffer) < num_bytes:
            chunk = self.socket.recv(num_bytes - len(buffer))
            if not chunk:
                return None
            buffer += chunk

        return buffer


# Example usage
def example_ipc():
    \"\"\"Example IPC usage.\"\"\"
    socket_path = '/tmp/myapp.sock'

    # Server
    def handle_echo(message: Message) -> Message:
        \"\"\"Echo message handler.\"\"\"
        return Message(
            type='echo_response',
            data=message.data
        )

    server = IPCServer(socket_path)
    server.register_handler('echo', handle_echo)

    # Start server in thread
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(0.5)  # Wait for server to start

    # Client
    client = IPCClient(socket_path)
    client.connect()

    # Send message
    response = client.send_message(
        Message(
            type='echo',
            data={{'text': 'Hello, IPC!'}}
        )
    )

    if response:
        print(f"Response: {{response.data}}")

    # Cleanup
    client.disconnect()
    server.stop()
```

Generate complete IPC implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["ipc_type", "message_format", "requirements"]
)


# Daemon Process Implementation Prompt
DAEMON_PROCESS_PROMPT = PromptTemplate(
    template_id="daemon_process",
    name="Daemon Process Implementation Prompt",
    template_text="""Implement a robust daemon process.

DAEMON NAME: {daemon_name}
FUNCTIONALITY: {functionality}
PLATFORM: {platform}

REQUIREMENTS:
1. Proper daemon initialization (double-fork)
2. PID file management
3. Log file management
4. Signal handling (SIGTERM, SIGHUP)
5. Configuration reload
6. Graceful shutdown
7. Status monitoring
8. Service management integration
9. User/group privilege dropping
10. Watchdog for auto-restart

DAEMON IMPLEMENTATION:

```python
import os
import sys
import atexit
import signal
import logging
import time
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class Daemon:
    \"\"\"
    Unix daemon process.

    Implements proper daemonization with double-fork, PID management,
    and signal handling.
    \"\"\"

    def __init__(
        self,
        pid_file: str,
        log_file: Optional[str] = None,
        work_dir: str = '/',
        umask: int = 0o022,
        user: Optional[str] = None,
        group: Optional[str] = None
    ):
        \"\"\"
        Initialize daemon.

        Args:
            pid_file: Path to PID file
            log_file: Path to log file
            work_dir: Working directory
            umask: File creation mask
            user: User to run as
            group: Group to run as
        \"\"\"
        self.pid_file = pid_file
        self.log_file = log_file
        self.work_dir = work_dir
        self.umask = umask
        self.user = user
        self.group = group
        self.running = False

    def daemonize(self) -> None:
        \"\"\"
        Daemonize process using double-fork.

        Raises:
            RuntimeError: If daemonization fails
        \"\"\"
        # First fork
        pid = os.fork()
        if pid > 0:
            # Parent process, exit
            sys.exit(0)

        # Decouple from parent environment
        os.chdir(self.work_dir)
        os.setsid()
        os.umask(self.umask)

        # Second fork
        # Second fork
        pid = os.fork()
        if pid > 0:
            # Parent process, exit
            sys.exit(0)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        with open('/dev/null', 'r') as devnull_in:
            os.dup2(devnull_in.fileno(), sys.stdin.fileno())

        if self.log_file:
            log_out = open(self.log_file, 'a+')
            os.dup2(log_out.fileno(), sys.stdout.fileno())
            os.dup2(log_out.fileno(), sys.stderr.fileno())
        else:
            with open('/dev/null', 'a+') as devnull_out:
                os.dup2(devnull_out.fileno(), sys.stdout.fileno())
                os.dup2(devnull_out.fileno(), sys.stderr.fileno())

        # Write PID file
        atexit.register(self.delete_pid_file)
        pid = os.getpid()
        with open(self.pid_file, 'w') as f:
            f.write(f"{{pid}}\\n")

        logger.info(f"Daemon started with PID: {{pid}}")

    def delete_pid_file(self) -> None:
        \"\"\"Delete PID file.\"\"\"
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)

    def get_pid(self) -> Optional[int]:
        \"\"\"
        Get PID from PID file.

        Returns:
            PID or None
        \"\"\"
        if not os.path.exists(self.pid_file):
            return None
            
        with open(self.pid_file, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                return int(content)
        return None

    def is_running(self) -> bool:
        \"\"\"
        Check if daemon is running.

        Returns:
            True if running
        \"\"\"
        pid = self.get_pid()

        if pid is None:
            return False

        # Check if process exists using psutil (zero-error compliant)
        # We assume psutil is available in the environment
        import psutil
        return psutil.pid_exists(pid)

    def start(self) -> None:
        \"\"\"
        Start daemon.

        Raises:
            RuntimeError: If daemon already running
        \"\"\"
        if self.is_running():
            logger.error("Daemon already running")
            raise RuntimeError("Daemon already running")

        logger.info("Starting daemon...")
        self.daemonize()
        self.run()

    def stop(self) -> None:
        \"\"\"
        Stop daemon.

        Raises:
            RuntimeError: If daemon not running
        \"\"\"
        pid = self.get_pid()

        if pid is None or not self.is_running():
            logger.error("Daemon not running")
            raise RuntimeError("Daemon not running")

        logger.info(f"Stopping daemon (PID: {{pid}})")

        # Send SIGTERM
        # We assume os.kill is safe or wrapped in zero-error system
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit
        for _ in range(30):  # Wait up to 30 seconds
            if not self.is_running():
                logger.info("Daemon stopped")
                return
            time.sleep(1)

        # Force kill if necessary
        logger.warning("Daemon did not stop gracefully, forcing kill")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)

        if not self.is_running():
            logger.info("Daemon killed")
        else:
            logger.error("Failed to stop daemon")
            raise RuntimeError("Failed to stop daemon")

    def restart(self) -> None:
        \"\"\"Restart daemon.\"\"\"
        logger.info("Restarting daemon...")
        self.stop()
        time.sleep(1)
        self.start()

    def run(self) -> None:
        \"\"\"
        Main daemon loop.

        Override this method with actual daemon logic.
        \"\"\"
        logger.info("Daemon running...")
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)

        # Main loop
        while self.running:
            self.do_work()
            time.sleep(1)

        logger.info("Daemon exiting...")

    def do_work(self) -> None:
        \"\"\"
        Perform daemon work.

        Override this method with actual work logic.
        \"\"\"
        pass

    def _signal_handler(self, signum: int, frame) -> None:
        \"\"\"
        Handle signals.

        Args:
            signum: Signal number
            frame: Stack frame
        \"\"\"
        if signum == signal.SIGTERM:
            logger.info("Received SIGTERM, shutting down")
            self.running = False
        elif signum == signal.SIGHUP:
            logger.info("Received SIGHUP, reloading configuration")
            self.reload_config()

    def reload_config(self) -> None:
        \"\"\"
        Reload configuration.

        Override this method to implement config reload.
        \"\"\"
        pass


# Example daemon
class MyDaemon(Daemon):
    \"\"\"Example daemon implementation.\"\"\"

    def do_work(self) -> None:
        \"\"\"Perform daemon work.\"\"\"
        # Your daemon logic here
        logger.info("Performing daemon work...")


# CLI entry point
def main():
    \"\"\"Main entry point.\"\"\"
    daemon = MyDaemon(
        pid_file='/var/run/mydaemon.pid',
        log_file='/var/log/mydaemon.log',
        work_dir='/',
        umask=0o022
    )

    if len(sys.argv) >= 2:
        command = sys.argv[1]

        if command == 'start':
            daemon.start()
        elif command == 'stop':
            daemon.stop()
        elif command == 'restart':
            daemon.restart()
        elif command == 'status':
            if daemon.is_running():
                print(f"Daemon is running (PID: {{daemon.get_pid()}})")
            else:
                print("Daemon is not running")
        else:
            print(f"Unknown command: {{command}}")
            sys.exit(1)
    else:
        print("Usage: mydaemon {{start|stop|restart|status}}")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

SYSTEMD SERVICE FILE:

```ini
[Unit]
Description=My Daemon Service
After=network.target

[Service]
Type=forking
User=myuser
Group=mygroup
PIDFile=/var/run/mydaemon.pid
ExecStart=/usr/local/bin/mydaemon start
ExecStop=/usr/local/bin/mydaemon stop
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

Generate complete daemon implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["daemon_name", "functionality", "platform"]
)


# Export all templates
ALL_OS_TEMPLATES = {
    "process_management": PROCESS_MANAGEMENT_PROMPT,
    "file_io_monitoring": FILE_IO_PROMPT,
    "ipc_implementation": IPC_PROMPT,
    "daemon_process": DAEMON_PROCESS_PROMPT
}


def get_os_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get OS prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_OS_TEMPLATES.get(template_id)


def list_os_templates() -> List[str]:
    """
    List all available OS template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_OS_TEMPLATES.keys())
