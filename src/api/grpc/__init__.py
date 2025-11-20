"""
gRPC API Module.

This module provides high-performance gRPC services for the zero-error system:
- Binary protocol for efficiency
- Streaming support for long-running tasks
- Strong typing via Protocol Buffers
- Language-agnostic client support

Components:
- server: gRPC server implementation
- services: Service handler implementations

Proto Services:
- TaskService: Task management operations
- AgentService: Agent coordination operations
- VerificationService: Code verification operations
- MonitoringService: System monitoring operations
"""

from .server import (
    GRPCServer,
    GRPCConfig
)

from .services import (
    TaskServiceHandler,
    AgentServiceHandler,
    VerificationServiceHandler,
    MonitoringServiceHandler
)

__all__ = [
    # Server
    'GRPCServer',
    'GRPCConfig',

    # Services
    'TaskServiceHandler',
    'AgentServiceHandler',
    'VerificationServiceHandler',
    'MonitoringServiceHandler',
]
