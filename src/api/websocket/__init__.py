"""
WebSocket API Module.

This module provides real-time WebSocket communication for the zero-error system:
- Bidirectional real-time communication
- Task progress streaming
- Agent status updates
- Red flag notifications
- System event broadcasting

Components:
- server: WebSocket server implementation
- handlers: WebSocket message handlers
"""

from .server import (
    WebSocketServer,
    WebSocketConfig,
    ConnectionManager
)

from .handlers import (
    WebSocketMessageHandler,
    MessageType,
    WebSocketMessage
)

__all__ = [
    # Server
    'WebSocketServer',
    'WebSocketConfig',
    'ConnectionManager',

    # Handlers
    'WebSocketMessageHandler',
    'MessageType',
    'WebSocketMessage',
]
