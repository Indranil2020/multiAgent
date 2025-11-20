"""
API Module.

This module provides all API interfaces for the zero-error system:
- REST API: FastAPI-based REST endpoints
- WebSocket API: Real-time bidirectional communication
- gRPC API: High-performance RPC services

Each API type provides access to:
- Task management
- Agent coordination
- Verification services
- System monitoring

Usage:
    from api.rest import create_application
    from api.websocket import WebSocketServer
    from api.grpc import GRPCServer

    # Create REST API
    rest_app = create_application()

    # Create WebSocket server
    ws_server = WebSocketServer()

    # Create gRPC server
    grpc_server = GRPCServer()
"""

from . import rest
from . import websocket
from . import grpc

__all__ = [
    'rest',
    'websocket',
    'grpc',
]
