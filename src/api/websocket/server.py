"""
WebSocket Server.

This module provides the WebSocket server implementation for real-time
bidirectional communication with clients.

Features:
- Connection management
- Message broadcasting
- Client subscriptions
- Automatic reconnection support
- Heartbeat/ping-pong
"""

from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class WebSocketConfig:
    """
    Configuration for WebSocket server.

    Attributes:
        host: Host to bind to
        port: Port to listen on
        max_connections: Maximum concurrent connections
        heartbeat_interval_seconds: Heartbeat interval
        message_queue_size: Maximum message queue size per client
        enable_compression: Enable message compression
    """
    host: str = "0.0.0.0"
    port: int = 8001
    max_connections: int = 1000
    heartbeat_interval_seconds: int = 30
    message_queue_size: int = 100
    enable_compression: bool = True

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid
        """
        if not (0 <= self.port <= 65535):
            return False

        if self.max_connections <= 0:
            return False

        if self.heartbeat_interval_seconds <= 0:
            return False

        if self.message_queue_size <= 0:
            return False

        return True


@dataclass
class WebSocketConnection:
    """
    Represents a WebSocket client connection.

    Attributes:
        connection_id: Unique connection identifier
        client_id: Optional client identifier
        connected_at: Connection timestamp
        last_heartbeat: Last heartbeat timestamp
        subscriptions: Set of subscription topics
        is_active: Whether connection is active
    """
    connection_id: str
    client_id: Optional[str] = None
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    subscriptions: Set[str] = field(default_factory=set)
    is_active: bool = True


class ConnectionManager:
    """
    Manages WebSocket connections.

    This class handles connection lifecycle, subscriptions,
    and message broadcasting.
    """

    def __init__(self, config: WebSocketConfig):
        """
        Initialize connection manager.

        Args:
            config: WebSocket configuration
        """
        if not config.validate():
            raise ValueError("Invalid WebSocket configuration")

        self.config = config
        self.connections: Dict[str, WebSocketConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> connection_ids

    def connect(
        self,
        connection_id: str,
        client_id: Optional[str] = None
    ) -> Optional[WebSocketConnection]:
        """
        Register a new connection.

        Args:
            connection_id: Unique connection identifier
            client_id: Optional client identifier

        Returns:
            WebSocketConnection or None if max connections reached
        """
        if len(self.connections) >= self.config.max_connections:
            return None

        if connection_id in self.connections:
            return None

        now = datetime.now().timestamp()
        connection = WebSocketConnection(
            connection_id=connection_id,
            client_id=client_id,
            connected_at=now,
            last_heartbeat=now,
            subscriptions=set(),
            is_active=True
        )

        self.connections[connection_id] = connection

        return connection

    def disconnect(self, connection_id: str) -> bool:
        """
        Disconnect a client.

        Args:
            connection_id: Connection identifier

        Returns:
            True if disconnected
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        # Remove from all subscriptions
        for topic in connection.subscriptions:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)

        # Remove connection
        del self.connections[connection_id]

        return True

    def subscribe(self, connection_id: str, topic: str) -> bool:
        """
        Subscribe connection to a topic.

        Args:
            connection_id: Connection identifier
            topic: Topic to subscribe to

        Returns:
            True if subscribed
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        connection.subscriptions.add(topic)

        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()

        self.subscriptions[topic].add(connection_id)

        return True

    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """
        Unsubscribe connection from a topic.

        Args:
            connection_id: Connection identifier
            topic: Topic to unsubscribe from

        Returns:
            True if unsubscribed
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        connection.subscriptions.discard(topic)

        if topic in self.subscriptions:
            self.subscriptions[topic].discard(connection_id)

        return True

    def broadcast(self, topic: str, message: Dict[str, Any]) -> int:
        """
        Broadcast message to all subscribers of a topic.

        Args:
            topic: Topic to broadcast to
            message: Message to broadcast

        Returns:
            Number of connections message was sent to
        """
        if topic not in self.subscriptions:
            return 0

        sent_count = 0

        for connection_id in self.subscriptions[topic]:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                if connection.is_active:
                    # In production, would send via websocket
                    # await websocket.send_json(message)
                    sent_count += 1

        return sent_count

    def send_to_connection(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Connection identifier
            message: Message to send

        Returns:
            True if sent
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        if not connection.is_active:
            return False

        # In production, would send via websocket
        # await websocket.send_json(message)

        return True

    def update_heartbeat(self, connection_id: str) -> bool:
        """
        Update connection heartbeat.

        Args:
            connection_id: Connection identifier

        Returns:
            True if updated
        """
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        connection.last_heartbeat = datetime.now().timestamp()

        return True

    def get_connection(self, connection_id: str) -> Optional[WebSocketConnection]:
        """
        Get connection by ID.

        Args:
            connection_id: Connection identifier

        Returns:
            WebSocketConnection or None
        """
        return self.connections.get(connection_id)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Statistics dictionary
        """
        active_connections = sum(
            1 for conn in self.connections.values()
            if conn.is_active
        )

        return {
            'total_connections': len(self.connections),
            'active_connections': active_connections,
            'total_subscriptions': sum(
                len(subs) for subs in self.subscriptions.values()
            ),
            'topics': len(self.subscriptions)
        }


class WebSocketServer:
    """
    WebSocket server implementation.

    This class provides the main WebSocket server that handles
    client connections and message routing.

    In production, this would integrate with FastAPI WebSocket:

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        # Handle connection
        # Process messages
        # Handle disconnection
    """

    def __init__(self, config: Optional[WebSocketConfig] = None):
        """
        Initialize WebSocket server.

        Args:
            config: WebSocket configuration
        """
        self.config = config or WebSocketConfig()
        self.connection_manager = ConnectionManager(self.config)
        self.is_running = False
        self.start_time = 0.0

        # Component references (injected)
        self.swarm_coordinator = None
        self.task_distributor = None
        self.message_handler = None

    def configure_dependencies(
        self,
        swarm_coordinator=None,
        task_distributor=None,
        message_handler=None
    ) -> bool:
        """
        Configure system component dependencies.

        Args:
            swarm_coordinator: Agent swarm coordinator
            task_distributor: Task distributor
            message_handler: WebSocket message handler

        Returns:
            True if dependencies configured
        """
        self.swarm_coordinator = swarm_coordinator
        self.task_distributor = task_distributor
        self.message_handler = message_handler

        return True

    def startup(self) -> bool:
        """
        Start the WebSocket server.

        Returns:
            True if startup successful
        """
        if self.is_running:
            return False

        self.start_time = datetime.now().timestamp()
        self.is_running = True

        return True

    def shutdown(self) -> bool:
        """
        Shutdown the WebSocket server.

        Returns:
            True if shutdown successful
        """
        if not self.is_running:
            return False

        # Disconnect all clients
        connection_ids = list(self.connection_manager.connections.keys())
        for connection_id in connection_ids:
            self.connection_manager.disconnect(connection_id)

        self.is_running = False

        return True

    def broadcast_task_update(
        self,
        task_id: str,
        status: str,
        progress: int
    ) -> int:
        """
        Broadcast task update to subscribers.

        Args:
            task_id: Task identifier
            status: Task status
            progress: Task progress (0-100)

        Returns:
            Number of connections notified
        """
        message = {
            'type': 'task_update',
            'task_id': task_id,
            'status': status,
            'progress': progress,
            'timestamp': datetime.now().timestamp()
        }

        return self.connection_manager.broadcast(f'task:{task_id}', message)

    def broadcast_agent_update(
        self,
        agent_id: str,
        status: str,
        task_id: Optional[str] = None
    ) -> int:
        """
        Broadcast agent status update.

        Args:
            agent_id: Agent identifier
            status: Agent status
            task_id: Optional task ID

        Returns:
            Number of connections notified
        """
        message = {
            'type': 'agent_update',
            'agent_id': agent_id,
            'status': status,
            'task_id': task_id,
            'timestamp': datetime.now().timestamp()
        }

        return self.connection_manager.broadcast('agents', message)

    def broadcast_red_flag(
        self,
        task_id: str,
        agent_id: str,
        flag_type: str,
        severity: int,
        description: str
    ) -> int:
        """
        Broadcast red flag event.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            flag_type: Type of red flag
            severity: Severity level
            description: Event description

        Returns:
            Number of connections notified
        """
        message = {
            'type': 'red_flag',
            'task_id': task_id,
            'agent_id': agent_id,
            'flag_type': flag_type,
            'severity': severity,
            'description': description,
            'timestamp': datetime.now().timestamp()
        }

        # Broadcast to task subscribers and red_flags topic
        count = self.connection_manager.broadcast(f'task:{task_id}', message)
        count += self.connection_manager.broadcast('red_flags', message)

        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get server statistics.

        Returns:
            Statistics dictionary
        """
        uptime = 0.0
        if self.is_running and self.start_time > 0:
            uptime = datetime.now().timestamp() - self.start_time

        conn_stats = self.connection_manager.get_stats()

        return {
            'is_running': self.is_running,
            'uptime_seconds': uptime,
            'connections': conn_stats
        }
