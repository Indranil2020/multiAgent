"""
WebSocket Message Handlers.

This module provides handlers for processing incoming WebSocket messages
from clients.

Message Types:
- subscribe: Subscribe to topic
- unsubscribe: Unsubscribe from topic
- ping: Heartbeat ping
- submit_task: Submit new task
- get_task_status: Request task status
- get_agent_status: Request agent status
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class MessageType(str, Enum):
    """WebSocket message types."""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    SUBMIT_TASK = "submit_task"
    GET_TASK_STATUS = "get_task_status"
    GET_AGENT_STATUS = "get_agent_status"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class WebSocketMessage:
    """
    WebSocket message structure.

    Attributes:
        type: Message type
        payload: Message payload
        message_id: Optional message ID for request/response
    """
    type: str
    payload: Dict[str, Any]
    message_id: Optional[str] = None

    def validate(self) -> bool:
        """
        Validate message.

        Returns:
            True if valid
        """
        if not self.type:
            return False

        try:
            MessageType(self.type)
        except ValueError:
            return False

        if self.payload is None:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Returns:
            Dictionary representation
        """
        result = {
            'type': self.type,
            'payload': self.payload
        }

        if self.message_id:
            result['message_id'] = self.message_id

        return result

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Optional['WebSocketMessage']:
        """
        Create message from dictionary.

        Args:
            data: Dictionary data

        Returns:
            WebSocketMessage or None if invalid
        """
        if not isinstance(data, dict):
            return None

        if 'type' not in data or 'payload' not in data:
            return None

        message = WebSocketMessage(
            type=data['type'],
            payload=data['payload'],
            message_id=data.get('message_id')
        )

        if not message.validate():
            return None

        return message


class WebSocketMessageHandler:
    """
    Handler for WebSocket messages.

    This class processes incoming messages from clients and
    coordinates with system components.
    """

    def __init__(
        self,
        connection_manager=None,
        swarm_coordinator=None,
        task_distributor=None
    ):
        """
        Initialize message handler.

        Args:
            connection_manager: Connection manager
            swarm_coordinator: Agent swarm coordinator
            task_distributor: Task distributor
        """
        self.connection_manager = connection_manager
        self.swarm_coordinator = swarm_coordinator
        self.task_distributor = task_distributor

    def handle_message(
        self,
        connection_id: str,
        message_data: Dict[str, Any]
    ) -> Optional[WebSocketMessage]:
        """
        Handle incoming message.

        Args:
            connection_id: Connection identifier
            message_data: Message data

        Returns:
            Response message or None
        """
        # Parse message
        message = WebSocketMessage.from_dict(message_data)
        if not message:
            return self._create_error_response(
                "Invalid message format",
                message_id=message_data.get('message_id')
            )

        # Route to appropriate handler
        message_type = MessageType(message.type)

        if message_type == MessageType.SUBSCRIBE:
            return self._handle_subscribe(connection_id, message)

        elif message_type == MessageType.UNSUBSCRIBE:
            return self._handle_unsubscribe(connection_id, message)

        elif message_type == MessageType.PING:
            return self._handle_ping(connection_id, message)

        elif message_type == MessageType.SUBMIT_TASK:
            return self._handle_submit_task(connection_id, message)

        elif message_type == MessageType.GET_TASK_STATUS:
            return self._handle_get_task_status(connection_id, message)

        elif message_type == MessageType.GET_AGENT_STATUS:
            return self._handle_get_agent_status(connection_id, message)

        else:
            return self._create_error_response(
                f"Unknown message type: {message.type}",
                message_id=message.message_id
            )

    def _handle_subscribe(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle subscribe message.

        Args:
            connection_id: Connection ID
            message: Subscribe message

        Returns:
            Response message
        """
        if 'topic' not in message.payload:
            return self._create_error_response(
                "Missing topic in subscribe message",
                message_id=message.message_id
            )

        topic = message.payload['topic']

        if not self.connection_manager:
            return self._create_error_response(
                "Connection manager not available",
                message_id=message.message_id
            )

        success = self.connection_manager.subscribe(connection_id, topic)

        if success:
            return WebSocketMessage(
                type=MessageType.SUCCESS,
                payload={'topic': topic, 'subscribed': True},
                message_id=message.message_id
            )
        else:
            return self._create_error_response(
                f"Failed to subscribe to topic: {topic}",
                message_id=message.message_id
            )

    def _handle_unsubscribe(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle unsubscribe message.

        Args:
            connection_id: Connection ID
            message: Unsubscribe message

        Returns:
            Response message
        """
        if 'topic' not in message.payload:
            return self._create_error_response(
                "Missing topic in unsubscribe message",
                message_id=message.message_id
            )

        topic = message.payload['topic']

        if not self.connection_manager:
            return self._create_error_response(
                "Connection manager not available",
                message_id=message.message_id
            )

        success = self.connection_manager.unsubscribe(connection_id, topic)

        if success:
            return WebSocketMessage(
                type=MessageType.SUCCESS,
                payload={'topic': topic, 'unsubscribed': True},
                message_id=message.message_id
            )
        else:
            return self._create_error_response(
                f"Failed to unsubscribe from topic: {topic}",
                message_id=message.message_id
            )

    def _handle_ping(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle ping message.

        Args:
            connection_id: Connection ID
            message: Ping message

        Returns:
            Pong response
        """
        if self.connection_manager:
            self.connection_manager.update_heartbeat(connection_id)

        return WebSocketMessage(
            type=MessageType.PONG,
            payload={'timestamp': datetime.now().timestamp()},
            message_id=message.message_id
        )

    def _handle_submit_task(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle submit task message.

        Args:
            connection_id: Connection ID
            message: Submit task message

        Returns:
            Response message
        """
        # In production, would validate and submit task
        # For now, return success with mock task ID

        import hashlib
        task_id = hashlib.sha256(
            f"{connection_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        return WebSocketMessage(
            type=MessageType.SUCCESS,
            payload={
                'task_id': task_id,
                'status': 'pending',
                'message': 'Task submitted successfully'
            },
            message_id=message.message_id
        )

    def _handle_get_task_status(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle get task status message.

        Args:
            connection_id: Connection ID
            message: Get task status message

        Returns:
            Response message
        """
        if 'task_id' not in message.payload:
            return self._create_error_response(
                "Missing task_id in request",
                message_id=message.message_id
            )

        task_id = message.payload['task_id']

        # In production, would query task status
        # For now, return mock status

        return WebSocketMessage(
            type=MessageType.SUCCESS,
            payload={
                'task_id': task_id,
                'status': 'in_progress',
                'progress': 50
            },
            message_id=message.message_id
        )

    def _handle_get_agent_status(
        self,
        connection_id: str,
        message: WebSocketMessage
    ) -> WebSocketMessage:
        """
        Handle get agent status message.

        Args:
            connection_id: Connection ID
            message: Get agent status message

        Returns:
            Response message
        """
        if 'agent_id' not in message.payload:
            return self._create_error_response(
                "Missing agent_id in request",
                message_id=message.message_id
            )

        agent_id = message.payload['agent_id']

        # In production, would query agent status
        # For now, return mock status

        return WebSocketMessage(
            type=MessageType.SUCCESS,
            payload={
                'agent_id': agent_id,
                'status': 'busy',
                'task_id': 'mock_task_123'
            },
            message_id=message.message_id
        )

    def _create_error_response(
        self,
        error_message: str,
        message_id: Optional[str] = None
    ) -> WebSocketMessage:
        """
        Create error response message.

        Args:
            error_message: Error message
            message_id: Optional message ID

        Returns:
            Error message
        """
        return WebSocketMessage(
            type=MessageType.ERROR,
            payload={'error': error_message},
            message_id=message_id
        )
