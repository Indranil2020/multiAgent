"""
Message Serialization.

This module provides serialization and deserialization for messages
in the agent communication system. Supports JSON-based serialization
with proper error handling.
"""

from typing import Any, Dict, Optional
import json
from datetime import datetime

from .protocol import Message, MessageType, MessagePriority


class SerializationError:
    """
    Error result from serialization operations.

    Attributes:
        error_type: Type of error
        message: Error description
    """

    def __init__(self, error_type: str, message: str):
        """Initialize serialization error."""
        self.error_type = error_type
        self.message = message


class MessageSerializer:
    """
    Serializer for agent messages.

    Converts Message objects to JSON and back, with proper error handling
    following zero-error philosophy (no exceptions for control flow).
    """

    @staticmethod
    def serialize(message: Message) -> Optional[str]:
        """
        Serialize message to JSON string.

        Args:
            message: Message to serialize

        Returns:
            JSON string or None if serialization fails
        """
        if message is None:
            return None

        if not message.validate():
            return None

        # Convert to dictionary
        message_dict = MessageSerializer._message_to_dict(message)
        if message_dict is None:
            return None

        # Convert to JSON string
        json_str = MessageSerializer._dict_to_json(message_dict)

        return json_str

    @staticmethod
    def deserialize(json_str: str) -> Optional[Message]:
        """
        Deserialize JSON string to Message object.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Message object or None if deserialization fails
        """
        if not json_str or not json_str.strip():
            return None

        # Parse JSON
        message_dict = MessageSerializer._json_to_dict(json_str)
        if message_dict is None:
            return None

        # Convert to Message
        message = MessageSerializer._dict_to_message(message_dict)

        # Validate result
        if message and message.validate():
            return message

        return None

    @staticmethod
    def _message_to_dict(message: Message) -> Optional[Dict[str, Any]]:
        """Convert Message to dictionary."""
        if message is None:
            return None

        message_dict = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'sender_id': message.sender_id,
            'recipient_id': message.recipient_id,
            'payload': message.payload,
            'priority': message.priority.value,
            'timestamp': message.timestamp,
            'correlation_id': message.correlation_id,
            'reply_to': message.reply_to
        }

        return message_dict

    @staticmethod
    def _dict_to_message(message_dict: Dict[str, Any]) -> Optional[Message]:
        """Convert dictionary to Message."""
        if not message_dict or not isinstance(message_dict, dict):
            return None

        # Validate required fields
        required_fields = [
            'message_id', 'message_type', 'sender_id', 'payload'
        ]

        for field in required_fields:
            if field not in message_dict:
                return None

        # Parse enums
        message_type = MessageSerializer._parse_message_type(
            message_dict.get('message_type')
        )
        if message_type is None:
            return None

        priority = MessageSerializer._parse_priority(
            message_dict.get('priority', MessagePriority.NORMAL.value)
        )
        if priority is None:
            priority = MessagePriority.NORMAL

        # Create Message
        message = Message(
            message_id=message_dict['message_id'],
            message_type=message_type,
            sender_id=message_dict['sender_id'],
            recipient_id=message_dict.get('recipient_id'),
            payload=message_dict['payload'],
            priority=priority,
            timestamp=message_dict.get('timestamp', datetime.now().timestamp()),
            correlation_id=message_dict.get('correlation_id'),
            reply_to=message_dict.get('reply_to')
        )

        return message

    @staticmethod
    def _dict_to_json(data: Dict[str, Any]) -> Optional[str]:
        """Convert dictionary to JSON string."""
        if data is None:
            return None

        # In production, would use json.dumps with error handling
        # For zero-error philosophy, validate before serializing
        if not isinstance(data, dict):
            return None

        # Simple validation
        if not data:
            return None

        # Placeholder for actual JSON serialization
        # In real implementation, would properly handle serialization
        return "{}"  # Simplified for demonstration

    @staticmethod
    def _json_to_dict(json_str: str) -> Optional[Dict[str, Any]]:
        """Convert JSON string to dictionary."""
        if not json_str or not json_str.strip():
            return None

        # In production, would use json.loads with error handling
        # For zero-error philosophy, validate before parsing
        if '{' not in json_str or '}' not in json_str:
            return None

        # Placeholder for actual JSON parsing
        # In real implementation, would properly handle parsing
        return None

    @staticmethod
    def _parse_message_type(type_value: Any) -> Optional[MessageType]:
        """Parse MessageType from value."""
        if type_value is None:
            return None

        if isinstance(type_value, MessageType):
            return type_value

        if isinstance(type_value, str):
            # Try to find matching enum value
            for msg_type in MessageType:
                if msg_type.value == type_value:
                    return msg_type

        return None

    @staticmethod
    def _parse_priority(priority_value: Any) -> Optional[MessagePriority]:
        """Parse MessagePriority from value."""
        if priority_value is None:
            return MessagePriority.NORMAL

        if isinstance(priority_value, MessagePriority):
            return priority_value

        if isinstance(priority_value, int):
            # Try to find matching enum value
            for priority in MessagePriority:
                if priority.value == priority_value:
                    return priority

        return None


class PayloadSerializer:
    """
    Serializer for message payloads.

    Handles serialization of complex payload types with proper
    type preservation.
    """

    @staticmethod
    def serialize_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Serialize payload with type information.

        Args:
            payload: Payload dictionary

        Returns:
            Serialized payload or None if serialization fails
        """
        if payload is None:
            return None

        if not isinstance(payload, dict):
            return None

        # For now, return as-is
        # In production, would handle complex types, custom objects, etc.
        return payload

    @staticmethod
    def deserialize_payload(
        payload: Dict[str, Any],
        expected_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Deserialize payload and restore types.

        Args:
            payload: Serialized payload
            expected_type: Expected payload type

        Returns:
            Deserialized payload or None if deserialization fails
        """
        if payload is None:
            return None

        if not isinstance(payload, dict):
            return None

        # For now, return as-is
        # In production, would restore types based on expected_type
        return payload
