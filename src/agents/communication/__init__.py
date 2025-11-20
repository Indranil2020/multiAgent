"""
Agent Communication Module.

This module provides the communication infrastructure for agents including:
- Message protocol definitions
- Message serialization/deserialization
- Message bus for pub-sub communication

Components:
- protocol: Message types and protocol definitions
- serialization: JSON-based message serialization
- message_bus: Lightweight in-memory message bus
"""

from .protocol import (
    Message,
    MessageType,
    MessagePriority,
    TaskAssignmentPayload,
    TaskCompletionPayload,
    VoteSubmissionPayload,
    RedFlagPayload,
    HeartbeatPayload,
    ProtocolValidator
)

from .serialization import (
    MessageSerializer,
    PayloadSerializer,
    SerializationError
)

from .message_bus import (
    MessageBus,
    Subscription
)

__all__ = [
    # Protocol
    'Message',
    'MessageType',
    'MessagePriority',
    'TaskAssignmentPayload',
    'TaskCompletionPayload',
    'VoteSubmissionPayload',
    'RedFlagPayload',
    'HeartbeatPayload',
    'ProtocolValidator',

    # Serialization
    'MessageSerializer',
    'PayloadSerializer',
    'SerializationError',

    # Message Bus
    'MessageBus',
    'Subscription',
]
