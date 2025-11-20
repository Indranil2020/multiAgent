"""
Communication Protocol Definitions.

This module defines the message types and protocols for agent communication
in the zero-error system. Messages are used for coordination between agents,
swarm components, and the voting system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime


class MessageType(Enum):
    """Types of messages in the system."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    VOTE_REQUEST = "vote_request"
    VOTE_SUBMISSION = "vote_submission"
    CONSENSUS_REACHED = "consensus_reached"
    RED_FLAG = "red_flag"
    AGENT_STATUS = "agent_status"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


class MessagePriority(Enum):
    """Message priority levels."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class Message:
    """
    Base message structure for agent communication.

    All messages in the system follow this structure for consistency
    and type safety.

    Attributes:
        message_id: Unique message identifier
        message_type: Type of message
        sender_id: ID of sending component
        recipient_id: ID of receiving component (None for broadcast)
        payload: Message payload data
        priority: Message priority
        timestamp: Message creation timestamp
        correlation_id: ID for correlating related messages
        reply_to: Message ID this is replying to
    """
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def validate(self) -> bool:
        """
        Validate message structure.

        Returns:
            True if message is valid
        """
        if not self.message_id or not self.message_id.strip():
            return False

        if not self.sender_id or not self.sender_id.strip():
            return False

        if not self.payload or not isinstance(self.payload, dict):
            return False

        if self.timestamp <= 0:
            return False

        return True

    def is_broadcast(self) -> bool:
        """Check if message is broadcast (no specific recipient)."""
        return self.recipient_id is None

    def is_reply(self) -> bool:
        """Check if message is a reply to another message."""
        return self.reply_to is not None

    def create_reply(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any]
    ) -> 'Message':
        """
        Create a reply message to this message.

        Args:
            sender_id: ID of the replying component
            message_type: Type of reply message
            payload: Reply payload

        Returns:
            New message that replies to this one
        """
        import hashlib

        reply_id = hashlib.sha256(
            f"{sender_id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        return Message(
            message_id=reply_id,
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=self.sender_id,  # Reply to original sender
            payload=payload,
            priority=self.priority,
            correlation_id=self.correlation_id or self.message_id,
            reply_to=self.message_id
        )


@dataclass
class TaskAssignmentPayload:
    """Payload for task assignment messages."""
    task_id: str
    task_spec: Dict[str, Any]
    agent_config: Dict[str, Any]
    deadline_timestamp: Optional[float] = None

    def validate(self) -> bool:
        """Validate payload."""
        if not self.task_id or not self.task_id.strip():
            return False
        if not self.task_spec or not isinstance(self.task_spec, dict):
            return False
        if not self.agent_config or not isinstance(self.agent_config, dict):
            return False
        if self.deadline_timestamp and self.deadline_timestamp <= 0:
            return False
        return True


@dataclass
class TaskCompletionPayload:
    """Payload for task completion messages."""
    task_id: str
    result: Any
    execution_time_ms: int
    quality_score: float

    def validate(self) -> bool:
        """Validate payload."""
        if not self.task_id or not self.task_id.strip():
            return False
        if self.execution_time_ms < 0:
            return False
        if not (0.0 <= self.quality_score <= 1.0):
            return False
        return True


@dataclass
class VoteSubmissionPayload:
    """Payload for vote submission messages."""
    task_id: str
    agent_id: str
    output: Any
    semantic_signature: str
    quality_score: float
    execution_time_ms: int

    def validate(self) -> bool:
        """Validate payload."""
        if not self.task_id or not self.task_id.strip():
            return False
        if not self.agent_id or not self.agent_id.strip():
            return False
        if not self.semantic_signature:
            return False
        if not (0.0 <= self.quality_score <= 1.0):
            return False
        if self.execution_time_ms < 0:
            return False
        return True


@dataclass
class RedFlagPayload:
    """Payload for red flag messages."""
    task_id: str
    agent_id: str
    flag_type: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate payload."""
        if not self.task_id or not self.task_id.strip():
            return False
        if not self.agent_id or not self.agent_id.strip():
            return False
        if not self.flag_type or not self.flag_type.strip():
            return False
        if not self.description or len(self.description) < 10:
            return False
        if self.severity not in ["low", "medium", "high", "critical"]:
            return False
        return True


@dataclass
class HeartbeatPayload:
    """Payload for heartbeat messages."""
    component_id: str
    status: str  # "healthy", "degraded", "failing"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate payload."""
        if not self.component_id or not self.component_id.strip():
            return False
        if self.status not in ["healthy", "degraded", "failing"]:
            return False
        return True


class ProtocolValidator:
    """
    Validator for protocol compliance.

    Ensures all messages follow the defined protocol and have valid payloads.
    """

    @staticmethod
    def validate_message(message: Message) -> bool:
        """
        Validate complete message including payload.

        Args:
            message: Message to validate

        Returns:
            True if message is valid
        """
        # Basic message validation
        if not message.validate():
            return False

        # Type-specific payload validation
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            return ProtocolValidator._validate_task_assignment(message.payload)
        elif message.message_type == MessageType.TASK_COMPLETION:
            return ProtocolValidator._validate_task_completion(message.payload)
        elif message.message_type == MessageType.VOTE_SUBMISSION:
            return ProtocolValidator._validate_vote_submission(message.payload)
        elif message.message_type == MessageType.RED_FLAG:
            return ProtocolValidator._validate_red_flag(message.payload)
        elif message.message_type == MessageType.HEARTBEAT:
            return ProtocolValidator._validate_heartbeat(message.payload)

        # Other message types have generic validation
        return True

    @staticmethod
    def _validate_task_assignment(payload: Dict[str, Any]) -> bool:
        """Validate task assignment payload."""
        required_keys = ['task_id', 'task_spec', 'agent_config']
        return all(key in payload for key in required_keys)

    @staticmethod
    def _validate_task_completion(payload: Dict[str, Any]) -> bool:
        """Validate task completion payload."""
        required_keys = ['task_id', 'result', 'execution_time_ms', 'quality_score']
        return all(key in payload for key in required_keys)

    @staticmethod
    def _validate_vote_submission(payload: Dict[str, Any]) -> bool:
        """Validate vote submission payload."""
        required_keys = [
            'task_id', 'agent_id', 'output',
            'semantic_signature', 'quality_score', 'execution_time_ms'
        ]
        return all(key in payload for key in required_keys)

    @staticmethod
    def _validate_red_flag(payload: Dict[str, Any]) -> bool:
        """Validate red flag payload."""
        required_keys = ['task_id', 'agent_id', 'flag_type', 'description', 'severity']
        return all(key in payload for key in required_keys)

    @staticmethod
    def _validate_heartbeat(payload: Dict[str, Any]) -> bool:
        """Validate heartbeat payload."""
        required_keys = ['component_id', 'status']
        return all(key in payload for key in required_keys)
