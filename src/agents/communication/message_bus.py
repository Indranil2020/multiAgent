"""
Message Bus for Agent Communication.

This module provides a lightweight message bus for communication between
agents and swarm components. Supports publish-subscribe pattern and
direct messaging.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from .protocol import Message, MessageType, MessagePriority


@dataclass
class Subscription:
    """
    Subscription to message types.

    Attributes:
        subscriber_id: ID of subscriber
        message_type: Type of message to receive
        handler: Callback function for messages
        filter_criteria: Optional filter for messages
    """
    subscriber_id: str
    message_type: MessageType
    handler: Callable[[Message], None]
    filter_criteria: Optional[Dict[str, Any]] = None

    def matches(self, message: Message) -> bool:
        """
        Check if message matches this subscription.

        Args:
            message: Message to check

        Returns:
            True if message matches subscription criteria
        """
        # Check message type
        if message.message_type != self.message_type:
            return False

        # Check filter criteria if present
        if self.filter_criteria:
            for key, value in self.filter_criteria.items():
                if key == 'sender_id':
                    if message.sender_id != value:
                        return False
                elif key == 'priority':
                    if message.priority != value:
                        return False
                # Add more filter criteria as needed

        return True


class MessageBus:
    """
    Lightweight message bus for agent communication.

    Provides publish-subscribe messaging between agents and swarm components.
    Messages are delivered synchronously in the same process.

    Design:
    - In-memory, single-process message bus
    - Supports pub-sub and direct messaging
    - Priority-based message delivery
    - No persistence (ephemeral)
    """

    def __init__(self):
        """Initialize message bus."""
        self.subscriptions: Dict[MessageType, List[Subscription]] = defaultdict(list)
        self.message_queue: List[Message] = []
        self.delivered_count: int = 0
        self.dropped_count: int = 0

    def subscribe(
        self,
        subscriber_id: str,
        message_type: MessageType,
        handler: Callable[[Message], None],
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Subscribe to messages of a specific type.

        Args:
            subscriber_id: ID of subscribing component
            message_type: Type of messages to receive
            handler: Callback function to handle messages
            filter_criteria: Optional filter criteria

        Returns:
            True if subscription successful
        """
        if not subscriber_id or not subscriber_id.strip():
            return False

        if not handler or not callable(handler):
            return False

        # Create subscription
        subscription = Subscription(
            subscriber_id=subscriber_id,
            message_type=message_type,
            handler=handler,
            filter_criteria=filter_criteria
        )

        # Add to subscriptions
        self.subscriptions[message_type].append(subscription)

        return True

    def unsubscribe(
        self,
        subscriber_id: str,
        message_type: Optional[MessageType] = None
    ) -> int:
        """
        Unsubscribe from messages.

        Args:
            subscriber_id: ID of subscriber
            message_type: Specific message type to unsubscribe from (None for all)

        Returns:
            Number of subscriptions removed
        """
        if not subscriber_id or not subscriber_id.strip():
            return 0

        removed_count = 0

        if message_type is None:
            # Unsubscribe from all message types
            for msg_type in list(self.subscriptions.keys()):
                removed = self._remove_subscriptions(subscriber_id, msg_type)
                removed_count += removed
        else:
            # Unsubscribe from specific message type
            removed_count = self._remove_subscriptions(subscriber_id, message_type)

        return removed_count

    def publish(self, message: Message) -> bool:
        """
        Publish a message to all subscribers.

        Args:
            message: Message to publish

        Returns:
            True if message was published successfully
        """
        if message is None:
            return False

        if not message.validate():
            return False

        # Add to queue
        self.message_queue.append(message)

        # Sort queue by priority (higher priority first)
        self.message_queue.sort(
            key=lambda m: m.priority.value,
            reverse=True
        )

        # Process immediately (synchronous delivery)
        return self._deliver_message(message)

    def send(self, message: Message) -> bool:
        """
        Send a direct message to a specific recipient.

        Args:
            message: Message to send

        Returns:
            True if message was delivered
        """
        if message is None:
            return False

        if not message.validate():
            return False

        if message.recipient_id is None:
            return False

        # Deliver directly to matching subscribers
        return self._deliver_message(message)

    def process_queue(self) -> int:
        """
        Process queued messages.

        Returns:
            Number of messages processed
        """
        processed = 0

        while self.message_queue:
            message = self.message_queue.pop(0)
            if self._deliver_message(message):
                processed += 1

        return processed

    def get_stats(self) -> Dict[str, Any]:
        """
        Get message bus statistics.

        Returns:
            Statistics dictionary
        """
        total_subscriptions = sum(
            len(subs) for subs in self.subscriptions.values()
        )

        return {
            'total_subscriptions': total_subscriptions,
            'subscriptions_by_type': {
                msg_type.value: len(subs)
                for msg_type, subs in self.subscriptions.items()
            },
            'queued_messages': len(self.message_queue),
            'delivered_count': self.delivered_count,
            'dropped_count': self.dropped_count
        }

    def _deliver_message(self, message: Message) -> bool:
        """
        Deliver message to matching subscribers.

        Args:
            message: Message to deliver

        Returns:
            True if delivered to at least one subscriber
        """
        delivered = False

        # Get subscriptions for this message type
        subscriptions = self.subscriptions.get(message.message_type, [])

        # Deliver to matching subscribers
        for subscription in subscriptions:
            # Check if subscription matches
            if not subscription.matches(message):
                continue

            # Check if direct message is for this subscriber
            if message.recipient_id and message.recipient_id != subscription.subscriber_id:
                continue

            # Deliver message
            if self._invoke_handler(subscription.handler, message):
                delivered = True

        if delivered:
            self.delivered_count += 1
        else:
            self.dropped_count += 1

        return delivered

    def _invoke_handler(
        self,
        handler: Callable[[Message], None],
        message: Message
    ) -> bool:
        """
        Invoke handler with message.

        Args:
            handler: Handler function
            message: Message to deliver

        Returns:
            True if handler invoked successfully
        """
        if not handler or not callable(handler):
            return False

        if message is None:
            return False

        # Invoke handler (in production, might want timeout/error handling)
        handler(message)

        return True

    def _remove_subscriptions(
        self,
        subscriber_id: str,
        message_type: MessageType
    ) -> int:
        """
        Remove subscriptions for a subscriber.

        Args:
            subscriber_id: Subscriber to remove
            message_type: Message type

        Returns:
            Number of subscriptions removed
        """
        if message_type not in self.subscriptions:
            return 0

        original_count = len(self.subscriptions[message_type])

        # Filter out subscriptions for this subscriber
        self.subscriptions[message_type] = [
            sub for sub in self.subscriptions[message_type]
            if sub.subscriber_id != subscriber_id
        ]

        removed_count = original_count - len(self.subscriptions[message_type])

        # Clean up empty lists
        if not self.subscriptions[message_type]:
            del self.subscriptions[message_type]

        return removed_count
