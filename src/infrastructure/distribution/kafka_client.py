"""
Kafka client implementation.

This module provides a comprehensive Kafka client wrapper for distributed messaging,
event streaming, and pub/sub patterns with explicit error handling and producer/consumer support.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import time
import json


class MessageFormat(Enum):
    """Message serialization formats."""
    JSON = "json"
    STRING = "string"
    BYTES = "bytes"


@dataclass
class KafkaConfig:
    """
    Configuration for Kafka client.
    
    Attributes:
        bootstrap_servers: List of Kafka broker addresses
        client_id: Client identifier
        group_id: Consumer group ID
        auto_offset_reset: Where to start reading (earliest/latest)
        enable_auto_commit: Enable automatic offset commits
        session_timeout_ms: Session timeout in milliseconds
        max_poll_records: Maximum records per poll
        compression_type: Compression type (none/gzip/snappy/lz4)
    """
    bootstrap_servers: List[str]
    client_id: str
    group_id: str = "default_group"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    session_timeout_ms: int = 10000
    max_poll_records: int = 500
    compression_type: str = "none"
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if not self.bootstrap_servers:
            return (False, "bootstrap_servers cannot be empty")
        
        if not self.client_id:
            return (False, "client_id cannot be empty")
        
        if not self.group_id:
            return (False, "group_id cannot be empty")
        
        if self.auto_offset_reset not in ["earliest", "latest"]:
            return (False, "auto_offset_reset must be 'earliest' or 'latest'")
        
        if self.session_timeout_ms <= 0:
            return (False, "session_timeout_ms must be positive")
        
        if self.max_poll_records <= 0:
            return (False, "max_poll_records must be positive")
        
        if self.compression_type not in ["none", "gzip", "snappy", "lz4"]:
            return (False, "invalid compression_type")
        
        return (True, "")


@dataclass
class KafkaMessage:
    """
    A Kafka message.
    
    Attributes:
        topic: Topic name
        key: Message key
        value: Message value
        partition: Partition number
        offset: Message offset
        timestamp: Message timestamp
        headers: Message headers
    """
    topic: str
    key: Optional[str]
    value: Any
    partition: int = 0
    offset: int = 0
    timestamp: float = 0.0
    headers: Dict[str, str] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if message is valid."""
        return bool(self.topic and self.value is not None)


class KafkaClient:
    """
    Kafka client wrapper with explicit error handling.
    
    Provides producer and consumer functionality for Kafka messaging
    with support for topics, partitions, and consumer groups.
    
    Note: This is a mock implementation for the zero-error architecture.
    In production, this would wrap an actual Kafka client library.
    """
    
    def __init__(self, config: KafkaConfig):
        """
        Initialize Kafka client.
        
        Args:
            config: Kafka configuration
        """
        self.config = config
        self.connected = False
        
        # Mock storage (in-memory simulation)
        self.topics: Dict[str, List[KafkaMessage]] = {}
        self.subscriptions: List[str] = []
        self.consumer_offsets: Dict[str, int] = {}
        
        # Statistics
        self.messages_produced = 0
        self.messages_consumed = 0
        self.connection_time = 0.0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate client configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.config.validate()
    
    def connect(self) -> Tuple[bool, str]:
        """
        Connect to Kafka cluster.
        
        Returns:
            Tuple of (success, message)
        """
        # Validate config first
        is_valid, error_msg = self.validate_config()
        if not is_valid:
            return (False, f"Invalid config: {error_msg}")
        
        # Simulate connection
        self.connected = True
        self.connection_time = time.time()
        
        brokers = ", ".join(self.config.bootstrap_servers)
        return (True, f"Connected to Kafka cluster: {brokers}")
    
    def disconnect(self) -> Tuple[bool, str]:
        """
        Disconnect from Kafka cluster.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected")
        
        self.connected = False
        self.subscriptions.clear()
        
        return (True, "Disconnected from Kafka")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected
    
    # Producer operations
    
    def produce(
        self,
        topic: str,
        value: Any,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        partition: Optional[int] = None
    ) -> Tuple[bool, Optional[int], str]:
        """
        Produce a message to a topic.
        
        Args:
            topic: Topic name
            value: Message value
            key: Optional message key
            headers: Optional message headers
            partition: Optional partition number
        
        Returns:
            Tuple of (success, offset or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Kafka")
        
        if not topic:
            return (False, None, "topic cannot be empty")
        
        if value is None:
            return (False, None, "value cannot be None")
        
        # Create topic if it doesn't exist
        if topic not in self.topics:
            self.topics[topic] = []
        
        # Determine partition
        if partition is None:
            partition = 0
        
        # Get next offset
        offset = len(self.topics[topic])
        
        # Create message
        message = KafkaMessage(
            topic=topic,
            key=key,
            value=value,
            partition=partition,
            offset=offset,
            timestamp=time.time(),
            headers=headers or {}
        )
        
        if not message.is_valid():
            return (False, None, "Invalid message created")
        
        # Store message
        self.topics[topic].append(message)
        self.messages_produced += 1
        
        return (True, offset, f"Message produced to topic '{topic}' at offset {offset}")
    
    def produce_batch(
        self,
        topic: str,
        messages: List[Tuple[Any, Optional[str]]]
    ) -> Tuple[bool, int, str]:
        """
        Produce a batch of messages to a topic.
        
        Args:
            topic: Topic name
            messages: List of (value, key) tuples
        
        Returns:
            Tuple of (success, count_produced, message)
        """
        if not self.connected:
            return (False, 0, "Not connected to Kafka")
        
        if not topic:
            return (False, 0, "topic cannot be empty")
        
        if not messages:
            return (False, 0, "messages list cannot be empty")
        
        count = 0
        for value, key in messages:
            success, offset, msg = self.produce(topic, value, key)
            if success:
                count += 1
        
        return (True, count, f"Produced {count}/{len(messages)} messages")
    
    # Consumer operations
    
    def subscribe(self, topics: List[str]) -> Tuple[bool, str]:
        """
        Subscribe to topics.
        
        Args:
            topics: List of topic names
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        if not topics:
            return (False, "topics list cannot be empty")
        
        # Validate topics
        for topic in topics:
            if not topic:
                return (False, "topic name cannot be empty")
        
        self.subscriptions = topics.copy()
        
        # Initialize offsets
        for topic in topics:
            if topic not in self.consumer_offsets:
                if self.config.auto_offset_reset == "earliest":
                    self.consumer_offsets[topic] = 0
                else:
                    # Latest: start at end
                    self.consumer_offsets[topic] = len(self.topics.get(topic, []))
        
        return (True, f"Subscribed to {len(topics)} topics")
    
    def unsubscribe(self) -> Tuple[bool, str]:
        """
        Unsubscribe from all topics.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        topic_count = len(self.subscriptions)
        self.subscriptions.clear()
        
        return (True, f"Unsubscribed from {topic_count} topics")
    
    def poll(
        self,
        timeout_ms: int = 1000,
        max_records: Optional[int] = None
    ) -> Tuple[bool, List[KafkaMessage], str]:
        """
        Poll for messages from subscribed topics.
        
        Args:
            timeout_ms: Poll timeout in milliseconds
            max_records: Maximum records to return
        
        Returns:
            Tuple of (success, messages list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Kafka")
        
        if not self.subscriptions:
            return (False, [], "Not subscribed to any topics")
        
        if timeout_ms < 0:
            return (False, [], "timeout_ms cannot be negative")
        
        # Determine max records
        if max_records is None:
            max_records = self.config.max_poll_records
        
        messages = []
        
        # Poll from each subscribed topic
        for topic in self.subscriptions:
            if topic not in self.topics:
                continue
            
            topic_messages = self.topics[topic]
            offset = self.consumer_offsets.get(topic, 0)
            
            # Get messages from current offset
            available = topic_messages[offset:]
            to_consume = available[:max_records - len(messages)]
            
            messages.extend(to_consume)
            
            # Update offset
            self.consumer_offsets[topic] = offset + len(to_consume)
            
            # Check if we've reached max
            if len(messages) >= max_records:
                break
        
        self.messages_consumed += len(messages)
        
        return (True, messages, f"Polled {len(messages)} messages")
    
    def commit(self) -> Tuple[bool, str]:
        """
        Commit current offsets.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        if not self.config.enable_auto_commit:
            # Manual commit (no-op in mock)
            return (True, "Offsets committed")
        
        return (True, "Auto-commit enabled")
    
    def seek(self, topic: str, offset: int) -> Tuple[bool, str]:
        """
        Seek to a specific offset in a topic.
        
        Args:
            topic: Topic name
            offset: Offset to seek to
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        if not topic:
            return (False, "topic cannot be empty")
        
        if offset < 0:
            return (False, "offset cannot be negative")
        
        if topic not in self.subscriptions:
            return (False, f"Not subscribed to topic '{topic}'")
        
        # Validate offset
        if topic in self.topics:
            max_offset = len(self.topics[topic])
            if offset > max_offset:
                return (False, f"offset {offset} exceeds max {max_offset}")
        
        self.consumer_offsets[topic] = offset
        
        return (True, f"Seeked to offset {offset} in topic '{topic}'")
    
    # Topic management
    
    def create_topic(
        self,
        topic: str,
        num_partitions: int = 1,
        replication_factor: int = 1
    ) -> Tuple[bool, str]:
        """
        Create a new topic.
        
        Args:
            topic: Topic name
            num_partitions: Number of partitions
            replication_factor: Replication factor
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        if not topic:
            return (False, "topic cannot be empty")
        
        if num_partitions < 1:
            return (False, "num_partitions must be at least 1")
        
        if replication_factor < 1:
            return (False, "replication_factor must be at least 1")
        
        if topic in self.topics:
            return (False, f"Topic '{topic}' already exists")
        
        self.topics[topic] = []
        
        return (True, f"Topic '{topic}' created with {num_partitions} partitions")
    
    def delete_topic(self, topic: str) -> Tuple[bool, str]:
        """
        Delete a topic.
        
        Args:
            topic: Topic name
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        if not topic:
            return (False, "topic cannot be empty")
        
        if topic not in self.topics:
            return (False, f"Topic '{topic}' does not exist")
        
        message_count = len(self.topics[topic])
        self.topics.pop(topic)
        
        # Remove from subscriptions
        if topic in self.subscriptions:
            self.subscriptions.remove(topic)
        
        # Remove offset
        self.consumer_offsets.pop(topic, None)
        
        return (True, f"Topic '{topic}' deleted ({message_count} messages)")
    
    def list_topics(self) -> Tuple[bool, List[str], str]:
        """
        List all topics.
        
        Returns:
            Tuple of (success, topics list, message)
        """
        if not self.connected:
            return (False, [], "Not connected to Kafka")
        
        topics = list(self.topics.keys())
        
        return (True, topics, f"Found {len(topics)} topics")
    
    def get_topic_info(
        self,
        topic: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Get information about a topic.
        
        Args:
            topic: Topic name
        
        Returns:
            Tuple of (success, topic_info or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Kafka")
        
        if not topic:
            return (False, None, "topic cannot be empty")
        
        if topic not in self.topics:
            return (False, None, f"Topic '{topic}' does not exist")
        
        messages = self.topics[topic]
        
        info = {
            "topic": topic,
            "message_count": len(messages),
            "earliest_offset": 0,
            "latest_offset": len(messages),
            "subscribed": topic in self.subscriptions,
            "current_offset": self.consumer_offsets.get(topic, 0)
        }
        
        return (True, info, "Topic info retrieved")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with stats
        """
        total_messages = sum(len(msgs) for msgs in self.topics.values())
        
        return {
            "connected": self.connected,
            "client_id": self.config.client_id,
            "group_id": self.config.group_id,
            "total_topics": len(self.topics),
            "total_messages": total_messages,
            "messages_produced": self.messages_produced,
            "messages_consumed": self.messages_consumed,
            "subscribed_topics": len(self.subscriptions),
            "uptime_seconds": time.time() - self.connection_time if self.connected else 0
        }
    
    def flush(self) -> Tuple[bool, str]:
        """
        Flush all pending messages.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Kafka")
        
        # In mock, this is a no-op
        return (True, "Messages flushed")
