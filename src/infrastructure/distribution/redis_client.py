"""
Redis client implementation.

This module provides a comprehensive Redis client wrapper for caching, state management,
pub/sub messaging, and distributed data structures with explicit error handling.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import time
import json


class RedisDataType(Enum):
    """Redis data types."""
    STRING = "string"
    HASH = "hash"
    LIST = "list"
    SET = "set"
    SORTED_SET = "sorted_set"


@dataclass
class RedisConfig:
    """
    Configuration for Redis client.
    
    Attributes:
        host: Redis server host
        port: Redis server port
        db: Database number
        password: Optional password
        connection_timeout_seconds: Connection timeout
        operation_timeout_seconds: Operation timeout
        max_connections: Maximum connection pool size
        enable_retry: Enable automatic retry
        max_retries: Maximum retry attempts
    """
    host: str
    port: int
    db: int = 0
    password: str = ""
    connection_timeout_seconds: float = 5.0
    operation_timeout_seconds: float = 3.0
    max_connections: int = 50
    enable_retry: bool = True
    max_retries: int = 3
    
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if not self.host:
            return (False, "host cannot be empty")
        
        if self.port < 1 or self.port > 65535:
            return (False, "port must be between 1 and 65535")
        
        if self.db < 0:
            return (False, "db cannot be negative")
        
        if self.connection_timeout_seconds <= 0:
            return (False, "connection_timeout_seconds must be positive")
        
        if self.operation_timeout_seconds <= 0:
            return (False, "operation_timeout_seconds must be positive")
        
        if self.max_connections < 1:
            return (False, "max_connections must be at least 1")
        
        if self.max_retries < 0:
            return (False, "max_retries cannot be negative")
        
        return (True, "")


class RedisClient:
    """
    Redis client wrapper with explicit error handling.
    
    Provides a simplified interface to Redis operations including
    key-value storage, hashes, lists, sets, pub/sub, and TTL management.
    
    Note: This is a mock implementation for the zero-error architecture.
    In production, this would wrap an actual Redis client library.
    """
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis client.
        
        Args:
            config: Redis configuration
        """
        self.config = config
        self.connected = False
        
        # Mock storage (in-memory simulation)
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self.data_types: Dict[str, RedisDataType] = {}
        
        # Pub/sub simulation
        self.subscribers: Dict[str, List[Any]] = {}
        
        # Statistics
        self.operation_count = 0
        self.error_count = 0
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
        Connect to Redis server.
        
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
        
        return (True, f"Connected to Redis at {self.config.host}:{self.config.port}")
    
    def disconnect(self) -> Tuple[bool, str]:
        """
        Disconnect from Redis server.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected")
        
        self.connected = False
        
        return (True, "Disconnected from Redis")
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self.connected
    
    # String operations
    
    def set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Set a string value.
        
        Args:
            key: Key to set
            value: Value to store
            ttl_seconds: Optional time-to-live in seconds
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key:
            return (False, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        self.data[key] = value
        self.data_types[key] = RedisDataType.STRING
        
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                return (False, "ttl_seconds must be positive")
            self.expiry[key] = time.time() + ttl_seconds
        
        self.operation_count += 1
        
        return (True, f"Set key '{key}'")
    
    def get(self, key: str) -> Tuple[bool, Optional[str], str]:
        """
        Get a string value.
        
        Args:
            key: Key to retrieve
        
        Returns:
            Tuple of (success, value or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Redis")
        
        if not key:
            return (False, None, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        if key not in self.data:
            return (False, None, f"Key '{key}' not found")
        
        if self.data_types.get(key) != RedisDataType.STRING:
            return (False, None, f"Key '{key}' is not a string")
        
        value = self.data[key]
        self.operation_count += 1
        
        return (True, value, "Value retrieved")
    
    def delete(self, key: str) -> Tuple[bool, str]:
        """
        Delete a key.
        
        Args:
            key: Key to delete
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key:
            return (False, "key cannot be empty")
        
        if key not in self.data:
            return (False, f"Key '{key}' not found")
        
        self.data.pop(key)
        self.data_types.pop(key, None)
        self.expiry.pop(key, None)
        
        self.operation_count += 1
        
        return (True, f"Deleted key '{key}'")
    
    def exists(self, key: str) -> Tuple[bool, bool, str]:
        """
        Check if a key exists.
        
        Args:
            key: Key to check
        
        Returns:
            Tuple of (success, exists, message)
        """
        if not self.connected:
            return (False, False, "Not connected to Redis")
        
        if not key:
            return (False, False, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        exists = key in self.data
        self.operation_count += 1
        
        return (True, exists, f"Key exists: {exists}")
    
    def set_ttl(self, key: str, ttl_seconds: int) -> Tuple[bool, str]:
        """
        Set time-to-live for a key.
        
        Args:
            key: Key to set TTL for
            ttl_seconds: Time-to-live in seconds
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key:
            return (False, "key cannot be empty")
        
        if ttl_seconds <= 0:
            return (False, "ttl_seconds must be positive")
        
        if key not in self.data:
            return (False, f"Key '{key}' not found")
        
        self.expiry[key] = time.time() + ttl_seconds
        self.operation_count += 1
        
        return (True, f"TTL set for key '{key}'")
    
    def get_ttl(self, key: str) -> Tuple[bool, Optional[int], str]:
        """
        Get remaining time-to-live for a key.
        
        Args:
            key: Key to check
        
        Returns:
            Tuple of (success, ttl_seconds or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Redis")
        
        if not key:
            return (False, None, "key cannot be empty")
        
        if key not in self.data:
            return (False, None, f"Key '{key}' not found")
        
        if key not in self.expiry:
            return (True, None, "Key has no TTL")
        
        remaining = int(self.expiry[key] - time.time())
        remaining = max(0, remaining)
        
        self.operation_count += 1
        
        return (True, remaining, f"TTL: {remaining} seconds")
    
    # Hash operations
    
    def hset(
        self,
        key: str,
        field: str,
        value: str
    ) -> Tuple[bool, str]:
        """
        Set a hash field.
        
        Args:
            key: Hash key
            field: Field name
            value: Field value
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key or not field:
            return (False, "key and field cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        # Initialize hash if needed
        if key not in self.data:
            self.data[key] = {}
            self.data_types[key] = RedisDataType.HASH
        
        if self.data_types.get(key) != RedisDataType.HASH:
            return (False, f"Key '{key}' is not a hash")
        
        self.data[key][field] = value
        self.operation_count += 1
        
        return (True, f"Set hash field '{field}' in '{key}'")
    
    def hget(
        self,
        key: str,
        field: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        Get a hash field value.
        
        Args:
            key: Hash key
            field: Field name
        
        Returns:
            Tuple of (success, value or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Redis")
        
        if not key or not field:
            return (False, None, "key and field cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        if key not in self.data:
            return (False, None, f"Key '{key}' not found")
        
        if self.data_types.get(key) != RedisDataType.HASH:
            return (False, None, f"Key '{key}' is not a hash")
        
        hash_data = self.data[key]
        
        if field not in hash_data:
            return (False, None, f"Field '{field}' not found in hash '{key}'")
        
        value = hash_data[field]
        self.operation_count += 1
        
        return (True, value, "Field value retrieved")
    
    def hgetall(self, key: str) -> Tuple[bool, Optional[Dict[str, str]], str]:
        """
        Get all fields and values from a hash.
        
        Args:
            key: Hash key
        
        Returns:
            Tuple of (success, hash_data or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Redis")
        
        if not key:
            return (False, None, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        if key not in self.data:
            return (False, None, f"Key '{key}' not found")
        
        if self.data_types.get(key) != RedisDataType.HASH:
            return (False, None, f"Key '{key}' is not a hash")
        
        hash_data = self.data[key].copy()
        self.operation_count += 1
        
        return (True, hash_data, f"Retrieved {len(hash_data)} fields")
    
    # List operations
    
    def lpush(self, key: str, value: str) -> Tuple[bool, str]:
        """
        Push value to the left of a list.
        
        Args:
            key: List key
            value: Value to push
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key:
            return (False, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        # Initialize list if needed
        if key not in self.data:
            self.data[key] = []
            self.data_types[key] = RedisDataType.LIST
        
        if self.data_types.get(key) != RedisDataType.LIST:
            return (False, f"Key '{key}' is not a list")
        
        self.data[key].insert(0, value)
        self.operation_count += 1
        
        return (True, f"Pushed value to list '{key}'")
    
    def rpush(self, key: str, value: str) -> Tuple[bool, str]:
        """
        Push value to the right of a list.
        
        Args:
            key: List key
            value: Value to push
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        if not key:
            return (False, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        # Initialize list if needed
        if key not in self.data:
            self.data[key] = []
            self.data_types[key] = RedisDataType.LIST
        
        if self.data_types.get(key) != RedisDataType.LIST:
            return (False, f"Key '{key}' is not a list")
        
        self.data[key].append(value)
        self.operation_count += 1
        
        return (True, f"Pushed value to list '{key}'")
    
    def lrange(
        self,
        key: str,
        start: int,
        stop: int
    ) -> Tuple[bool, Optional[List[str]], str]:
        """
        Get a range of elements from a list.
        
        Args:
            key: List key
            start: Start index
            stop: Stop index
        
        Returns:
            Tuple of (success, elements or None, message)
        """
        if not self.connected:
            return (False, None, "Not connected to Redis")
        
        if not key:
            return (False, None, "key cannot be empty")
        
        # Clean up expired keys
        self._cleanup_expired()
        
        if key not in self.data:
            return (False, None, f"Key '{key}' not found")
        
        if self.data_types.get(key) != RedisDataType.LIST:
            return (False, None, f"Key '{key}' is not a list")
        
        list_data = self.data[key]
        
        # Handle negative indices
        if stop == -1:
            elements = list_data[start:]
        else:
            elements = list_data[start:stop+1]
        
        self.operation_count += 1
        
        return (True, elements, f"Retrieved {len(elements)} elements")
    
    # Pub/Sub operations
    
    def publish(
        self,
        channel: str,
        message: str
    ) -> Tuple[bool, int, str]:
        """
        Publish a message to a channel.
        
        Args:
            channel: Channel name
            message: Message to publish
        
        Returns:
            Tuple of (success, subscriber_count, message)
        """
        if not self.connected:
            return (False, 0, "Not connected to Redis")
        
        if not channel:
            return (False, 0, "channel cannot be empty")
        
        # Count subscribers (mock)
        subscriber_count = len(self.subscribers.get(channel, []))
        
        self.operation_count += 1
        
        return (True, subscriber_count, f"Published to {subscriber_count} subscribers")
    
    def _cleanup_expired(self) -> int:
        """
        Clean up expired keys.
        
        Returns:
            Number of keys removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry.items()
            if current_time >= expiry_time
        ]
        
        for key in expired_keys:
            self.data.pop(key, None)
            self.data_types.pop(key, None)
            self.expiry.pop(key, None)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary with stats
        """
        # Clean up first
        expired_count = self._cleanup_expired()
        
        return {
            "connected": self.connected,
            "host": self.config.host,
            "port": self.config.port,
            "db": self.config.db,
            "total_keys": len(self.data),
            "keys_with_ttl": len(self.expiry),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "uptime_seconds": time.time() - self.connection_time if self.connected else 0
        }
    
    def flush_db(self) -> Tuple[bool, str]:
        """
        Flush all keys from current database.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.connected:
            return (False, "Not connected to Redis")
        
        key_count = len(self.data)
        
        self.data.clear()
        self.expiry.clear()
        self.data_types.clear()
        
        return (True, f"Flushed {key_count} keys")
