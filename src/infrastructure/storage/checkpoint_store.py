"""
Checkpoint storage implementation.

This module provides comprehensive checkpoint management for long-running tasks,
enabling save/restore functionality, versioning, cleanup, and recovery support.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time
import json


class CheckpointType(Enum):
    """Types of checkpoints."""
    MANUAL = "manual"  # User-initiated checkpoint
    AUTO = "auto"  # Automatic periodic checkpoint
    RECOVERY = "recovery"  # Recovery point after failure
    MILESTONE = "milestone"  # Major milestone checkpoint


@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.
    
    Attributes:
        checkpoint_id: Unique checkpoint identifier
        task_id: Associated task ID
        checkpoint_type: Type of checkpoint
        created_at: Creation timestamp
        version: Checkpoint version
        description: Human-readable description
        tags: Tags for categorization
        size_bytes: Approximate size in bytes
    """
    checkpoint_id: str
    task_id: str
    checkpoint_type: CheckpointType
    created_at: float
    version: int
    description: str
    tags: List[str] = field(default_factory=list)
    size_bytes: int = 0
    
    def is_valid(self) -> bool:
        """Check if metadata is valid."""
        return bool(
            self.checkpoint_id and
            self.task_id and
            self.description
        )
    
    def age_seconds(self) -> float:
        """Get age of checkpoint in seconds."""
        return time.time() - self.created_at
    
    def has_tag(self, tag: str) -> bool:
        """Check if checkpoint has a specific tag."""
        return tag in self.tags


@dataclass
class Checkpoint:
    """
    A complete checkpoint with state and metadata.
    
    Attributes:
        metadata: Checkpoint metadata
        state: Saved state data
        progress_percent: Progress percentage (0-100)
        can_resume: Whether checkpoint supports resumption
    """
    metadata: CheckpointMetadata
    state: Dict[str, Any]
    progress_percent: float
    can_resume: bool = True
    
    def is_valid(self) -> bool:
        """Check if checkpoint is valid."""
        return (
            self.metadata.is_valid() and
            0 <= self.progress_percent <= 100
        )
    
    def estimate_size(self) -> int:
        """Estimate checkpoint size in bytes."""
        # Simple estimation based on JSON serialization
        state_str = json.dumps(self.state)
        return len(state_str.encode('utf-8'))


class CheckpointStore:
    """
    Comprehensive checkpoint storage and management.
    
    Provides checkpoint save/load, versioning, cleanup policies,
    and recovery support for long-running tasks.
    """
    
    def __init__(
        self,
        max_checkpoints_per_task: int = 10,
        max_age_seconds: float = 86400.0,  # 24 hours
        auto_cleanup: bool = True
    ):
        """
        Initialize checkpoint store.
        
        Args:
            max_checkpoints_per_task: Maximum checkpoints to keep per task
            max_age_seconds: Maximum age before cleanup
            auto_cleanup: Whether to auto-cleanup old checkpoints
        """
        self.max_checkpoints_per_task = max_checkpoints_per_task
        self.max_age_seconds = max_age_seconds
        self.auto_cleanup = auto_cleanup
        self.checkpoints: Dict[str, List[Checkpoint]] = {}
        self.checkpoint_counter = 0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate store configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.max_checkpoints_per_task < 1:
            return (False, "max_checkpoints_per_task must be at least 1")
        
        if self.max_age_seconds <= 0:
            return (False, "max_age_seconds must be positive")
        
        return (True, "")
    
    def save_checkpoint(
        self,
        task_id: str,
        state: Dict[str, Any],
        progress_percent: float,
        checkpoint_type: CheckpointType = CheckpointType.AUTO,
        description: str = "",
        tags: Optional[List[str]] = None,
        can_resume: bool = True
    ) -> Tuple[bool, Optional[str], str]:
        """
        Save a checkpoint for a task.
        
        Args:
            task_id: Task identifier
            state: State data to save
            progress_percent: Progress percentage (0-100)
            checkpoint_type: Type of checkpoint
            description: Human-readable description
            tags: Optional tags
            can_resume: Whether checkpoint supports resumption
        
        Returns:
            Tuple of (success, checkpoint_id or None, message)
        """
        # Validate inputs
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if not isinstance(state, dict):
            return (False, None, "state must be a dictionary")
        
        if not 0 <= progress_percent <= 100:
            return (False, None, "progress_percent must be between 0 and 100")
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(task_id)
        
        # Determine version
        version = self._get_next_version(task_id)
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            task_id=task_id,
            checkpoint_type=checkpoint_type,
            created_at=time.time(),
            version=version,
            description=description or f"Checkpoint at {progress_percent:.1f}%",
            tags=tags or [],
            size_bytes=0  # Will be updated
        )
        
        # Create checkpoint
        checkpoint = Checkpoint(
            metadata=metadata,
            state=state,
            progress_percent=progress_percent,
            can_resume=can_resume
        )
        
        if not checkpoint.is_valid():
            return (False, None, "Invalid checkpoint created")
        
        # Update size estimate
        checkpoint.metadata.size_bytes = checkpoint.estimate_size()
        
        # Store checkpoint
        if task_id not in self.checkpoints:
            self.checkpoints[task_id] = []
        
        self.checkpoints[task_id].append(checkpoint)
        self.checkpoint_counter += 1
        
        # Auto-cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_old_checkpoints(task_id)
        
        return (True, checkpoint_id, f"Checkpoint saved (version {version}, {checkpoint.metadata.size_bytes} bytes)")
    
    def load_checkpoint(
        self,
        task_id: str,
        version: Optional[int] = None,
        checkpoint_id: Optional[str] = None
    ) -> Tuple[bool, Optional[Checkpoint], str]:
        """
        Load a checkpoint by task ID and version or checkpoint ID.
        
        Args:
            task_id: Task identifier
            version: Specific version (None for latest)
            checkpoint_id: Specific checkpoint ID (overrides version)
        
        Returns:
            Tuple of (success, checkpoint or None, message)
        """
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if task_id not in self.checkpoints:
            return (False, None, f"No checkpoints for task {task_id}")
        
        checkpoints = self.checkpoints[task_id]
        
        if not checkpoints:
            return (False, None, f"No checkpoints for task {task_id}")
        
        # Search by checkpoint ID if provided
        if checkpoint_id:
            for checkpoint in checkpoints:
                if checkpoint.metadata.checkpoint_id == checkpoint_id:
                    return (True, checkpoint, f"Checkpoint {checkpoint_id} loaded")
            
            return (False, None, f"Checkpoint {checkpoint_id} not found")
        
        # Search by version
        if version is None:
            # Return latest version
            checkpoint = checkpoints[-1]
            return (True, checkpoint, f"Latest checkpoint loaded (version {checkpoint.metadata.version})")
        
        # Find specific version
        for checkpoint in checkpoints:
            if checkpoint.metadata.version == version:
                return (True, checkpoint, f"Checkpoint version {version} loaded")
        
        return (False, None, f"Version {version} not found for task {task_id}")
    
    def get_all_checkpoints(
        self,
        task_id: str
    ) -> Tuple[bool, List[Checkpoint], str]:
        """
        Get all checkpoints for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, checkpoints list, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if task_id not in self.checkpoints:
            return (True, [], f"No checkpoints for task {task_id}")
        
        checkpoints = self.checkpoints[task_id]
        return (True, checkpoints, f"Retrieved {len(checkpoints)} checkpoints")
    
    def get_latest_checkpoint(
        self,
        task_id: str
    ) -> Tuple[bool, Optional[Checkpoint], str]:
        """
        Get the latest checkpoint for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, checkpoint or None, message)
        """
        return self.load_checkpoint(task_id, version=None)
    
    def get_checkpoints_by_type(
        self,
        task_id: str,
        checkpoint_type: CheckpointType
    ) -> Tuple[bool, List[Checkpoint], str]:
        """
        Get checkpoints of a specific type.
        
        Args:
            task_id: Task identifier
            checkpoint_type: Type to filter by
        
        Returns:
            Tuple of (success, checkpoints list, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if task_id not in self.checkpoints:
            return (True, [], f"No checkpoints for task {task_id}")
        
        filtered = [
            cp for cp in self.checkpoints[task_id]
            if cp.metadata.checkpoint_type == checkpoint_type
        ]
        
        return (True, filtered, f"Found {len(filtered)} {checkpoint_type.value} checkpoints")
    
    def get_checkpoints_by_tag(
        self,
        task_id: str,
        tag: str
    ) -> Tuple[bool, List[Checkpoint], str]:
        """
        Get checkpoints with a specific tag.
        
        Args:
            task_id: Task identifier
            tag: Tag to filter by
        
        Returns:
            Tuple of (success, checkpoints list, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if not tag:
            return (False, [], "tag cannot be empty")
        
        if task_id not in self.checkpoints:
            return (True, [], f"No checkpoints for task {task_id}")
        
        filtered = [
            cp for cp in self.checkpoints[task_id]
            if cp.metadata.has_tag(tag)
        ]
        
        return (True, filtered, f"Found {len(filtered)} checkpoints with tag '{tag}'")
    
    def delete_checkpoint(
        self,
        task_id: str,
        version: Optional[int] = None,
        checkpoint_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Delete a specific checkpoint.
        
        Args:
            task_id: Task identifier
            version: Specific version
            checkpoint_id: Specific checkpoint ID (overrides version)
        
        Returns:
            Tuple of (success, message)
        """
        if not task_id:
            return (False, "task_id cannot be empty")
        
        if task_id not in self.checkpoints:
            return (False, f"No checkpoints for task {task_id}")
        
        checkpoints = self.checkpoints[task_id]
        
        # Delete by checkpoint ID
        if checkpoint_id:
            new_checkpoints = [
                cp for cp in checkpoints
                if cp.metadata.checkpoint_id != checkpoint_id
            ]
            
            if len(new_checkpoints) == len(checkpoints):
                return (False, f"Checkpoint {checkpoint_id} not found")
            
            self.checkpoints[task_id] = new_checkpoints
            return (True, f"Deleted checkpoint {checkpoint_id}")
        
        # Delete by version
        if version is not None:
            new_checkpoints = [
                cp for cp in checkpoints
                if cp.metadata.version != version
            ]
            
            if len(new_checkpoints) == len(checkpoints):
                return (False, f"Version {version} not found")
            
            self.checkpoints[task_id] = new_checkpoints
            return (True, f"Deleted checkpoint version {version}")
        
        return (False, "Must specify either version or checkpoint_id")
    
    def delete_all_checkpoints(self, task_id: str) -> Tuple[bool, str]:
        """
        Delete all checkpoints for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, message)
        """
        if not task_id:
            return (False, "task_id cannot be empty")
        
        if task_id not in self.checkpoints:
            return (True, f"No checkpoints for task {task_id}")
        
        count = len(self.checkpoints[task_id])
        self.checkpoints.pop(task_id)
        
        return (True, f"Deleted {count} checkpoints for task {task_id}")
    
    def cleanup_old_checkpoints(
        self,
        task_id: Optional[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Clean up old checkpoints based on age and count limits.
        
        Args:
            task_id: Specific task ID (None for all tasks)
        
        Returns:
            Tuple of (success, count_removed, message)
        """
        if task_id:
            # Clean up specific task
            if task_id not in self.checkpoints:
                return (True, 0, f"No checkpoints for task {task_id}")
            
            removed = self._cleanup_old_checkpoints(task_id)
            return (True, removed, f"Removed {removed} old checkpoints")
        
        # Clean up all tasks
        total_removed = 0
        for tid in list(self.checkpoints.keys()):
            removed = self._cleanup_old_checkpoints(tid)
            total_removed += removed
        
        return (True, total_removed, f"Removed {total_removed} old checkpoints across all tasks")
    
    def _cleanup_old_checkpoints(self, task_id: str) -> int:
        """
        Internal cleanup for a specific task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Number of checkpoints removed
        """
        if task_id not in self.checkpoints:
            return 0
        
        checkpoints = self.checkpoints[task_id]
        original_count = len(checkpoints)
        
        # Remove checkpoints older than max_age
        current_time = time.time()
        checkpoints = [
            cp for cp in checkpoints
            if (current_time - cp.metadata.created_at) < self.max_age_seconds
        ]
        
        # Keep only the most recent max_checkpoints_per_task
        if len(checkpoints) > self.max_checkpoints_per_task:
            # Sort by creation time and keep latest
            checkpoints.sort(key=lambda cp: cp.metadata.created_at)
            checkpoints = checkpoints[-self.max_checkpoints_per_task:]
        
        self.checkpoints[task_id] = checkpoints
        
        return original_count - len(checkpoints)
    
    def _generate_checkpoint_id(self, task_id: str) -> str:
        """Generate unique checkpoint ID."""
        self.checkpoint_counter += 1
        timestamp = int(time.time())
        return f"{task_id}_cp_{timestamp}_{self.checkpoint_counter}"
    
    def _get_next_version(self, task_id: str) -> int:
        """Get next version number for task."""
        if task_id not in self.checkpoints or not self.checkpoints[task_id]:
            return 1
        
        latest_version = max(cp.metadata.version for cp in self.checkpoints[task_id])
        return latest_version + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive checkpoint store statistics.
        
        Returns:
            Dictionary with detailed stats
        """
        total_checkpoints = sum(len(cps) for cps in self.checkpoints.values())
        total_size = sum(
            sum(cp.metadata.size_bytes for cp in cps)
            for cps in self.checkpoints.values()
        )
        
        # Count by type
        type_counts = {ct.value: 0 for ct in CheckpointType}
        for checkpoints in self.checkpoints.values():
            for cp in checkpoints:
                type_counts[cp.metadata.checkpoint_type.value] += 1
        
        # Average checkpoints per task
        avg_per_task = total_checkpoints / len(self.checkpoints) if self.checkpoints else 0
        
        return {
            "total_tasks_with_checkpoints": len(self.checkpoints),
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "average_checkpoints_per_task": avg_per_task,
            "checkpoints_by_type": type_counts,
            "max_checkpoints_per_task": self.max_checkpoints_per_task,
            "max_age_seconds": self.max_age_seconds,
            "auto_cleanup_enabled": self.auto_cleanup
        }
    
    def clear(self) -> None:
        """Clear all checkpoints from the store."""
        self.checkpoints.clear()
        self.checkpoint_counter = 0
