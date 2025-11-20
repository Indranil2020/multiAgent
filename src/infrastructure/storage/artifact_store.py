"""
Artifact storage implementation.

This module provides comprehensive artifact management for generated code, documentation,
and other build artifacts, with content addressing, versioning, metadata tracking,
and efficient storage/retrieval mechanisms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import time
import hashlib
import json


class ArtifactType(Enum):
    """Types of artifacts."""
    SOURCE_CODE = "source_code"
    DOCUMENTATION = "documentation"
    TEST_FILE = "test_file"
    CONFIGURATION = "configuration"
    BINARY = "binary"
    DATA_FILE = "data_file"
    REPORT = "report"
    OTHER = "other"


class ArtifactStatus(Enum):
    """Status of an artifact."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ArtifactMetadata:
    """
    Metadata for an artifact.
    
    Attributes:
        artifact_id: Unique artifact identifier
        name: Artifact name
        artifact_type: Type of artifact
        status: Current status
        content_hash: SHA-256 hash of content
        size_bytes: Size in bytes
        created_at: Creation timestamp
        updated_at: Last update timestamp
        version: Artifact version
        tags: Tags for categorization
        author: Author identifier
        description: Human-readable description
        parent_id: Parent artifact ID (for versioning)
        related_task_id: Associated task ID
    """
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    status: ArtifactStatus
    content_hash: str
    size_bytes: int
    created_at: float
    updated_at: float
    version: int
    tags: List[str] = field(default_factory=list)
    author: str = ""
    description: str = ""
    parent_id: str = ""
    related_task_id: str = ""
    
    def is_valid(self) -> bool:
        """Check if metadata is valid."""
        return bool(
            self.artifact_id and
            self.name and
            self.content_hash
        )
    
    def age_seconds(self) -> float:
        """Get age of artifact in seconds."""
        return time.time() - self.created_at
    
    def has_tag(self, tag: str) -> bool:
        """Check if artifact has a specific tag."""
        return tag in self.tags
    
    def add_tag(self, tag: str) -> bool:
        """Add a tag to the artifact."""
        if not tag or tag in self.tags:
            return False
        self.tags.append(tag)
        return True
    
    def remove_tag(self, tag: str) -> bool:
        """Remove a tag from the artifact."""
        if tag not in self.tags:
            return False
        self.tags.remove(tag)
        return True


@dataclass
class Artifact:
    """
    A complete artifact with content and metadata.
    
    Attributes:
        metadata: Artifact metadata
        content: Artifact content (bytes or string)
        dependencies: List of dependent artifact IDs
        custom_metadata: Custom key-value metadata
    """
    metadata: ArtifactMetadata
    content: Any
    dependencies: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, str] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if artifact is valid."""
        return self.metadata.is_valid() and self.content is not None
    
    def get_content_as_string(self) -> Tuple[bool, str, str]:
        """
        Get content as string.
        
        Returns:
            Tuple of (success, content_string, message)
        """
        if isinstance(self.content, str):
            return (True, self.content, "Content is string")
        
        if isinstance(self.content, bytes):
            decoded = self.content.decode('utf-8', errors='replace')
            return (True, decoded, "Content decoded from bytes")
        
        # Try to convert to string
        content_str = str(self.content)
        return (True, content_str, "Content converted to string")
    
    def get_content_as_bytes(self) -> Tuple[bool, bytes, str]:
        """
        Get content as bytes.
        
        Returns:
            Tuple of (success, content_bytes, message)
        """
        if isinstance(self.content, bytes):
            return (True, self.content, "Content is bytes")
        
        if isinstance(self.content, str):
            encoded = self.content.encode('utf-8')
            return (True, encoded, "Content encoded to bytes")
        
        # Try to convert to bytes
        content_str = str(self.content)
        content_bytes = content_str.encode('utf-8')
        return (True, content_bytes, "Content converted to bytes")


class ArtifactStore:
    """
    Comprehensive artifact storage and management.
    
    Provides content-addressed storage, versioning, metadata tracking,
    dependency management, and efficient retrieval for generated artifacts.
    """
    
    def __init__(
        self,
        enable_deduplication: bool = True,
        max_artifact_size_mb: float = 100.0,
        enable_compression: bool = False
    ):
        """
        Initialize artifact store.
        
        Args:
            enable_deduplication: Enable content-based deduplication
            max_artifact_size_mb: Maximum artifact size in MB
            enable_compression: Enable content compression (not implemented)
        """
        self.enable_deduplication = enable_deduplication
        self.max_artifact_size_mb = max_artifact_size_mb
        self.enable_compression = enable_compression
        
        # Storage: artifact_id -> Artifact
        self.artifacts: Dict[str, Artifact] = {}
        
        # Content-addressed storage: content_hash -> artifact_id
        self.content_index: Dict[str, str] = {}
        
        # Name index: name -> List[artifact_id]
        self.name_index: Dict[str, List[str]] = {}
        
        # Tag index: tag -> Set[artifact_id]
        self.tag_index: Dict[str, Set[str]] = {}
        
        # Task index: task_id -> List[artifact_id]
        self.task_index: Dict[str, List[str]] = {}
        
        self.artifact_counter = 0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate store configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.max_artifact_size_mb <= 0:
            return (False, "max_artifact_size_mb must be positive")
        
        return (True, "")
    
    def save_artifact(
        self,
        name: str,
        content: Any,
        artifact_type: ArtifactType,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: str = "",
        related_task_id: str = "",
        dependencies: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Save an artifact to the store.
        
        Args:
            name: Artifact name
            content: Artifact content
            artifact_type: Type of artifact
            description: Human-readable description
            tags: Optional tags
            author: Author identifier
            related_task_id: Associated task ID
            dependencies: List of dependent artifact IDs
            custom_metadata: Custom metadata
        
        Returns:
            Tuple of (success, artifact_id or None, message)
        """
        # Validate inputs
        if not name:
            return (False, None, "name cannot be empty")
        
        if content is None:
            return (False, None, "content cannot be None")
        
        # Calculate content hash
        content_hash = self._calculate_content_hash(content)
        
        # Check for deduplication
        if self.enable_deduplication and content_hash in self.content_index:
            existing_id = self.content_index[content_hash]
            return (True, existing_id, f"Artifact already exists (deduplicated): {existing_id}")
        
        # Calculate size
        size_bytes = self._calculate_size(content)
        max_size_bytes = int(self.max_artifact_size_mb * 1024 * 1024)
        
        if size_bytes > max_size_bytes:
            return (False, None, f"Artifact size ({size_bytes} bytes) exceeds limit ({max_size_bytes} bytes)")
        
        # Generate artifact ID
        artifact_id = self._generate_artifact_id(name)
        
        # Determine version
        version = self._get_next_version(name)
        
        # Create metadata
        current_time = time.time()
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            status=ArtifactStatus.DRAFT,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=current_time,
            updated_at=current_time,
            version=version,
            tags=tags or [],
            author=author,
            description=description,
            related_task_id=related_task_id
        )
        
        # Create artifact
        artifact = Artifact(
            metadata=metadata,
            content=content,
            dependencies=dependencies or [],
            custom_metadata=custom_metadata or {}
        )
        
        if not artifact.is_valid():
            return (False, None, "Invalid artifact created")
        
        # Store artifact
        self.artifacts[artifact_id] = artifact
        
        # Update indices
        self.content_index[content_hash] = artifact_id
        
        if name not in self.name_index:
            self.name_index[name] = []
        self.name_index[name].append(artifact_id)
        
        for tag in metadata.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(artifact_id)
        
        if related_task_id:
            if related_task_id not in self.task_index:
                self.task_index[related_task_id] = []
            self.task_index[related_task_id].append(artifact_id)
        
        self.artifact_counter += 1
        
        return (True, artifact_id, f"Artifact saved (version {version}, {size_bytes} bytes)")
    
    def get_artifact(
        self,
        artifact_id: str
    ) -> Tuple[bool, Optional[Artifact], str]:
        """
        Retrieve an artifact by ID.
        
        Args:
            artifact_id: Artifact identifier
        
        Returns:
            Tuple of (success, artifact or None, message)
        """
        if not artifact_id:
            return (False, None, "artifact_id cannot be empty")
        
        if artifact_id not in self.artifacts:
            return (False, None, f"Artifact {artifact_id} not found")
        
        artifact = self.artifacts[artifact_id]
        return (True, artifact, "Artifact retrieved")
    
    def get_artifact_by_name(
        self,
        name: str,
        version: Optional[int] = None
    ) -> Tuple[bool, Optional[Artifact], str]:
        """
        Retrieve an artifact by name and optional version.
        
        Args:
            name: Artifact name
            version: Specific version (None for latest)
        
        Returns:
            Tuple of (success, artifact or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if name not in self.name_index:
            return (False, None, f"No artifacts with name '{name}'")
        
        artifact_ids = self.name_index[name]
        
        if not artifact_ids:
            return (False, None, f"No artifacts with name '{name}'")
        
        # Get artifacts
        artifacts = [self.artifacts[aid] for aid in artifact_ids if aid in self.artifacts]
        
        if version is None:
            # Return latest version
            latest = max(artifacts, key=lambda a: a.metadata.version)
            return (True, latest, f"Latest version retrieved (v{latest.metadata.version})")
        
        # Find specific version
        for artifact in artifacts:
            if artifact.metadata.version == version:
                return (True, artifact, f"Version {version} retrieved")
        
        return (False, None, f"Version {version} not found for '{name}'")
    
    def get_artifacts_by_tag(
        self,
        tag: str
    ) -> Tuple[bool, List[Artifact], str]:
        """
        Get all artifacts with a specific tag.
        
        Args:
            tag: Tag to filter by
        
        Returns:
            Tuple of (success, artifacts list, message)
        """
        if not tag:
            return (False, [], "tag cannot be empty")
        
        if tag not in self.tag_index:
            return (True, [], f"No artifacts with tag '{tag}'")
        
        artifact_ids = self.tag_index[tag]
        artifacts = [self.artifacts[aid] for aid in artifact_ids if aid in self.artifacts]
        
        return (True, artifacts, f"Found {len(artifacts)} artifacts with tag '{tag}'")
    
    def get_artifacts_by_task(
        self,
        task_id: str
    ) -> Tuple[bool, List[Artifact], str]:
        """
        Get all artifacts related to a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, artifacts list, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if task_id not in self.task_index:
            return (True, [], f"No artifacts for task {task_id}")
        
        artifact_ids = self.task_index[task_id]
        artifacts = [self.artifacts[aid] for aid in artifact_ids if aid in self.artifacts]
        
        return (True, artifacts, f"Found {len(artifacts)} artifacts for task {task_id}")
    
    def get_artifacts_by_type(
        self,
        artifact_type: ArtifactType
    ) -> List[Artifact]:
        """
        Get all artifacts of a specific type.
        
        Args:
            artifact_type: Type to filter by
        
        Returns:
            List of matching artifacts
        """
        return [
            artifact for artifact in self.artifacts.values()
            if artifact.metadata.artifact_type == artifact_type
        ]
    
    def update_artifact_status(
        self,
        artifact_id: str,
        status: ArtifactStatus
    ) -> Tuple[bool, str]:
        """
        Update artifact status.
        
        Args:
            artifact_id: Artifact identifier
            status: New status
        
        Returns:
            Tuple of (success, message)
        """
        if not artifact_id:
            return (False, "artifact_id cannot be empty")
        
        if artifact_id not in self.artifacts:
            return (False, f"Artifact {artifact_id} not found")
        
        artifact = self.artifacts[artifact_id]
        artifact.metadata.status = status
        artifact.metadata.updated_at = time.time()
        
        return (True, f"Artifact status updated to {status.value}")
    
    def add_tag_to_artifact(
        self,
        artifact_id: str,
        tag: str
    ) -> Tuple[bool, str]:
        """
        Add a tag to an artifact.
        
        Args:
            artifact_id: Artifact identifier
            tag: Tag to add
        
        Returns:
            Tuple of (success, message)
        """
        if not artifact_id:
            return (False, "artifact_id cannot be empty")
        
        if not tag:
            return (False, "tag cannot be empty")
        
        if artifact_id not in self.artifacts:
            return (False, f"Artifact {artifact_id} not found")
        
        artifact = self.artifacts[artifact_id]
        
        if not artifact.metadata.add_tag(tag):
            return (False, f"Tag '{tag}' already exists or invalid")
        
        # Update tag index
        if tag not in self.tag_index:
            self.tag_index[tag] = set()
        self.tag_index[tag].add(artifact_id)
        
        artifact.metadata.updated_at = time.time()
        
        return (True, f"Tag '{tag}' added")
    
    def delete_artifact(
        self,
        artifact_id: str
    ) -> Tuple[bool, str]:
        """
        Delete an artifact from the store.
        
        Args:
            artifact_id: Artifact identifier
        
        Returns:
            Tuple of (success, message)
        """
        if not artifact_id:
            return (False, "artifact_id cannot be empty")
        
        if artifact_id not in self.artifacts:
            return (False, f"Artifact {artifact_id} not found")
        
        artifact = self.artifacts.pop(artifact_id)
        
        # Update indices
        content_hash = artifact.metadata.content_hash
        if content_hash in self.content_index:
            self.content_index.pop(content_hash)
        
        name = artifact.metadata.name
        if name in self.name_index:
            self.name_index[name] = [aid for aid in self.name_index[name] if aid != artifact_id]
            if not self.name_index[name]:
                self.name_index.pop(name)
        
        for tag in artifact.metadata.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(artifact_id)
                if not self.tag_index[tag]:
                    self.tag_index.pop(tag)
        
        task_id = artifact.metadata.related_task_id
        if task_id and task_id in self.task_index:
            self.task_index[task_id] = [aid for aid in self.task_index[task_id] if aid != artifact_id]
            if not self.task_index[task_id]:
                self.task_index.pop(task_id)
        
        return (True, f"Artifact {artifact_id} deleted")
    
    def _calculate_content_hash(self, content: Any) -> str:
        """Calculate SHA-256 hash of content."""
        if isinstance(content, bytes):
            content_bytes = content
        elif isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_str = str(content)
            content_bytes = content_str.encode('utf-8')
        
        return hashlib.sha256(content_bytes).hexdigest()
    
    def _calculate_size(self, content: Any) -> int:
        """Calculate size of content in bytes."""
        if isinstance(content, bytes):
            return len(content)
        elif isinstance(content, str):
            return len(content.encode('utf-8'))
        else:
            content_str = str(content)
            return len(content_str.encode('utf-8'))
    
    def _generate_artifact_id(self, name: str) -> str:
        """Generate unique artifact ID."""
        self.artifact_counter += 1
        timestamp = int(time.time())
        # Sanitize name for ID
        safe_name = name.replace(' ', '_').replace('/', '_')[:50]
        return f"artifact_{safe_name}_{timestamp}_{self.artifact_counter}"
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number for artifact name."""
        if name not in self.name_index or not self.name_index[name]:
            return 1
        
        artifact_ids = self.name_index[name]
        artifacts = [self.artifacts[aid] for aid in artifact_ids if aid in self.artifacts]
        
        if not artifacts:
            return 1
        
        latest_version = max(a.metadata.version for a in artifacts)
        return latest_version + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive artifact store statistics.
        
        Returns:
            Dictionary with detailed stats
        """
        total_size = sum(a.metadata.size_bytes for a in self.artifacts.values())
        
        # Count by type
        type_counts = {at.value: 0 for at in ArtifactType}
        for artifact in self.artifacts.values():
            type_counts[artifact.metadata.artifact_type.value] += 1
        
        # Count by status
        status_counts = {st.value: 0 for st in ArtifactStatus}
        for artifact in self.artifacts.values():
            status_counts[artifact.metadata.status.value] += 1
        
        return {
            "total_artifacts": len(self.artifacts),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "unique_names": len(self.name_index),
            "total_tags": len(self.tag_index),
            "artifacts_by_type": type_counts,
            "artifacts_by_status": status_counts,
            "deduplication_enabled": self.enable_deduplication,
            "deduplicated_content_hashes": len(self.content_index)
        }
    
    def clear(self) -> None:
        """Clear all artifacts from the store."""
        self.artifacts.clear()
        self.content_index.clear()
        self.name_index.clear()
        self.tag_index.clear()
        self.task_index.clear()
        self.artifact_counter = 0
