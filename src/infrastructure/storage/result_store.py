"""
Result storage implementation.

This module provides comprehensive persistent storage for execution results,
enabling result retrieval, versioning, querying, and detailed analytics
with thread-safe operations and efficient indexing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import time
import threading


class ResultStatus(Enum):
    """Status of a stored result."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class StoredResult:
    """
    An execution result stored in the result store.
    
    Attributes:
        result_id: Unique result identifier
        task_id: Associated task identifier
        status: Result status
        output: Result output data
        error_message: Error message if failed
        version: Result version
        created_at: Creation timestamp
        execution_time_seconds: Execution duration
        metadata: Additional metadata
        tags: Result tags
        metrics: Performance metrics
    """
    result_id: str
    task_id: str
    status: ResultStatus
    output: Any
    error_message: str
    version: int
    created_at: float
    execution_time_seconds: float
    metadata: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if stored result is valid."""
        return bool(self.result_id and self.task_id)
    
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self.status == ResultStatus.SUCCESS
    
    def age_seconds(self) -> float:
        """Get result age in seconds."""
        return time.time() - self.created_at
    
    def has_tag(self, tag: str) -> bool:
        """Check if result has a specific tag."""
        return tag in self.tags


class ResultStore:
    """
    Comprehensive result storage with versioning and analytics.
    
    Provides thread-safe storage and retrieval of execution results
    with versioning support, advanced querying capabilities, and
    detailed statistics collection.
    """
    
    def __init__(self, max_results: int = 100000, max_versions_per_task: int = 10):
        """
        Initialize result store.
        
        Args:
            max_results: Maximum number of results to store
            max_versions_per_task: Maximum versions to keep per task
        """
        self.max_results = max_results
        self.max_versions_per_task = max_versions_per_task
        self.results: Dict[str, StoredResult] = {}
        self.result_counter = 0
        
        # Indices for efficient querying
        self.task_index: Dict[str, List[str]] = {}  # task_id -> [result_ids]
        self.status_index: Dict[ResultStatus, Set[str]] = {status: set() for status in ResultStatus}
        self.tag_index: Dict[str, Set[str]] = {}
        self.version_index: Dict[str, Dict[int, str]] = {}  # task_id -> {version -> result_id}
        
        # Thread safety
        self.lock = threading.Lock()
    
    def save_result(
        self,
        task_id: str,
        status: ResultStatus,
        output: Any,
        error_message: str = "",
        execution_time_seconds: float = 0.0,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Optional[str], str]:
        """
        Save a result to the store.
        
        Args:
            task_id: Associated task identifier
            status: Result status
            output: Result output data
            error_message: Error message if failed
            execution_time_seconds: Execution duration
            metadata: Optional metadata
            tags: Optional tags
            metrics: Optional performance metrics
        
        Returns:
            Tuple of (success, result_id or None, message)
        """
        with self.lock:
            if not task_id:
                return (False, None, "task_id cannot be empty")
            
            if execution_time_seconds < 0:
                return (False, None, "execution_time_seconds cannot be negative")
            
            # Check result limit
            if len(self.results) >= self.max_results:
                # Try to cleanup old versions
                self._cleanup_old_versions()
                
                if len(self.results) >= self.max_results:
                    return (False, None, f"Maximum results ({self.max_results}) reached")
            
            # Determine version
            version = self._get_next_version(task_id)
            
            # Generate result ID
            result_id = self._generate_result_id(task_id, version)
            
            # Create result
            result = StoredResult(
                result_id=result_id,
                task_id=task_id,
                status=status,
                output=output,
                error_message=error_message,
                version=version,
                created_at=time.time(),
                execution_time_seconds=execution_time_seconds,
                metadata=metadata or {},
                tags=tags or [],
                metrics=metrics or {}
            )
            
            if not result.is_valid():
                return (False, None, "Invalid result created")
            
            # Store result
            self.results[result_id] = result
            self.result_counter += 1
            
            # Add to indices
            self._add_to_indices(result)
            
            # Cleanup old versions if needed
            self._cleanup_task_versions(task_id)
            
            return (True, result_id, f"Result saved (version {version})")
    
    def get_result(self, result_id: str) -> Tuple[bool, Optional[StoredResult], str]:
        """
        Retrieve a result by ID.
        
        Args:
            result_id: Result identifier
        
        Returns:
            Tuple of (success, result or None, message)
        """
        with self.lock:
            if not result_id:
                return (False, None, "result_id cannot be empty")
            
            if result_id not in self.results:
                return (False, None, f"Result {result_id} not found")
            
            result = self.results[result_id]
            return (True, result, "Result retrieved")
    
    def get_result_by_task(
        self,
        task_id: str,
        version: Optional[int] = None
    ) -> Tuple[bool, Optional[StoredResult], str]:
        """
        Retrieve a result by task ID and optional version.
        
        Args:
            task_id: Task identifier
            version: Specific version (None for latest)
        
        Returns:
            Tuple of (success, result or None, message)
        """
        with self.lock:
            if not task_id:
                return (False, None, "task_id cannot be empty")
            
            if task_id not in self.task_index:
                return (False, None, f"No results for task {task_id}")
            
            if version is None:
                # Get latest version
                if task_id not in self.version_index:
                    return (False, None, f"No results for task {task_id}")
                
                latest_version = max(self.version_index[task_id].keys())
                result_id = self.version_index[task_id][latest_version]
                
                if result_id not in self.results:
                    return (False, None, f"Result not found")
                
                result = self.results[result_id]
                return (True, result, f"Latest result retrieved (version {latest_version})")
            
            # Get specific version
            if task_id not in self.version_index or version not in self.version_index[task_id]:
                return (False, None, f"Version {version} not found for task {task_id}")
            
            result_id = self.version_index[task_id][version]
            
            if result_id not in self.results:
                return (False, None, f"Result not found")
            
            result = self.results[result_id]
            return (True, result, f"Result version {version} retrieved")
    
    def get_all_versions(self, task_id: str) -> Tuple[bool, List[StoredResult], str]:
        """
        Get all result versions for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, results list, message)
        """
        with self.lock:
            if not task_id:
                return (False, [], "task_id cannot be empty")
            
            if task_id not in self.task_index:
                return (False, [], f"No results for task {task_id}")
            
            result_ids = self.task_index[task_id]
            results = [self.results[rid] for rid in result_ids if rid in self.results]
            
            # Sort by version
            results.sort(key=lambda r: r.version)
            
            return (True, results, f"Retrieved {len(results)} versions")
    
    def query_by_status(self, status: ResultStatus) -> List[StoredResult]:
        """
        Query results by status.
        
        Args:
            status: Status to filter by
        
        Returns:
            List of matching results
        """
        with self.lock:
            result_ids = self.status_index.get(status, set())
            return [self.results[rid] for rid in result_ids if rid in self.results]
    
    def query_by_tag(self, tag: str) -> List[StoredResult]:
        """
        Query results by tag.
        
        Args:
            tag: Tag to filter by
        
        Returns:
            List of matching results
        """
        with self.lock:
            if not tag:
                return []
            
            result_ids = self.tag_index.get(tag, set())
            return [self.results[rid] for rid in result_ids if rid in self.results]
    
    def get_successful_results(self, limit: int = 100) -> List[StoredResult]:
        """
        Get successful results.
        
        Args:
            limit: Maximum results to return
        
        Returns:
            List of successful results
        """
        successful = self.query_by_status(ResultStatus.SUCCESS)
        
        # Sort by creation time (newest first)
        successful.sort(key=lambda r: r.created_at, reverse=True)
        
        return successful[:limit]
    
    def get_failed_results(self, limit: int = 100) -> List[StoredResult]:
        """
        Get failed results.
        
        Args:
            limit: Maximum results to return
        
        Returns:
            List of failed results
        """
        failed = self.query_by_status(ResultStatus.FAILURE)
        
        # Sort by creation time (newest first)
        failed.sort(key=lambda r: r.created_at, reverse=True)
        
        return failed[:limit]
    
    def delete_result(self, result_id: str) -> Tuple[bool, str]:
        """
        Delete a result from the store.
        
        Args:
            result_id: Result identifier
        
        Returns:
            Tuple of (success, message)
        """
        with self.lock:
            if not result_id:
                return (False, "result_id cannot be empty")
            
            if result_id not in self.results:
                return (False, f"Result {result_id} not found")
            
            result = self.results.pop(result_id)
            
            # Remove from indices
            self._remove_from_indices(result)
            
            return (True, f"Result {result_id} deleted")
    
    def delete_task_results(self, task_id: str) -> Tuple[bool, int, str]:
        """
        Delete all results for a task.
        
        Args:
            task_id: Task identifier
        
        Returns:
            Tuple of (success, count_deleted, message)
        """
        with self.lock:
            if not task_id:
                return (False, 0, "task_id cannot be empty")
            
            if task_id not in self.task_index:
                return (True, 0, f"No results for task {task_id}")
            
            result_ids = self.task_index[task_id].copy()
            
            for result_id in result_ids:
                if result_id in self.results:
                    result = self.results.pop(result_id)
                    self._remove_from_indices(result)
            
            return (True, len(result_ids), f"Deleted {len(result_ids)} results")
    
    def _add_to_indices(self, result: StoredResult) -> None:
        """Add result to all indices."""
        # Task index
        if result.task_id not in self.task_index:
            self.task_index[result.task_id] = []
        self.task_index[result.task_id].append(result.result_id)
        
        # Status index
        self.status_index[result.status].add(result.result_id)
        
        # Tag index
        for tag in result.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(result.result_id)
        
        # Version index
        if result.task_id not in self.version_index:
            self.version_index[result.task_id] = {}
        self.version_index[result.task_id][result.version] = result.result_id
    
    def _remove_from_indices(self, result: StoredResult) -> None:
        """Remove result from all indices."""
        # Task index
        if result.task_id in self.task_index:
            self.task_index[result.task_id] = [
                rid for rid in self.task_index[result.task_id]
                if rid != result.result_id
            ]
            
            if not self.task_index[result.task_id]:
                self.task_index.pop(result.task_id)
        
        # Status index
        self.status_index[result.status].discard(result.result_id)
        
        # Tag index
        for tag in result.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(result.result_id)
        
        # Version index
        if result.task_id in self.version_index:
            self.version_index[result.task_id].pop(result.version, None)
            
            if not self.version_index[result.task_id]:
                self.version_index.pop(result.task_id)
    
    def _get_next_version(self, task_id: str) -> int:
        """Get next version number for task."""
        if task_id not in self.version_index or not self.version_index[task_id]:
            return 1
        
        return max(self.version_index[task_id].keys()) + 1
    
    def _cleanup_task_versions(self, task_id: str) -> None:
        """Cleanup old versions for a task."""
        if task_id not in self.version_index:
            return
        
        versions = sorted(self.version_index[task_id].keys())
        
        if len(versions) <= self.max_versions_per_task:
            return
        
        # Remove oldest versions
        to_remove = versions[:-self.max_versions_per_task]
        
        for version in to_remove:
            result_id = self.version_index[task_id][version]
            
            if result_id in self.results:
                result = self.results.pop(result_id)
                self._remove_from_indices(result)
    
    def _cleanup_old_versions(self) -> None:
        """Cleanup old versions across all tasks."""
        for task_id in list(self.version_index.keys()):
            self._cleanup_task_versions(task_id)
    
    def _generate_result_id(self, task_id: str, version: int) -> str:
        """Generate unique result ID."""
        timestamp = int(time.time())
        return f"{task_id}_result_v{version}_{timestamp}"
    
    def get_all_results(self) -> List[StoredResult]:
        """Get all results in the store."""
        with self.lock:
            return list(self.results.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive result store statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            status_counts = {
                status.value: len(self.status_index[status])
                for status in ResultStatus
            }
            
            # Calculate success rate
            total_results = len(self.results)
            success_count = len(self.status_index[ResultStatus.SUCCESS])
            success_rate = (success_count / total_results * 100) if total_results > 0 else 0.0
            
            # Calculate average execution time
            all_results = list(self.results.values())
            avg_execution_time = 0.0
            if all_results:
                avg_execution_time = sum(r.execution_time_seconds for r in all_results) / len(all_results)
            
            return {
                "total_results": total_results,
                "max_results": self.max_results,
                "total_tasks_with_results": len(self.task_index),
                "results_by_status": status_counts,
                "success_rate": f"{success_rate:.2f}%",
                "average_execution_time_seconds": f"{avg_execution_time:.3f}",
                "total_tags": len(self.tag_index),
                "max_versions_per_task": self.max_versions_per_task
            }
    
    def clear(self) -> None:
        """Clear all results from the store."""
        with self.lock:
            self.results.clear()
            self.result_counter = 0
            
            # Clear indices
            self.task_index.clear()
            
            for status in ResultStatus:
                self.status_index[status].clear()
            
            self.tag_index.clear()
            self.version_index.clear()
