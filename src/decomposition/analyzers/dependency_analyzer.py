"""
Dependency analysis for task decomposition.

This module analyzes dependencies between tasks to build execution graphs
and detect circular dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict
from enum import Enum


class DependencyType(Enum):
    """Types of dependencies between tasks."""
    DATA_DEPENDENCY = "data"  # Task B needs output from Task A
    CONTROL_DEPENDENCY = "control"  # Task B must run after Task A
    RESOURCE_DEPENDENCY = "resource"  # Tasks share a resource


@dataclass
class TaskDependency:
    """
    Represents a dependency relationship between tasks.
    
    Attributes:
        from_task_id: ID of the task that must complete first
        to_task_id: ID of the task that depends on the first
        dependency_type: Type of dependency
        description: Human-readable description
    """
    from_task_id: str
    to_task_id: str
    dependency_type: DependencyType
    description: str
    
    def validate_ids(self) -> bool:
        """Validate task IDs are non-empty and different."""
        if not self.from_task_id or not self.to_task_id:
            return False
        if self.from_task_id == self.to_task_id:
            return False  # Self-dependency not allowed
        return True
    
    def is_valid(self) -> bool:
        """Check if dependency is valid."""
        return self.validate_ids()


@dataclass
class DependencyGraph:
    """
    Graph structure representing task dependencies.
    
    Attributes:
        tasks: Set of all task IDs in the graph
        dependencies: List of dependency relationships
        adjacency_list: Adjacency list representation
    """
    tasks: Set[str] = field(default_factory=set)
    dependencies: List[TaskDependency] = field(default_factory=list)
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_task(self, task_id: str) -> bool:
        """
        Add a task to the graph.
        
        Args:
            task_id: ID of task to add
        
        Returns:
            True if added, False if already exists
        """
        if not task_id:
            return False
        
        if task_id in self.tasks:
            return False
        
        self.tasks.add(task_id)
        self.adjacency_list[task_id] = []
        return True
    
    def add_dependency(self, dependency: TaskDependency) -> Tuple[bool, str]:
        """
        Add a dependency to the graph.
        
        Args:
            dependency: Dependency to add
        
        Returns:
            Tuple of (success, message)
        """
        if not dependency.is_valid():
            return (False, "Invalid dependency")
        
        # Ensure both tasks exist
        if dependency.from_task_id not in self.tasks:
            return (False, f"Task {dependency.from_task_id} not in graph")
        
        if dependency.to_task_id not in self.tasks:
            return (False, f"Task {dependency.to_task_id} not in graph")
        
        # Add to dependencies list
        self.dependencies.append(dependency)
        
        # Update adjacency list
        if dependency.to_task_id not in self.adjacency_list[dependency.from_task_id]:
            self.adjacency_list[dependency.from_task_id].append(dependency.to_task_id)
        
        return (True, "Dependency added")
    
    def get_dependencies(self, task_id: str) -> List[str]:
        """
        Get all tasks that depend on the given task.
        
        Args:
            task_id: Task ID to query
        
        Returns:
            List of dependent task IDs
        """
        if task_id not in self.adjacency_list:
            return []
        
        return self.adjacency_list[task_id].copy()
    
    def get_predecessors(self, task_id: str) -> List[str]:
        """
        Get all tasks that the given task depends on.
        
        Args:
            task_id: Task ID to query
        
        Returns:
            List of predecessor task IDs
        """
        predecessors = []
        
        for from_id, to_ids in self.adjacency_list.items():
            if task_id in to_ids:
                predecessors.append(from_id)
        
        return predecessors


class DependencyAnalyzer:
    """
    Analyzes dependencies between tasks.
    
    This analyzer identifies dependencies, detects cycles, and calculates
    execution ordering for task graphs.
    """
    
    def __init__(self):
        """Initialize dependency analyzer."""
        self.graph = DependencyGraph()
    
    def build_graph(
        self,
        task_ids: List[str],
        dependencies: List[TaskDependency]
    ) -> Tuple[bool, str]:
        """
        Build dependency graph from tasks and dependencies.
        
        Args:
            task_ids: List of all task IDs
            dependencies: List of dependencies
        
        Returns:
            Tuple of (success, message)
        """
        if not task_ids:
            return (False, "No tasks provided")
        
        # Add all tasks
        for task_id in task_ids:
            if not self.graph.add_task(task_id):
                return (False, f"Failed to add task {task_id}")
        
        # Add all dependencies
        for dep in dependencies:
            success, message = self.graph.add_dependency(dep)
            if not success:
                return (False, f"Failed to add dependency: {message}")
        
        return (True, f"Graph built with {len(task_ids)} tasks")
    
    def detect_cycles(self) -> Tuple[bool, List[List[str]]]:
        """
        Detect circular dependencies in the graph.
        
        Returns:
            Tuple of (has_cycles, list of cycles found)
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str, path: List[str]) -> bool:
            """DFS to detect cycles."""
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)
            
            # Check all dependencies
            for dep_id in self.graph.get_dependencies(task_id):
                if dep_id not in visited:
                    if dfs(dep_id, path.copy()):
                        return True
                elif dep_id in rec_stack:
                    # Cycle detected
                    cycle_start = path.index(dep_id)
                    cycle = path[cycle_start:] + [dep_id]
                    cycles.append(cycle)
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check each task
        for task_id in self.graph.tasks:
            if task_id not in visited:
                dfs(task_id, [])
        
        has_cycles = len(cycles) > 0
        return (has_cycles, cycles)
    
    def calculate_topological_order(self) -> Tuple[bool, Optional[List[str]], str]:
        """
        Calculate topological ordering of tasks.
        
        Returns:
            Tuple of (success, ordered task list or None, message)
        """
        # First check for cycles
        has_cycles, cycles = self.detect_cycles()
        if has_cycles:
            cycle_str = " -> ".join(cycles[0])
            return (False, None, f"Cycle detected: {cycle_str}")
        
        # Kahn's algorithm for topological sort
        in_degree = {task_id: 0 for task_id in self.graph.tasks}
        
        # Calculate in-degrees
        for task_id in self.graph.tasks:
            for dep_id in self.graph.get_dependencies(task_id):
                in_degree[dep_id] += 1
        
        # Queue of tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort queue for deterministic ordering
            queue.sort()
            
            # Process task with no remaining dependencies
            task_id = queue.pop(0)
            result.append(task_id)
            
            # Reduce in-degree for dependent tasks
            for dep_id in self.graph.get_dependencies(task_id):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)
        
        # Check if all tasks were processed
        if len(result) != len(self.graph.tasks):
            return (False, None, "Could not order all tasks")
        
        return (True, result, "Topological order calculated")
    
    def calculate_execution_levels(self) -> Tuple[bool, Optional[List[List[str]]], str]:
        """
        Calculate execution levels for parallel execution.
        
        Tasks in the same level can be executed in parallel.
        
        Returns:
            Tuple of (success, list of levels or None, message)
        """
        # Get topological order first
        success, topo_order, message = self.calculate_topological_order()
        if not success:
            return (False, None, message)
        
        # Calculate level for each task
        task_levels = {}
        
        for task_id in topo_order:
            # Level is max(predecessor levels) + 1
            predecessors = self.graph.get_predecessors(task_id)
            
            if not predecessors:
                task_levels[task_id] = 0
            else:
                max_pred_level = max(task_levels[pred] for pred in predecessors)
                task_levels[task_id] = max_pred_level + 1
        
        # Group tasks by level
        max_level = max(task_levels.values()) if task_levels else 0
        levels = [[] for _ in range(max_level + 1)]
        
        for task_id, level in task_levels.items():
            levels[level].append(task_id)
        
        # Sort tasks within each level for determinism
        for level in levels:
            level.sort()
        
        return (True, levels, f"Calculated {len(levels)} execution levels")
    
    def calculate_critical_path(self) -> Tuple[bool, Optional[List[str]], int]:
        """
        Calculate the critical path (longest path) through the graph.
        
        Returns:
            Tuple of (success, critical path or None, path length)
        """
        # Get topological order
        success, topo_order, message = self.calculate_topological_order()
        if not success:
            return (False, None, 0)
        
        # Calculate longest path to each task
        distances = {task_id: 0 for task_id in self.graph.tasks}
        predecessors = {task_id: None for task_id in self.graph.tasks}
        
        for task_id in topo_order:
            for dep_id in self.graph.get_dependencies(task_id):
                if distances[task_id] + 1 > distances[dep_id]:
                    distances[dep_id] = distances[task_id] + 1
                    predecessors[dep_id] = task_id
        
        # Find task with maximum distance
        if not distances:
            return (False, None, 0)
        
        end_task = max(distances.items(), key=lambda x: x[1])[0]
        path_length = distances[end_task]
        
        # Reconstruct path
        path = []
        current = end_task
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        path.reverse()
        
        return (True, path, path_length)
    
    def get_independent_tasks(self) -> List[str]:
        """
        Get tasks that have no dependencies.
        
        Returns:
            List of independent task IDs
        """
        independent = []
        
        for task_id in self.graph.tasks:
            predecessors = self.graph.get_predecessors(task_id)
            if not predecessors:
                independent.append(task_id)
        
        independent.sort()
        return independent
    
    def get_dependency_depth(self, task_id: str) -> Tuple[bool, int, str]:
        """
        Get the dependency depth of a task.
        
        Args:
            task_id: Task ID to analyze
        
        Returns:
            Tuple of (success, depth, message)
        """
        if task_id not in self.graph.tasks:
            return (False, 0, f"Task {task_id} not in graph")
        
        # BFS to find maximum depth
        visited = set()
        queue = [(task_id, 0)]
        max_depth = 0
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            max_depth = max(max_depth, depth)
            
            # Add predecessors with increased depth
            for pred_id in self.graph.get_predecessors(current_id):
                if pred_id not in visited:
                    queue.append((pred_id, depth + 1))
        
        return (True, max_depth, f"Dependency depth: {max_depth}")
    
    def clear(self) -> None:
        """Clear the dependency graph."""
        self.graph = DependencyGraph()
