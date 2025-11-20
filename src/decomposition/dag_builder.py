"""
DAG (Directed Acyclic Graph) builder for task execution.

This module builds execution graphs from decomposed tasks, determining
execution order and identifying parallel execution opportunities.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict
from enum import Enum


class NodeStatus(Enum):
    """Status of a task node in the DAG."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    """
    Node in the task execution DAG.
    
    Attributes:
        task_id: Unique task identifier
        name: Task name
        description: Task description
        dependencies: Set of task IDs this depends on
        dependents: Set of task IDs that depend on this
        status: Current execution status
        level: Execution level (for parallel execution)
    """
    task_id: str
    name: str
    description: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: NodeStatus = NodeStatus.PENDING
    level: int = -1
    
    def is_valid(self) -> bool:
        """Check if node is valid."""
        return bool(self.task_id and self.name and self.description)
    
    def is_ready(self) -> bool:
        """Check if node is ready for execution."""
        return self.status == NodeStatus.READY
    
    def add_dependency(self, task_id: str) -> bool:
        """
        Add a dependency.
        
        Args:
            task_id: ID of task this depends on
        
        Returns:
            True if added successfully
        """
        if not task_id or task_id == self.task_id:
            return False
        
        self.dependencies.add(task_id)
        return True
    
    def add_dependent(self, task_id: str) -> bool:
        """
        Add a dependent task.
        
        Args:
            task_id: ID of task that depends on this
        
        Returns:
            True if added successfully
        """
        if not task_id or task_id == self.task_id:
            return False
        
        self.dependents.add(task_id)
        return True


@dataclass
class TaskDAG:
    """
    Directed Acyclic Graph for task execution.
    
    Attributes:
        nodes: Dictionary of task nodes by ID
        execution_levels: List of task IDs grouped by execution level
        topological_order: Topological ordering of tasks
    """
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    execution_levels: List[List[str]] = field(default_factory=list)
    topological_order: List[str] = field(default_factory=list)
    
    def add_node(self, node: TaskNode) -> Tuple[bool, str]:
        """
        Add a node to the DAG.
        
        Args:
            node: Task node to add
        
        Returns:
            Tuple of (success, message)
        """
        if not node.is_valid():
            return (False, "Invalid node")
        
        if node.task_id in self.nodes:
            return (False, f"Node {node.task_id} already exists")
        
        self.nodes[node.task_id] = node
        return (True, "Node added")
    
    def get_node(self, task_id: str) -> Optional[TaskNode]:
        """Get a node by ID."""
        return self.nodes.get(task_id)
    
    def get_ready_tasks(self) -> List[str]:
        """Get all tasks ready for execution."""
        return [
            task_id for task_id, node in self.nodes.items()
            if node.is_ready()
        ]
    
    def get_level_tasks(self, level: int) -> List[str]:
        """Get all tasks at a specific execution level."""
        if level < 0 or level >= len(self.execution_levels):
            return []
        
        return self.execution_levels[level].copy()


class DAGBuilder:
    """
    Builds directed acyclic graphs for task execution.
    
    This class constructs DAGs from decomposed tasks, validates them,
    and calculates execution ordering.
    """
    
    def __init__(self):
        """Initialize DAG builder."""
        self.dag = TaskDAG()
    
    def build_from_tasks(
        self,
        task_ids: List[str],
        task_names: Dict[str, str],
        task_descriptions: Dict[str, str],
        dependencies: Dict[str, List[str]]
    ) -> Tuple[bool, str]:
        """
        Build DAG from task information.
        
        Args:
            task_ids: List of all task IDs
            task_names: Mapping of task ID to name
            task_descriptions: Mapping of task ID to description
            dependencies: Mapping of task ID to list of dependency IDs
        
        Returns:
            Tuple of (success, message)
        """
        if not task_ids:
            return (False, "No tasks provided")
        
        # Create nodes
        for task_id in task_ids:
            name = task_names.get(task_id, task_id)
            description = task_descriptions.get(task_id, "")
            
            node = TaskNode(
                task_id=task_id,
                name=name,
                description=description
            )
            
            success, message = self.dag.add_node(node)
            if not success:
                return (False, f"Failed to add node: {message}")
        
        # Add dependencies
        for task_id, deps in dependencies.items():
            node = self.dag.get_node(task_id)
            if not node:
                return (False, f"Task {task_id} not found")
            
            for dep_id in deps:
                if dep_id not in self.dag.nodes:
                    return (False, f"Dependency {dep_id} not found")
                
                node.add_dependency(dep_id)
                
                # Add reverse dependency
                dep_node = self.dag.get_node(dep_id)
                if dep_node:
                    dep_node.add_dependent(task_id)
        
        # Validate DAG
        success, message = self.validate_dag()
        if not success:
            return (False, message)
        
        # Calculate execution order
        success, message = self.calculate_execution_order()
        if not success:
            return (False, message)
        
        return (True, f"DAG built with {len(task_ids)} tasks")
    
    def validate_dag(self) -> Tuple[bool, str]:
        """
        Validate the DAG has no cycles.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            """DFS to detect cycles."""
            visited.add(task_id)
            rec_stack.add(task_id)
            
            node = self.dag.get_node(task_id)
            if not node:
                return False
            
            # Check all dependents
            for dep_id in node.dependents:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check each node
        for task_id in self.dag.nodes:
            if task_id not in visited:
                if has_cycle(task_id):
                    return (False, "Cycle detected in DAG")
        
        return (True, "DAG is valid")
    
    def calculate_execution_order(self) -> Tuple[bool, str]:
        """
        Calculate topological order and execution levels.
        
        Returns:
            Tuple of (success, message)
        """
        # Kahn's algorithm for topological sort
        in_degree = {task_id: len(node.dependencies) 
                     for task_id, node in self.dag.nodes.items()}
        
        # Queue of tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        topo_order = []
        
        # Track levels for parallel execution
        levels = []
        current_level = queue.copy()
        
        while queue:
            # Sort for deterministic ordering
            queue.sort()
            current_level.sort()
            
            # Add current level
            if current_level:
                levels.append(current_level.copy())
            
            # Process all tasks in current level
            next_level = []
            
            for task_id in current_level:
                topo_order.append(task_id)
                queue.remove(task_id)
                
                node = self.dag.get_node(task_id)
                if not node:
                    continue
                
                # Update in-degrees for dependents
                for dep_id in node.dependents:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        queue.append(dep_id)
                        next_level.append(dep_id)
            
            current_level = next_level
        
        # Check if all tasks were processed
        if len(topo_order) != len(self.dag.nodes):
            return (False, "Could not order all tasks - possible cycle")
        
        # Update DAG with results
        self.dag.topological_order = topo_order
        self.dag.execution_levels = levels
        
        # Set levels on nodes
        for level_idx, level_tasks in enumerate(levels):
            for task_id in level_tasks:
                node = self.dag.get_node(task_id)
                if node:
                    node.level = level_idx
        
        # Mark tasks with no dependencies as ready
        for task_id in levels[0] if levels else []:
            node = self.dag.get_node(task_id)
            if node:
                node.status = NodeStatus.READY
        
        return (True, f"Calculated {len(levels)} execution levels")
    
    def get_critical_path(self) -> Tuple[bool, List[str], int]:
        """
        Calculate the critical path (longest path) through the DAG.
        
        Returns:
            Tuple of (success, critical path, path length)
        """
        if not self.dag.topological_order:
            return (False, [], 0)
        
        # Calculate longest path to each node
        distances = {task_id: 0 for task_id in self.dag.nodes}
        predecessors = {task_id: None for task_id in self.dag.nodes}
        
        for task_id in self.dag.topological_order:
            node = self.dag.get_node(task_id)
            if not node:
                continue
            
            for dep_id in node.dependents:
                if distances[task_id] + 1 > distances[dep_id]:
                    distances[dep_id] = distances[task_id] + 1
                    predecessors[dep_id] = task_id
        
        # Find task with maximum distance
        if not distances:
            return (False, [], 0)
        
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
    
    def get_parallelism_factor(self) -> float:
        """
        Calculate parallelism factor (average tasks per level).
        
        Returns:
            Average number of tasks per execution level
        """
        if not self.dag.execution_levels:
            return 0.0
        
        total_tasks = sum(len(level) for level in self.dag.execution_levels)
        num_levels = len(self.dag.execution_levels)
        
        return total_tasks / num_levels if num_levels > 0 else 0.0
    
    def get_dag(self) -> TaskDAG:
        """Get the constructed DAG."""
        return self.dag
    
    def clear(self) -> None:
        """Clear the DAG."""
        self.dag = TaskDAG()
