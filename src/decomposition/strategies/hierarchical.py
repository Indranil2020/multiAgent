"""
Hierarchical decomposition strategy.

This module implements top-down hierarchical decomposition, breaking tasks
into parent-child relationships across multiple levels.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys
sys.path.append('/home/niel/git/multiAgent')
from src.core.task_spec.types import TaskType, TypedParameter, Predicate


@dataclass
class DecompositionNode:
    """
    Node in a hierarchical decomposition tree.
    
    Attributes:
        task_id: Unique identifier
        name: Task name
        description: Task description
        level: Hierarchy level (0 = root)
        parent_id: ID of parent task (None for root)
        children: List of child nodes
        is_atomic: Whether this is an atomic task
    """
    task_id: str
    name: str
    description: str
    level: int
    parent_id: Optional[str] = None
    children: List['DecompositionNode'] = field(default_factory=list)
    is_atomic: bool = False
    
    def validate_level(self) -> bool:
        """Validate level is non-negative."""
        return self.level >= 0
    
    def validate_parent(self) -> bool:
        """Validate parent relationship."""
        if self.level == 0:
            return self.parent_id is None
        return self.parent_id is not None
    
    def is_valid(self) -> bool:
        """Check if node is valid."""
        return (
            bool(self.task_id and self.name and self.description) and
            self.validate_level() and
            self.validate_parent()
        )
    
    def add_child(self, child: 'DecompositionNode') -> bool:
        """
        Add a child node.
        
        Args:
            child: Child node to add
        
        Returns:
            True if added successfully
        """
        if not child.is_valid():
            return False
        
        if child.parent_id != self.task_id:
            return False
        
        if child.level != self.level + 1:
            return False
        
        self.children.append(child)
        return True
    
    def get_depth(self) -> int:
        """Get maximum depth of subtree rooted at this node."""
        if not self.children:
            return 0
        
        return 1 + max(child.get_depth() for child in self.children)
    
    def get_leaf_count(self) -> int:
        """Get number of leaf nodes in subtree."""
        if not self.children:
            return 1
        
        return sum(child.get_leaf_count() for child in self.children)


class HierarchicalStrategy:
    """
    Hierarchical decomposition strategy.
    
    This strategy decomposes tasks in a top-down manner, creating a tree
    structure with parent-child relationships across multiple levels.
    """
    
    def __init__(self, max_children_per_node: int = 5):
        """
        Initialize hierarchical strategy.
        
        Args:
            max_children_per_node: Maximum children per node
        """
        self.max_children_per_node = max_children_per_node
        self.node_counter = 0
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.max_children_per_node < 2:
            return (False, "max_children_per_node must be at least 2")
        
        return (True, "")
    
    def decompose(
        self,
        task_id: str,
        name: str,
        description: str,
        estimated_complexity: int,
        level: int = 0,
        parent_id: Optional[str] = None
    ) -> Tuple[bool, Optional[DecompositionNode], str]:
        """
        Decompose a task hierarchically.
        
        Args:
            task_id: Task ID
            name: Task name
            description: Task description
            estimated_complexity: Estimated complexity
            level: Current hierarchy level
            parent_id: Parent task ID
        
        Returns:
            Tuple of (success, decomposition node or None, message)
        """
        # Validate inputs
        if not task_id:
            return (False, None, "task_id cannot be empty")
        
        if not name:
            return (False, None, "name cannot be empty")
        
        if not description:
            return (False, None, "description cannot be empty")
        
        if estimated_complexity < 0:
            return (False, None, "estimated_complexity cannot be negative")
        
        if level < 0:
            return (False, None, "level cannot be negative")
        
        # Create root node
        node = DecompositionNode(
            task_id=task_id,
            name=name,
            description=description,
            level=level,
            parent_id=parent_id,
            is_atomic=estimated_complexity <= 5
        )
        
        if not node.is_valid():
            return (False, None, "Invalid node created")
        
        # If atomic, no further decomposition
        if node.is_atomic:
            return (True, node, "Atomic task - no decomposition needed")
        
        # Determine number of subtasks
        subtask_count = self._calculate_subtask_count(estimated_complexity)
        
        # Create subtasks
        for i in range(subtask_count):
            subtask_id = f"{task_id}_sub{i}"
            subtask_name = f"{name} - Part {i+1}"
            subtask_desc = self._generate_subtask_description(description, i, subtask_count)
            subtask_complexity = estimated_complexity // subtask_count
            
            # Recursively decompose subtask
            success, child_node, message = self.decompose(
                task_id=subtask_id,
                name=subtask_name,
                description=subtask_desc,
                estimated_complexity=subtask_complexity,
                level=level + 1,
                parent_id=task_id
            )
            
            if success and child_node:
                node.add_child(child_node)
        
        return (True, node, f"Decomposed into {len(node.children)} subtasks")
    
    def _calculate_subtask_count(self, estimated_complexity: int) -> int:
        """
        Calculate number of subtasks to create.
        
        Args:
            estimated_complexity: Estimated complexity
        
        Returns:
            Number of subtasks
        """
        # Divide complexity to get close to atomic threshold
        if estimated_complexity <= 5:
            return 1
        
        # Calculate how many subtasks needed
        subtask_count = (estimated_complexity + 4) // 5  # Round up division by 5
        
        # Cap at max children
        subtask_count = min(subtask_count, self.max_children_per_node)
        
        # Minimum 2 subtasks if decomposing
        subtask_count = max(subtask_count, 2)
        
        return subtask_count
    
    def _generate_subtask_description(
        self,
        parent_description: str,
        subtask_index: int,
        total_subtasks: int
    ) -> str:
        """
        Generate description for a subtask.
        
        Args:
            parent_description: Parent task description
            subtask_index: Index of this subtask
            total_subtasks: Total number of subtasks
        
        Returns:
            Subtask description
        """
        # Simple approach: append part indicator
        return f"{parent_description} (Part {subtask_index + 1} of {total_subtasks})"
    
    def collect_leaf_nodes(
        self,
        root: DecompositionNode
    ) -> List[DecompositionNode]:
        """
        Collect all leaf nodes (atomic tasks) from tree.
        
        Args:
            root: Root node of tree
        
        Returns:
            List of leaf nodes
        """
        if not root.children:
            return [root]
        
        leaves = []
        for child in root.children:
            leaves.extend(self.collect_leaf_nodes(child))
        
        return leaves
    
    def get_tree_statistics(
        self,
        root: DecompositionNode
    ) -> Tuple[int, int, int]:
        """
        Get statistics about decomposition tree.
        
        Args:
            root: Root node of tree
        
        Returns:
            Tuple of (total_nodes, max_depth, leaf_count)
        """
        def count_nodes(node: DecompositionNode) -> int:
            """Count total nodes in subtree."""
            if not node.children:
                return 1
            return 1 + sum(count_nodes(child) for child in node.children)
        
        total_nodes = count_nodes(root)
        max_depth = root.get_depth()
        leaf_count = root.get_leaf_count()
        
        return (total_nodes, max_depth, leaf_count)
    
    def validate_tree(
        self,
        root: DecompositionNode
    ) -> Tuple[bool, str]:
        """
        Validate decomposition tree.
        
        Args:
            root: Root node to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not root.is_valid():
            return (False, f"Invalid root node: {root.task_id}")
        
        # Check root is at level 0
        if root.level != 0:
            return (False, "Root must be at level 0")
        
        # Check root has no parent
        if root.parent_id is not None:
            return (False, "Root must have no parent")
        
        # Recursively validate children
        def validate_subtree(node: DecompositionNode) -> Tuple[bool, str]:
            """Validate subtree rooted at node."""
            if not node.is_valid():
                return (False, f"Invalid node: {node.task_id}")
            
            for child in node.children:
                if child.parent_id != node.task_id:
                    return (False, f"Child {child.task_id} has wrong parent")
                
                if child.level != node.level + 1:
                    return (False, f"Child {child.task_id} has wrong level")
                
                success, message = validate_subtree(child)
                if not success:
                    return (False, message)
            
            return (True, "")
        
        return validate_subtree(root)
    
    def reset_counter(self) -> None:
        """Reset the node counter."""
        self.node_counter = 0
