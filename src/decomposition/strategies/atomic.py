"""
Atomic task creation for decomposition.

This module creates atomic task specifications that cannot be decomposed further
and are ready for direct implementation.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
sys.path.append('/home/niel/git/multiAgent')
from src.core.task_spec.types import (
    TaskType,
    TypedParameter,
    Predicate,
    TestCase,
    Property
)


@dataclass
class AtomicTaskSpec:
    """
    Specification for an atomic task.
    
    Attributes:
        task_id: Unique identifier
        name: Task name
        description: Detailed description
        task_type: Type of task
        inputs: Input parameters
        outputs: Output parameters
        preconditions: Precondition predicates
        postconditions: Postcondition predicates
        test_cases: Test cases
        properties: Properties for property-based testing
        estimated_lines: Estimated lines of code
        estimated_complexity: Estimated cyclomatic complexity
    """
    task_id: str
    name: str
    description: str
    task_type: TaskType
    inputs: List[TypedParameter]
    outputs: List[TypedParameter]
    preconditions: List[Predicate]
    postconditions: List[Predicate]
    test_cases: List[TestCase]
    properties: List[Property]
    estimated_lines: int
    estimated_complexity: int
    
    def validate_basic_fields(self) -> bool:
        """Validate basic fields are non-empty."""
        return bool(self.task_id and self.name and self.description)
    
    def validate_estimates(self) -> bool:
        """Validate estimates are non-negative."""
        return self.estimated_lines >= 0 and self.estimated_complexity >= 0
    
    def is_valid(self) -> bool:
        """Check if atomic task spec is valid."""
        return self.validate_basic_fields() and self.validate_estimates()


class AtomicTaskCreator:
    """
    Creates atomic task specifications.
    
    This class generates complete, atomic task specifications that meet
    the atomic criteria (≤20 lines, complexity ≤5) and are ready for
    implementation.
    """
    
    MAX_ATOMIC_LINES = 20
    MAX_ATOMIC_COMPLEXITY = 5
    
    def __init__(self):
        """Initialize atomic task creator."""
        self.task_counter = 0
    
    def create_atomic_task(
        self,
        parent_task_id: str,
        subtask_index: int,
        name: str,
        description: str,
        task_type: TaskType,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: Optional[List[Predicate]] = None,
        postconditions: Optional[List[Predicate]] = None
    ) -> Tuple[bool, Optional[AtomicTaskSpec], str]:
        """
        Create an atomic task specification.
        
        Args:
            parent_task_id: ID of parent task
            subtask_index: Index of this subtask
            name: Task name
            description: Task description
            task_type: Type of task
            inputs: Input parameters
            outputs: Output parameters
            preconditions: Optional preconditions
            postconditions: Optional postconditions
        
        Returns:
            Tuple of (success, atomic task spec or None, message)
        """
        # Validate inputs
        if not parent_task_id:
            return (False, None, "parent_task_id cannot be empty")
        
        if not name:
            return (False, None, "name cannot be empty")
        
        if not description:
            return (False, None, "description cannot be empty")
        
        if subtask_index < 0:
            return (False, None, "subtask_index cannot be negative")
        
        # Generate task ID
        task_id = f"{parent_task_id}_atomic_{subtask_index}"
        
        # Use provided or create empty lists
        preconditions = preconditions or []
        postconditions = postconditions or []
        
        # Estimate complexity
        estimated_lines = self._estimate_lines(inputs, outputs, preconditions, postconditions)
        estimated_complexity = self._estimate_complexity(preconditions, postconditions, description)
        
        # Verify atomic criteria
        if estimated_lines > self.MAX_ATOMIC_LINES:
            return (False, None, f"Task exceeds max lines ({estimated_lines} > {self.MAX_ATOMIC_LINES})")
        
        if estimated_complexity > self.MAX_ATOMIC_COMPLEXITY:
            return (False, None, f"Task exceeds max complexity ({estimated_complexity} > {self.MAX_ATOMIC_COMPLEXITY})")
        
        # Create atomic task spec
        atomic_task = AtomicTaskSpec(
            task_id=task_id,
            name=name,
            description=description,
            task_type=task_type,
            inputs=inputs,
            outputs=outputs,
            preconditions=preconditions,
            postconditions=postconditions,
            test_cases=[],  # Will be added later
            properties=[],  # Will be added later
            estimated_lines=estimated_lines,
            estimated_complexity=estimated_complexity
        )
        
        if not atomic_task.is_valid():
            return (False, None, "Invalid atomic task created")
        
        self.task_counter += 1
        return (True, atomic_task, "Atomic task created successfully")
    
    def add_test_cases(
        self,
        atomic_task: AtomicTaskSpec,
        test_cases: List[TestCase]
    ) -> Tuple[bool, str]:
        """
        Add test cases to an atomic task.
        
        Args:
            atomic_task: Atomic task to modify
            test_cases: Test cases to add
        
        Returns:
            Tuple of (success, message)
        """
        if not atomic_task.is_valid():
            return (False, "Invalid atomic task")
        
        # Add test cases (modifying in place)
        atomic_task.test_cases.extend(test_cases)
        
        return (True, f"Added {len(test_cases)} test cases")
    
    def add_properties(
        self,
        atomic_task: AtomicTaskSpec,
        properties: List[Property]
    ) -> Tuple[bool, str]:
        """
        Add properties for property-based testing.
        
        Args:
            atomic_task: Atomic task to modify
            properties: Properties to add
        
        Returns:
            Tuple of (success, message)
        """
        if not atomic_task.is_valid():
            return (False, "Invalid atomic task")
        
        # Add properties (modifying in place)
        atomic_task.properties.extend(properties)
        
        return (True, f"Added {len(properties)} properties")
    
    def _estimate_lines(
        self,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: List[Predicate],
        postconditions: List[Predicate]
    ) -> int:
        """Estimate lines of code needed."""
        base_lines = 5
        validation_lines = len(preconditions)
        logic_lines = max(3, len(inputs) * 2)
        output_lines = len(outputs)
        postcondition_lines = len(postconditions)
        
        return (
            base_lines +
            validation_lines +
            logic_lines +
            output_lines +
            postcondition_lines
        )
    
    def _estimate_complexity(
        self,
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        description: str
    ) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1
        complexity += len(preconditions)
        
        for postcond in postconditions:
            if 'if' in postcond.expression.lower():
                complexity += 1
        
        desc_lower = description.lower()
        if 'if' in desc_lower or 'when' in desc_lower:
            complexity += 1
        
        return complexity
    
    def validate_atomic_criteria(
        self,
        estimated_lines: int,
        estimated_complexity: int
    ) -> Tuple[bool, str]:
        """
        Validate that task meets atomic criteria.
        
        Args:
            estimated_lines: Estimated line count
            estimated_complexity: Estimated complexity
        
        Returns:
            Tuple of (is_atomic, reason)
        """
        if estimated_lines > self.MAX_ATOMIC_LINES:
            return (False, f"Exceeds max lines ({estimated_lines} > {self.MAX_ATOMIC_LINES})")
        
        if estimated_complexity > self.MAX_ATOMIC_COMPLEXITY:
            return (False, f"Exceeds max complexity ({estimated_complexity} > {self.MAX_ATOMIC_COMPLEXITY})")
        
        return (True, "Meets atomic criteria")
    
    def reset_counter(self) -> None:
        """Reset the task counter."""
        self.task_counter = 0
