"""
Validator for task specifications.

This module provides comprehensive validation logic for task specifications,
including semantic validation, consistency checking, and dependency validation.
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field

from .types import TaskID, TaskStatus
from .language import TaskSpecification
from .contracts import ContractValidator


@dataclass
class ValidationResult:
    """
    Result of a validation check.
    
    Contains information about whether validation passed and any errors/warnings.
    """
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.passed:
            self.passed = False
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def get_summary(self) -> str:
        """Get a summary of the validation result."""
        if self.passed:
            summary = "✅ Validation passed"
        else:
            summary = "❌ Validation failed"
        
        if self.errors:
            summary += f"\n  Errors: {len(self.errors)}"
        if self.warnings:
            summary += f"\n  Warnings: {len(self.warnings)}"
        if self.info:
            summary += f"\n  Info: {len(self.info)}"
        
        return summary


class TaskSpecificationValidator:
    """
    Comprehensive validator for task specifications.
    
    Performs multiple levels of validation including:
    - Basic field validation
    - Contract validation
    - Dependency validation
    - Semantic consistency checks
    """
    
    def __init__(self):
        """Initialize the validator."""
        self.contract_validator = ContractValidator()
    
    def validate(self, spec: TaskSpecification) -> ValidationResult:
        """
        Perform comprehensive validation of a task specification.
        
        Args:
            spec: The task specification to validate
        
        Returns:
            ValidationResult with detailed feedback
        """
        result = ValidationResult(passed=True)
        
        # Level 1: Basic validation
        basic_result = self.validate_basic(spec)
        result.merge(basic_result)
        
        # Level 2: Contract validation
        contract_result = self.validate_contracts(spec)
        result.merge(contract_result)
        
        # Level 3: Test validation
        test_result = self.validate_tests(spec)
        result.merge(test_result)
        
        # Level 4: Constraint validation
        constraint_result = self.validate_constraints(spec)
        result.merge(constraint_result)
        
        # Level 5: Semantic validation
        semantic_result = self.validate_semantics(spec)
        result.merge(semantic_result)
        
        return result
    
    def validate_basic(self, spec: TaskSpecification) -> ValidationResult:
        """Validate basic fields."""
        result = ValidationResult(passed=True)
        
        # Validate ID
        if not spec.validate_id():
            result.add_error("Task ID is invalid or empty")
        
        # Validate name
        if not spec.validate_name():
            result.add_error("Task name is invalid or too short (minimum 3 characters)")
        
        # Validate description
        if not spec.validate_description():
            result.add_error("Task description is invalid or too short (minimum 10 characters)")
        
        # Validate inputs
        if not spec.validate_inputs():
            result.add_error("One or more input parameters are invalid")
            for i, inp in enumerate(spec.inputs):
                if not inp.is_valid():
                    result.add_error(f"  Input parameter {i} '{inp.name}' is invalid")
        
        # Validate outputs
        if not spec.validate_outputs():
            result.add_error("One or more output parameters are invalid")
            for i, out in enumerate(spec.outputs):
                if not out.is_valid():
                    result.add_error(f"  Output parameter {i} '{out.name}' is invalid")
        
        return result
    
    def validate_contracts(self, spec: TaskSpecification) -> ValidationResult:
        """Validate contracts (preconditions, postconditions, invariants)."""
        result = ValidationResult(passed=True)
        
        # Validate preconditions
        if not spec.validate_preconditions():
            result.add_error("One or more preconditions are invalid")
        
        if not self.contract_validator.validate_contract_set(spec.preconditions):
            result.add_error("Preconditions set is invalid or inconsistent")
        
        # Validate postconditions
        if not spec.validate_postconditions():
            result.add_error("One or more postconditions are invalid")
        
        if not self.contract_validator.validate_contract_set(spec.postconditions):
            result.add_error("Postconditions set is invalid or inconsistent")
        
        # Validate invariants
        if not spec.validate_invariants():
            result.add_error("One or more invariants are invalid")
        
        if not self.contract_validator.validate_contract_set(spec.invariants):
            result.add_error("Invariants set is invalid or inconsistent")
        
        # Check for contract completeness
        if len(spec.inputs) > 0 and len(spec.preconditions) == 0:
            result.add_warning("Task has inputs but no preconditions defined")
        
        if len(spec.outputs) > 0 and len(spec.postconditions) == 0:
            result.add_warning("Task has outputs but no postconditions defined")
        
        return result
    
    def validate_tests(self, spec: TaskSpecification) -> ValidationResult:
        """Validate test cases and properties."""
        result = ValidationResult(passed=True)
        
        # Validate test cases
        if not spec.validate_test_cases():
            result.add_error("One or more test cases are invalid")
            for i, tc in enumerate(spec.test_cases):
                if not tc.is_valid():
                    result.add_error(f"  Test case {i} '{tc.name}' is invalid")
        
        # Validate properties
        if not spec.validate_properties():
            result.add_error("One or more properties are invalid")
            for i, prop in enumerate(spec.properties):
                if not prop.is_valid():
                    result.add_error(f"  Property {i} '{prop.name}' is invalid")
        
        # Check for test coverage
        if spec.task_type.value == "code_generation":
            if len(spec.test_cases) == 0 and len(spec.properties) == 0:
                result.add_warning("Code generation task has no test cases or properties")
        
        # Validate test case inputs match task inputs
        for tc in spec.test_cases:
            input_names = {inp.name for inp in spec.inputs}
            test_input_names = set(tc.inputs.keys())
            
            # Check for missing inputs
            missing = input_names - test_input_names
            if missing:
                result.add_warning(f"Test case '{tc.name}' missing inputs: {missing}")
            
            # Check for extra inputs
            extra = test_input_names - input_names
            if extra:
                result.add_warning(f"Test case '{tc.name}' has extra inputs: {extra}")
        
        return result
    
    def validate_constraints(self, spec: TaskSpecification) -> ValidationResult:
        """Validate constraints and requirements."""
        result = ValidationResult(passed=True)
        
        # Validate basic constraints
        if not spec.validate_constraints():
            result.add_error("Task constraints are invalid")
            if not (1 <= spec.max_complexity <= 20):
                result.add_error(f"  max_complexity {spec.max_complexity} out of range [1, 20]")
            if not (5 <= spec.max_lines <= 100):
                result.add_error(f"  max_lines {spec.max_lines} out of range [5, 100]")
            if not (100 <= spec.timeout_ms <= 60000):
                result.add_error(f"  timeout_ms {spec.timeout_ms} out of range [100, 60000]")
        
        # Validate quality metrics
        if not spec.validate_quality_metrics():
            result.add_error("Quality metrics are invalid")
        
        # Validate performance requirements
        if not spec.validate_performance_req():
            result.add_error("Performance requirements are invalid")
        
        # Validate security requirements
        if not spec.validate_security_req():
            result.add_error("Security requirements are invalid")
        
        # Check for reasonable constraints
        if spec.max_complexity < 3 and spec.max_lines > 15:
            result.add_warning("Low complexity limit with high line limit may be difficult to achieve")
        
        return result
    
    def validate_semantics(self, spec: TaskSpecification) -> ValidationResult:
        """Validate semantic consistency."""
        result = ValidationResult(passed=True)
        
        # Validate dependencies
        if not spec.validate_dependencies():
            result.add_error("Task has itself as a dependency")
        
        # Validate hierarchy
        if not spec.validate_hierarchy():
            result.add_error("Task hierarchy is invalid (task is its own parent)")
        
        # Check for atomic task consistency
        if spec.is_atomic():
            if len(spec.children) > 0:
                result.add_error("Atomic task should not have children")
            
            if spec.max_lines > 20 or spec.max_complexity > 5:
                result.add_warning("Task marked as atomic but constraints suggest it could be decomposed")
        
        # Check for leaf task consistency
        if spec.is_leaf() and spec.task_type.value == "decomposition":
            result.add_warning("Leaf task has type 'decomposition' but no children")
        
        # Check for verification level consistency
        if spec.verification_level.value == "critical":
            if len(spec.test_cases) < 3:
                result.add_warning("Critical verification level but fewer than 3 test cases")
            if len(spec.preconditions) == 0 or len(spec.postconditions) == 0:
                result.add_warning("Critical verification level but missing contracts")
        
        return result
    
    def validate_dependency_graph(
        self,
        specs: List[TaskSpecification]
    ) -> ValidationResult:
        """
        Validate a set of task specifications for dependency consistency.
        
        Args:
            specs: List of task specifications to validate
        
        Returns:
            ValidationResult with dependency graph validation
        """
        result = ValidationResult(passed=True)
        
        # Build task ID set
        task_ids = {spec.id for spec in specs}
        
        # Check each task's dependencies
        for spec in specs:
            for dep_id in spec.dependencies:
                if dep_id not in task_ids:
                    result.add_error(
                        f"Task '{spec.id}' depends on non-existent task '{dep_id}'"
                    )
        
        # Check for circular dependencies
        circular = self._detect_circular_dependencies(specs)
        if circular:
            result.add_error(f"Circular dependency detected: {' -> '.join(circular)}")
        
        # Check parent-child consistency
        for spec in specs:
            if spec.parent:
                if spec.parent not in task_ids:
                    result.add_error(
                        f"Task '{spec.id}' has non-existent parent '{spec.parent}'"
                    )
                else:
                    # Find parent and check if this task is in its children
                    parent = next((s for s in specs if s.id == spec.parent), None)
                    if parent and spec.id not in parent.children:
                        result.add_warning(
                            f"Task '{spec.id}' has parent '{spec.parent}' but is not in parent's children list"
                        )
        
        return result
    
    def _detect_circular_dependencies(
        self,
        specs: List[TaskSpecification]
    ) -> Optional[List[TaskID]]:
        """
        Detect circular dependencies in task graph.
        
        Args:
            specs: List of task specifications
        
        Returns:
            List of task IDs forming a cycle, or None if no cycle
        """
        # Build adjacency list
        graph: Dict[TaskID, List[TaskID]] = {}
        for spec in specs:
            graph[spec.id] = spec.dependencies.copy()
        
        # Track visited nodes and recursion stack
        visited: Set[TaskID] = set()
        rec_stack: Set[TaskID] = set()
        path: List[TaskID] = []
        
        def dfs(node: TaskID) -> Optional[List[TaskID]]:
            """DFS to detect cycle."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        cycle = dfs(neighbor)
                        if cycle:
                            return cycle
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]
            
            path.pop()
            rec_stack.remove(node)
            return None
        
        # Check each node
        for task_id in graph.keys():
            if task_id not in visited:
                cycle = dfs(task_id)
                if cycle:
                    return cycle
        
        return None


class TaskSetValidator:
    """
    Validator for sets of related task specifications.
    
    Validates consistency across multiple tasks including dependency graphs,
    hierarchies, and resource allocation.
    """
    
    def __init__(self):
        """Initialize the set validator."""
        self.spec_validator = TaskSpecificationValidator()
    
    def validate_task_set(
        self,
        specs: List[TaskSpecification]
    ) -> ValidationResult:
        """
        Validate a set of task specifications.
        
        Args:
            specs: List of task specifications to validate
        
        Returns:
            ValidationResult for the entire set
        """
        result = ValidationResult(passed=True)
        
        # Validate each individual spec
        for i, spec in enumerate(specs):
            spec_result = self.spec_validator.validate(spec)
            if not spec_result.passed:
                result.add_error(f"Task {i} '{spec.id}' failed validation")
                result.merge(spec_result)
        
        # Validate dependency graph
        dep_result = self.spec_validator.validate_dependency_graph(specs)
        result.merge(dep_result)
        
        # Validate unique IDs
        id_result = self._validate_unique_ids(specs)
        result.merge(id_result)
        
        # Validate hierarchy consistency
        hierarchy_result = self._validate_hierarchy(specs)
        result.merge(hierarchy_result)
        
        return result
    
    def _validate_unique_ids(
        self,
        specs: List[TaskSpecification]
    ) -> ValidationResult:
        """Validate that all task IDs are unique."""
        result = ValidationResult(passed=True)
        
        ids = [spec.id for spec in specs]
        unique_ids = set(ids)
        
        if len(ids) != len(unique_ids):
            # Find duplicates
            seen = set()
            duplicates = set()
            for task_id in ids:
                if task_id in seen:
                    duplicates.add(task_id)
                seen.add(task_id)
            
            result.add_error(f"Duplicate task IDs found: {duplicates}")
        
        return result
    
    def _validate_hierarchy(
        self,
        specs: List[TaskSpecification]
    ) -> ValidationResult:
        """Validate task hierarchy consistency."""
        result = ValidationResult(passed=True)
        
        # Build parent-child map
        task_map = {spec.id: spec for spec in specs}
        
        # Check each parent-child relationship
        for spec in specs:
            # Check children exist and have correct parent
            for child_id in spec.children:
                if child_id not in task_map:
                    result.add_error(
                        f"Task '{spec.id}' has non-existent child '{child_id}'"
                    )
                else:
                    child = task_map[child_id]
                    if child.parent != spec.id:
                        result.add_error(
                            f"Child '{child_id}' does not have '{spec.id}' as parent"
                        )
        
        # Check for orphaned tasks (tasks with parent but parent doesn't list them as child)
        for spec in specs:
            if spec.parent:
                if spec.parent in task_map:
                    parent = task_map[spec.parent]
                    if spec.id not in parent.children:
                        result.add_warning(
                            f"Task '{spec.id}' has parent '{spec.parent}' but parent doesn't list it as child"
                        )
        
        return result


def validate_task_spec(spec: TaskSpecification) -> ValidationResult:
    """
    Convenience function to validate a single task specification.
    
    Args:
        spec: The task specification to validate
    
    Returns:
        ValidationResult with detailed feedback
    """
    validator = TaskSpecificationValidator()
    return validator.validate(spec)


def validate_task_specs(specs: List[TaskSpecification]) -> ValidationResult:
    """
    Convenience function to validate a set of task specifications.
    
    Args:
        specs: List of task specifications to validate
    
    Returns:
        ValidationResult for the entire set
    """
    validator = TaskSetValidator()
    return validator.validate_task_set(specs)
