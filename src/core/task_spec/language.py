"""
Task Specification Language.

This module provides the main TaskSpecification class and related builders
for defining formal task specifications in the zero-error system.
"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional
from datetime import datetime

from .types import (
    TaskType, VerificationLevel, PriorityLevel, DifficultyLevel,
    TaskStatus, TypedParameter, Predicate, TestCase, Property,
    PerformanceRequirement, SecurityRequirement, QualityMetrics,
    TaskID, Timestamp
)


@dataclass
class TaskSpecification:
    """
    Formal specification for a task in the zero-error system.
    
    This is the core data structure that defines what needs to be done,
    with complete formal contracts, verification requirements, and metadata.
    """
    
    # Identity
    id: TaskID
    name: str
    description: str
    task_type: TaskType
    
    # Formal contracts
    inputs: List[TypedParameter] = field(default_factory=list)
    outputs: List[TypedParameter] = field(default_factory=list)
    preconditions: List[Predicate] = field(default_factory=list)
    postconditions: List[Predicate] = field(default_factory=list)
    invariants: List[Predicate] = field(default_factory=list)
    
    # Dependencies and hierarchy
    dependencies: List[TaskID] = field(default_factory=list)
    parent: Optional[TaskID] = None
    children: List[TaskID] = field(default_factory=list)
    
    # Verification
    test_cases: List[TestCase] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    verification_level: VerificationLevel = VerificationLevel.STANDARD
    
    # Quality and performance
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    performance_req: PerformanceRequirement = field(default_factory=PerformanceRequirement)
    security_req: SecurityRequirement = field(default_factory=SecurityRequirement)
    
    # Constraints
    max_complexity: int = 10
    max_lines: int = 20
    timeout_ms: int = 5000
    
    # Metadata
    priority: PriorityLevel = PriorityLevel.NORMAL
    estimated_difficulty: DifficultyLevel = DifficultyLevel.MODERATE
    status: TaskStatus = TaskStatus.PENDING
    created_at: Timestamp = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: Timestamp = field(default_factory=lambda: datetime.now().timestamp())
    
    # Context and hints
    context: Dict[str, Any] = field(default_factory=dict)
    hints: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    # Results (populated during execution)
    result: Optional[Any] = None
    verification_results: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    
    def validate_id(self) -> bool:
        """Validate task ID is non-empty."""
        return bool(self.id and self.id.strip())
    
    def validate_name(self) -> bool:
        """Validate task name is meaningful."""
        return bool(self.name and len(self.name.strip()) >= 3)
    
    def validate_description(self) -> bool:
        """Validate description is meaningful."""
        return bool(self.description and len(self.description.strip()) >= 10)
    
    def validate_inputs(self) -> bool:
        """Validate all inputs are valid."""
        return all(inp.is_valid() for inp in self.inputs)
    
    def validate_outputs(self) -> bool:
        """Validate all outputs are valid."""
        return all(out.is_valid() for out in self.outputs)
    
    def validate_preconditions(self) -> bool:
        """Validate all preconditions are valid."""
        return all(pre.is_valid() for pre in self.preconditions)
    
    def validate_postconditions(self) -> bool:
        """Validate all postconditions are valid."""
        return all(post.is_valid() for post in self.postconditions)
    
    def validate_invariants(self) -> bool:
        """Validate all invariants are valid."""
        return all(inv.is_valid() for inv in self.invariants)
    
    def validate_test_cases(self) -> bool:
        """Validate all test cases are valid."""
        return all(tc.is_valid() for tc in self.test_cases)
    
    def validate_properties(self) -> bool:
        """Validate all properties are valid."""
        return all(prop.is_valid() for prop in self.properties)
    
    def validate_constraints(self) -> bool:
        """Validate constraint values are reasonable."""
        return (
            1 <= self.max_complexity <= 20 and
            5 <= self.max_lines <= 100 and
            100 <= self.timeout_ms <= 60000
        )
    
    def validate_quality_metrics(self) -> bool:
        """Validate quality metrics."""
        return self.quality_metrics.is_valid()
    
    def validate_performance_req(self) -> bool:
        """Validate performance requirements."""
        return self.performance_req.is_valid()
    
    def validate_security_req(self) -> bool:
        """Validate security requirements."""
        return self.security_req.is_valid()
    
    def validate_dependencies(self) -> bool:
        """Validate dependencies don't include self."""
        return self.id not in self.dependencies
    
    def validate_hierarchy(self) -> bool:
        """Validate parent-child relationships."""
        if self.parent and self.parent == self.id:
            return False  # Can't be own parent
        return True
    
    def is_valid(self) -> bool:
        """
        Check if task specification is fully valid.
        
        Returns:
            True if all validation checks pass
        """
        return (
            self.validate_id() and
            self.validate_name() and
            self.validate_description() and
            self.validate_inputs() and
            self.validate_outputs() and
            self.validate_preconditions() and
            self.validate_postconditions() and
            self.validate_invariants() and
            self.validate_test_cases() and
            self.validate_properties() and
            self.validate_constraints() and
            self.validate_quality_metrics() and
            self.validate_performance_req() and
            self.validate_security_req() and
            self.validate_dependencies() and
            self.validate_hierarchy()
        )
    
    def is_atomic(self) -> bool:
        """
        Check if task is atomic (cannot be decomposed further).
        
        Returns:
            True if task is atomic
        """
        return (
            len(self.children) == 0 and
            self.max_lines <= 20 and
            self.max_complexity <= 5
        )
    
    def is_leaf(self) -> bool:
        """Check if task is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if task is a root node (no parent)."""
        return self.parent is None
    
    def has_dependencies(self) -> bool:
        """Check if task has dependencies."""
        return len(self.dependencies) > 0
    
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies met)."""
        return self.status == TaskStatus.PENDING and not self.has_dependencies()
    
    def mark_in_progress(self) -> 'TaskSpecification':
        """Mark task as in progress."""
        return replace(
            self,
            status=TaskStatus.IN_PROGRESS,
            updated_at=datetime.now().timestamp()
        )
    
    def mark_completed(self, result: Any) -> 'TaskSpecification':
        """Mark task as completed with result."""
        return replace(
            self,
            status=TaskStatus.COMPLETED,
            result=result,
            updated_at=datetime.now().timestamp()
        )
    
    def mark_failed(self, error: str) -> 'TaskSpecification':
        """Mark task as failed."""
        return replace(
            self,
            status=TaskStatus.FAILED,
            verification_results={'error': error},
            updated_at=datetime.now().timestamp()
        )
    
    def mark_escalated(self, reason: str) -> 'TaskSpecification':
        """Mark task as escalated to human."""
        return replace(
            self,
            status=TaskStatus.ESCALATED,
            verification_results={'escalation_reason': reason},
            updated_at=datetime.now().timestamp()
        )
    
    def add_dependency(self, task_id: TaskID) -> 'TaskSpecification':
        """Add a dependency."""
        if task_id not in self.dependencies and task_id != self.id:
            new_deps = self.dependencies + [task_id]
            return replace(self, dependencies=new_deps)
        return self
    
    def remove_dependency(self, task_id: TaskID) -> 'TaskSpecification':
        """Remove a dependency."""
        if task_id in self.dependencies:
            new_deps = [d for d in self.dependencies if d != task_id]
            return replace(self, dependencies=new_deps)
        return self
    
    def add_child(self, task_id: TaskID) -> 'TaskSpecification':
        """Add a child task."""
        if task_id not in self.children and task_id != self.id:
            new_children = self.children + [task_id]
            return replace(self, children=new_children)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type.value,
            'status': self.status.value,
            'priority': self.priority.value,
            'difficulty': self.estimated_difficulty.value,
            'max_complexity': self.max_complexity,
            'max_lines': self.max_lines,
            'timeout_ms': self.timeout_ms,
            'dependencies': self.dependencies,
            'parent': self.parent,
            'children': self.children,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }


class TaskSpecificationBuilder:
    """
    Builder for creating TaskSpecification instances.
    
    Provides a fluent interface for constructing task specifications
    with proper validation.
    """
    
    def __init__(self, task_id: TaskID, name: str, description: str, task_type: TaskType):
        """Initialize builder with required fields."""
        self.task_id = task_id
        self.name = name
        self.description = description
        self.task_type = task_type
        
        # Optional fields
        self.inputs: List[TypedParameter] = []
        self.outputs: List[TypedParameter] = []
        self.preconditions: List[Predicate] = []
        self.postconditions: List[Predicate] = []
        self.invariants: List[Predicate] = []
        self.dependencies: List[TaskID] = []
        self.parent: Optional[TaskID] = None
        self.test_cases: List[TestCase] = []
        self.properties: List[Property] = []
        self.verification_level = VerificationLevel.STANDARD
        self.quality_metrics = QualityMetrics()
        self.performance_req = PerformanceRequirement()
        self.security_req = SecurityRequirement()
        self.max_complexity = 10
        self.max_lines = 20
        self.timeout_ms = 5000
        self.priority = PriorityLevel.NORMAL
        self.estimated_difficulty = DifficultyLevel.MODERATE
        self.context: Dict[str, Any] = {}
        self.hints: List[str] = []
        self.examples: List[str] = []
    
    def with_inputs(self, inputs: List[TypedParameter]) -> 'TaskSpecificationBuilder':
        """Set input parameters."""
        self.inputs = inputs
        return self
    
    def add_input(self, param: TypedParameter) -> 'TaskSpecificationBuilder':
        """Add a single input parameter."""
        if param.is_valid():
            self.inputs.append(param)
        return self
    
    def with_outputs(self, outputs: List[TypedParameter]) -> 'TaskSpecificationBuilder':
        """Set output parameters."""
        self.outputs = outputs
        return self
    
    def add_output(self, param: TypedParameter) -> 'TaskSpecificationBuilder':
        """Add a single output parameter."""
        if param.is_valid():
            self.outputs.append(param)
        return self
    
    def with_preconditions(self, preconditions: List[Predicate]) -> 'TaskSpecificationBuilder':
        """Set preconditions."""
        self.preconditions = preconditions
        return self
    
    def add_precondition(self, predicate: Predicate) -> 'TaskSpecificationBuilder':
        """Add a single precondition."""
        if predicate.is_valid():
            self.preconditions.append(predicate)
        return self
    
    def with_postconditions(self, postconditions: List[Predicate]) -> 'TaskSpecificationBuilder':
        """Set postconditions."""
        self.postconditions = postconditions
        return self
    
    def add_postcondition(self, predicate: Predicate) -> 'TaskSpecificationBuilder':
        """Add a single postcondition."""
        if predicate.is_valid():
            self.postconditions.append(predicate)
        return self
    
    def with_invariants(self, invariants: List[Predicate]) -> 'TaskSpecificationBuilder':
        """Set invariants."""
        self.invariants = invariants
        return self
    
    def add_invariant(self, predicate: Predicate) -> 'TaskSpecificationBuilder':
        """Add a single invariant."""
        if predicate.is_valid():
            self.invariants.append(predicate)
        return self
    
    def with_dependencies(self, dependencies: List[TaskID]) -> 'TaskSpecificationBuilder':
        """Set dependencies."""
        self.dependencies = dependencies
        return self
    
    def add_dependency(self, task_id: TaskID) -> 'TaskSpecificationBuilder':
        """Add a single dependency."""
        if task_id and task_id != self.task_id:
            self.dependencies.append(task_id)
        return self
    
    def with_parent(self, parent_id: TaskID) -> 'TaskSpecificationBuilder':
        """Set parent task."""
        if parent_id and parent_id != self.task_id:
            self.parent = parent_id
        return self
    
    def with_test_cases(self, test_cases: List[TestCase]) -> 'TaskSpecificationBuilder':
        """Set test cases."""
        self.test_cases = test_cases
        return self
    
    def add_test_case(self, test_case: TestCase) -> 'TaskSpecificationBuilder':
        """Add a single test case."""
        if test_case.is_valid():
            self.test_cases.append(test_case)
        return self
    
    def with_properties(self, properties: List[Property]) -> 'TaskSpecificationBuilder':
        """Set properties for property-based testing."""
        self.properties = properties
        return self
    
    def add_property(self, prop: Property) -> 'TaskSpecificationBuilder':
        """Add a single property."""
        if prop.is_valid():
            self.properties.append(prop)
        return self
    
    def with_verification_level(self, level: VerificationLevel) -> 'TaskSpecificationBuilder':
        """Set verification level."""
        self.verification_level = level
        return self
    
    def with_quality_metrics(self, metrics: QualityMetrics) -> 'TaskSpecificationBuilder':
        """Set quality metrics."""
        if metrics.is_valid():
            self.quality_metrics = metrics
        return self
    
    def with_performance_req(self, req: PerformanceRequirement) -> 'TaskSpecificationBuilder':
        """Set performance requirements."""
        if req.is_valid():
            self.performance_req = req
        return self
    
    def with_security_req(self, req: SecurityRequirement) -> 'TaskSpecificationBuilder':
        """Set security requirements."""
        if req.is_valid():
            self.security_req = req
        return self
    
    def with_constraints(
        self,
        max_complexity: int = 10,
        max_lines: int = 20,
        timeout_ms: int = 5000
    ) -> 'TaskSpecificationBuilder':
        """Set complexity and size constraints."""
        if 1 <= max_complexity <= 20:
            self.max_complexity = max_complexity
        if 5 <= max_lines <= 100:
            self.max_lines = max_lines
        if 100 <= timeout_ms <= 60000:
            self.timeout_ms = timeout_ms
        return self
    
    def with_priority(self, priority: PriorityLevel) -> 'TaskSpecificationBuilder':
        """Set priority level."""
        self.priority = priority
        return self
    
    def with_difficulty(self, difficulty: DifficultyLevel) -> 'TaskSpecificationBuilder':
        """Set estimated difficulty."""
        self.estimated_difficulty = difficulty
        return self
    
    def with_context(self, context: Dict[str, Any]) -> 'TaskSpecificationBuilder':
        """Set context dictionary."""
        self.context = context
        return self
    
    def add_context(self, key: str, value: Any) -> 'TaskSpecificationBuilder':
        """Add a context entry."""
        self.context[key] = value
        return self
    
    def with_hints(self, hints: List[str]) -> 'TaskSpecificationBuilder':
        """Set implementation hints."""
        self.hints = hints
        return self
    
    def add_hint(self, hint: str) -> 'TaskSpecificationBuilder':
        """Add a single hint."""
        if hint and hint.strip():
            self.hints.append(hint)
        return self
    
    def with_examples(self, examples: List[str]) -> 'TaskSpecificationBuilder':
        """Set examples."""
        self.examples = examples
        return self
    
    def add_example(self, example: str) -> 'TaskSpecificationBuilder':
        """Add a single example."""
        if example and example.strip():
            self.examples.append(example)
        return self
    
    def build(self) -> TaskSpecification:
        """Build and return the TaskSpecification."""
        spec = TaskSpecification(
            id=self.task_id,
            name=self.name,
            description=self.description,
            task_type=self.task_type,
            inputs=self.inputs,
            outputs=self.outputs,
            preconditions=self.preconditions,
            postconditions=self.postconditions,
            invariants=self.invariants,
            dependencies=self.dependencies,
            parent=self.parent,
            test_cases=self.test_cases,
            properties=self.properties,
            verification_level=self.verification_level,
            quality_metrics=self.quality_metrics,
            performance_req=self.performance_req,
            security_req=self.security_req,
            max_complexity=self.max_complexity,
            max_lines=self.max_lines,
            timeout_ms=self.timeout_ms,
            priority=self.priority,
            estimated_difficulty=self.estimated_difficulty,
            context=self.context,
            hints=self.hints,
            examples=self.examples
        )
        
        return spec
