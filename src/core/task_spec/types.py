"""
Type definitions for task specification system.

This module provides the foundational type system for task specifications,
including enums, dataclasses, and type aliases used throughout the task_spec module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union


class TaskType(Enum):
    """Types of tasks in the system."""
    DECOMPOSITION = "decomposition"
    SPECIFICATION = "specification"
    ARCHITECTURE = "architecture"
    CODE_GENERATION = "code_generation"
    VERIFICATION = "verification"
    INTEGRATION = "integration"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OPTIMIZATION = "optimization"
    REVIEW = "review"


class VerificationLevel(Enum):
    """Verification strictness levels."""
    MINIMAL = "minimal"          # Basic syntax and type checking
    STANDARD = "standard"        # Standard 8-layer verification
    STRICT = "strict"            # All layers + formal verification
    CRITICAL = "critical"        # Maximum verification for critical components


class PriorityLevel(Enum):
    """Task priority levels."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    MINIMAL = 1


class DifficultyLevel(Enum):
    """Estimated task difficulty."""
    TRIVIAL = 1
    EASY = 2
    MODERATE = 3
    HARD = 4
    EXPERT = 5


class TaskStatus(Enum):
    """Current status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VOTING = "voting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    BLOCKED = "blocked"


class ConstraintType(Enum):
    """Types of constraints on parameters."""
    RANGE = "range"              # Numeric range constraint
    LENGTH = "length"            # String/collection length
    PATTERN = "pattern"          # Regex pattern
    ENUM = "enum"                # Must be one of specific values
    TYPE = "type"                # Type constraint
    CUSTOM = "custom"            # Custom validation function


@dataclass(frozen=True)
class TypedParameter:
    """
    A typed parameter with constraints.
    
    Represents an input or output parameter with full type information,
    description, and validation constraints.
    """
    name: str
    type_annotation: str
    description: str
    constraints: List[str] = field(default_factory=list)
    default_value: Optional[Any] = None
    is_optional: bool = False
    
    def validate_name(self) -> bool:
        """Validate parameter name follows Python naming conventions."""
        if not self.name:
            return False
        if not self.name.isidentifier():
            return False
        if self.name.startswith('_'):
            return False
        return True
    
    def validate_type_annotation(self) -> bool:
        """Validate type annotation is non-empty."""
        return bool(self.type_annotation and self.type_annotation.strip())
    
    def validate_description(self) -> bool:
        """Validate description is meaningful."""
        return bool(self.description and len(self.description.strip()) >= 10)
    
    def is_valid(self) -> bool:
        """Check if parameter is fully valid."""
        return (
            self.validate_name() and
            self.validate_type_annotation() and
            self.validate_description()
        )


@dataclass(frozen=True)
class Predicate:
    """
    A logical predicate for contracts.
    
    Represents a condition that must be true (precondition, postcondition, or invariant).
    """
    name: str
    expression: str
    description: str
    severity: str = "error"  # error, warning, info
    
    def validate_name(self) -> bool:
        """Validate predicate name."""
        if not self.name:
            return False
        if not self.name.replace('_', '').isalnum():
            return False
        return True
    
    def validate_expression(self) -> bool:
        """Validate expression is non-empty."""
        return bool(self.expression and self.expression.strip())
    
    def validate_description(self) -> bool:
        """Validate description is meaningful."""
        return bool(self.description and len(self.description.strip()) >= 10)
    
    def validate_severity(self) -> bool:
        """Validate severity level."""
        return self.severity in {"error", "warning", "info"}
    
    def is_valid(self) -> bool:
        """Check if predicate is fully valid."""
        return (
            self.validate_name() and
            self.validate_expression() and
            self.validate_description() and
            self.validate_severity()
        )


@dataclass(frozen=True)
class TestCase:
    """
    A test case for verification.
    
    Represents a concrete test with inputs and expected outputs.
    """
    name: str
    inputs: Dict[str, Any]
    expected_output: Any
    description: str
    timeout_ms: int = 5000
    
    def validate_name(self) -> bool:
        """Validate test case name."""
        if not self.name:
            return False
        if not self.name.replace('_', '').replace(' ', '').isalnum():
            return False
        return True
    
    def validate_inputs(self) -> bool:
        """Validate inputs dictionary."""
        if not isinstance(self.inputs, dict):
            return False
        return len(self.inputs) >= 0  # Can be empty for no-arg functions
    
    def validate_description(self) -> bool:
        """Validate description."""
        return bool(self.description and len(self.description.strip()) >= 10)
    
    def validate_timeout(self) -> bool:
        """Validate timeout is positive."""
        return self.timeout_ms > 0
    
    def is_valid(self) -> bool:
        """Check if test case is fully valid."""
        return (
            self.validate_name() and
            self.validate_inputs() and
            self.validate_description() and
            self.validate_timeout()
        )


@dataclass(frozen=True)
class Property:
    """
    A property for property-based testing.
    
    Represents an invariant that should hold for all valid inputs.
    """
    name: str
    property_function: str
    description: str
    num_examples: int = 100
    
    def validate_name(self) -> bool:
        """Validate property name."""
        if not self.name:
            return False
        if not self.name.replace('_', '').isalnum():
            return False
        return True
    
    def validate_property_function(self) -> bool:
        """Validate property function is defined."""
        return bool(self.property_function and self.property_function.strip())
    
    def validate_description(self) -> bool:
        """Validate description."""
        return bool(self.description and len(self.description.strip()) >= 10)
    
    def validate_num_examples(self) -> bool:
        """Validate number of examples is positive."""
        return self.num_examples > 0
    
    def is_valid(self) -> bool:
        """Check if property is fully valid."""
        return (
            self.validate_name() and
            self.validate_property_function() and
            self.validate_description() and
            self.validate_num_examples()
        )


@dataclass
class PerformanceRequirement:
    """
    Performance requirements for a task.
    
    Specifies time and space complexity bounds.
    """
    max_time_ms: Optional[int] = None
    max_memory_mb: Optional[int] = None
    time_complexity: Optional[str] = None  # e.g., "O(n)", "O(n log n)"
    space_complexity: Optional[str] = None  # e.g., "O(1)", "O(n)"
    
    def validate_max_time(self) -> bool:
        """Validate max time if specified."""
        if self.max_time_ms is None:
            return True
        return self.max_time_ms > 0
    
    def validate_max_memory(self) -> bool:
        """Validate max memory if specified."""
        if self.max_memory_mb is None:
            return True
        return self.max_memory_mb > 0
    
    def validate_time_complexity(self) -> bool:
        """Validate time complexity notation."""
        if self.time_complexity is None:
            return True
        return self.time_complexity.startswith("O(") and self.time_complexity.endswith(")")
    
    def validate_space_complexity(self) -> bool:
        """Validate space complexity notation."""
        if self.space_complexity is None:
            return True
        return self.space_complexity.startswith("O(") and self.space_complexity.endswith(")")
    
    def is_valid(self) -> bool:
        """Check if performance requirements are valid."""
        return (
            self.validate_max_time() and
            self.validate_max_memory() and
            self.validate_time_complexity() and
            self.validate_space_complexity()
        )


@dataclass
class SecurityRequirement:
    """
    Security requirements for a task.
    
    Specifies security constraints and checks needed.
    """
    requires_input_validation: bool = True
    requires_output_sanitization: bool = True
    allowed_operations: List[str] = field(default_factory=list)
    forbidden_operations: List[str] = field(default_factory=list)
    requires_encryption: bool = False
    requires_authentication: bool = False
    
    def validate_allowed_operations(self) -> bool:
        """Validate allowed operations list."""
        return isinstance(self.allowed_operations, list)
    
    def validate_forbidden_operations(self) -> bool:
        """Validate forbidden operations list."""
        return isinstance(self.forbidden_operations, list)
    
    def validate_no_conflicts(self) -> bool:
        """Ensure no operation is both allowed and forbidden."""
        allowed_set = set(self.allowed_operations)
        forbidden_set = set(self.forbidden_operations)
        return len(allowed_set & forbidden_set) == 0
    
    def is_valid(self) -> bool:
        """Check if security requirements are valid."""
        return (
            self.validate_allowed_operations() and
            self.validate_forbidden_operations() and
            self.validate_no_conflicts()
        )


@dataclass
class QualityMetrics:
    """
    Quality metrics and thresholds for code.
    
    Defines acceptable quality levels for generated code.
    """
    max_cyclomatic_complexity: int = 10
    max_lines_per_function: int = 20
    min_code_coverage: float = 0.95
    max_nesting_depth: int = 4
    requires_documentation: bool = True
    requires_type_hints: bool = True
    
    def validate_complexity(self) -> bool:
        """Validate cyclomatic complexity threshold."""
        return 1 <= self.max_cyclomatic_complexity <= 20
    
    def validate_lines(self) -> bool:
        """Validate lines per function threshold."""
        return 5 <= self.max_lines_per_function <= 100
    
    def validate_coverage(self) -> bool:
        """Validate code coverage threshold."""
        return 0.0 <= self.min_code_coverage <= 1.0
    
    def validate_nesting(self) -> bool:
        """Validate nesting depth threshold."""
        return 1 <= self.max_nesting_depth <= 10
    
    def is_valid(self) -> bool:
        """Check if quality metrics are valid."""
        return (
            self.validate_complexity() and
            self.validate_lines() and
            self.validate_coverage() and
            self.validate_nesting()
        )


# Type aliases for clarity
TaskID = str
AgentID = str
ResultHash = str
Timestamp = float
