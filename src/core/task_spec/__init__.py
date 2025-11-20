"""
Task Specification Module.

This module provides a comprehensive system for defining, parsing, and validating
formal task specifications in the zero-error software development architecture.

Key Components:
- Types: Core type definitions and enums
- Contracts: Pre/postconditions and invariants
- Language: TaskSpecification class and builders
- Parser: Parse specifications from dict/JSON/YAML
- Validator: Comprehensive validation logic

Example Usage:
    ```python
    from task_spec import (
        TaskSpecification,
        TaskSpecificationBuilder,
        TaskType,
        TypedParameter,
        validate_task_spec
    )
    
    # Create a task specification
    builder = TaskSpecificationBuilder(
        task_id="task_001",
        name="Calculate Fibonacci",
        description="Calculate the nth Fibonacci number efficiently",
        task_type=TaskType.CODE_GENERATION
    )
    
    # Add inputs
    builder.add_input(TypedParameter(
        name="n",
        type_annotation="int",
        description="The position in Fibonacci sequence (n >= 0)"
    ))
    
    # Add outputs
    builder.add_output(TypedParameter(
        name="result",
        type_annotation="int",
        description="The nth Fibonacci number"
    ))
    
    # Build and validate
    spec = builder.build()
    validation_result = validate_task_spec(spec)
    
    if validation_result.passed:
        print("✅ Task specification is valid")
    else:
        print("❌ Validation failed:")
        for error in validation_result.errors:
            print(f"  - {error}")
    ```
"""

# Core types and enums
from .types import (
    TaskType,
    VerificationLevel,
    PriorityLevel,
    DifficultyLevel,
    TaskStatus,
    ConstraintType,
    TypedParameter,
    Predicate,
    TestCase,
    Property,
    PerformanceRequirement,
    SecurityRequirement,
    QualityMetrics,
    TaskID,
    AgentID,
    ResultHash,
    Timestamp,
)

# Contract builders and validators
from .contracts import (
    ContractBuilder,
    PreconditionBuilder,
    PostconditionBuilder,
    InvariantBuilder,
    ContractValidator,
    create_standard_preconditions,
    create_standard_postconditions,
)

# Task specification language
from .language import (
    TaskSpecification,
    TaskSpecificationBuilder,
)

# Parsers
from .parser import (
    TaskSpecificationParser,
    parse_task_spec_from_dict,
    parse_task_spec_from_json,
    parse_task_spec_from_yaml,
)

# Validators
from .validator import (
    ValidationResult,
    TaskSpecificationValidator,
    TaskSetValidator,
    validate_task_spec,
    validate_task_specs,
)

# Version
__version__ = "0.1.0"

# Public API
__all__ = [
    # Types and enums
    "TaskType",
    "VerificationLevel",
    "PriorityLevel",
    "DifficultyLevel",
    "TaskStatus",
    "ConstraintType",
    "TypedParameter",
    "Predicate",
    "TestCase",
    "Property",
    "PerformanceRequirement",
    "SecurityRequirement",
    "QualityMetrics",
    "TaskID",
    "AgentID",
    "ResultHash",
    "Timestamp",
    
    # Contract builders
    "ContractBuilder",
    "PreconditionBuilder",
    "PostconditionBuilder",
    "InvariantBuilder",
    "ContractValidator",
    "create_standard_preconditions",
    "create_standard_postconditions",
    
    # Task specification
    "TaskSpecification",
    "TaskSpecificationBuilder",
    
    # Parsers
    "TaskSpecificationParser",
    "parse_task_spec_from_dict",
    "parse_task_spec_from_json",
    "parse_task_spec_from_yaml",
    
    # Validators
    "ValidationResult",
    "TaskSpecificationValidator",
    "TaskSetValidator",
    "validate_task_spec",
    "validate_task_specs",
]
