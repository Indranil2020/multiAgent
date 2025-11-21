"""
Coding Prompts Module.

This module provides comprehensive prompt templates for coder agents in the
zero-error system. Coder agents are responsible for generating syntactically
correct, complete, and fully functional code implementations.

Key Responsibilities:
- Generate complete function/class implementations
- Handle all edge cases explicitly
- Include proper error handling
- Add comprehensive docstrings
- Follow coding best practices
- Ensure type safety with annotations

All prompts enforce zero-error philosophy with no placeholders or TODOs.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .base_prompts import PromptTemplate, PromptFormat


# Function Implementation Prompt
FUNCTION_IMPLEMENTATION_PROMPT = PromptTemplate(
    template_id="function_implementation",
    name="Function Implementation Prompt",
    template_text="""Generate a complete Python function implementation.

FUNCTION NAME: {function_name}
DESCRIPTION: {description}
INPUTS: {inputs}
OUTPUTS: {outputs}
CONSTRAINTS: {constraints}

REQUIREMENTS:
1. Complete implementation (no placeholders)
2. Handle ALL edge cases explicitly
3. Validate all inputs at function start
4. Return explicit error values (not exceptions) for expected failures
5. Include comprehensive docstring with:
   - Function purpose
   - Args with types and descriptions
   - Returns with type and description
   - Raises (only for truly exceptional conditions)
   - Examples of usage
6. Use type hints for all parameters and return value
7. Follow PEP 8 style guidelines
8. Single Responsibility Principle

EXAMPLE STRUCTURE:
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    \"\"\"
    Brief description.

    Detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Examples:
        >>> function_name(value1, value2)
        expected_result
    \"\"\"
    # Validate inputs
    if not valid_input(param1):
        return error_value

    # Implementation
    result = process(param1, param2)

    return result
```

Generate the complete implementation now.""",
    format=PromptFormat.MARKDOWN,
    variables=["function_name", "description", "inputs", "outputs", "constraints"]
)


# Class Implementation Prompt
CLASS_IMPLEMENTATION_PROMPT = PromptTemplate(
    template_id="class_implementation",
    name="Class Implementation Prompt",
    template_text="""Generate a complete Python class implementation.

CLASS NAME: {class_name}
DESCRIPTION: {description}
ATTRIBUTES: {attributes}
METHODS: {methods}
CONSTRAINTS: {constraints}

REQUIREMENTS:
1. Complete class with all methods implemented
2. Proper __init__ with input validation
3. Type hints for all attributes and methods
4. Comprehensive class and method docstrings
5. Private methods prefixed with underscore
6. Properties for computed attributes
7. __repr__ and __str__ if appropriate
8. Dataclass decorator if primarily data storage

STRUCTURE:
```python
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass  # if applicable
class ClassName:
    \"\"\"
    Brief class description.

    Detailed description.

    Attributes:
        attr1: Description
        attr2: Description
    \"\"\"
    attr1: Type1
    attr2: Type2

    def __init__(self, param1: Type1, param2: Type2):
        \"\"\"Initialize class.\"\"\"
        # Validate inputs
        if not valid(param1):
            raise ValueError("Invalid param1")

        self.attr1 = param1
        self.attr2 = param2

    def method1(self, param: Type) -> ReturnType:
        \"\"\"Method description.\"\"\"
        # Implementation
        pass

    def _private_method(self) -> None:
        \"\"\"Private helper method.\"\"\"
        pass
```

Generate complete implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["class_name", "description", "attributes", "methods", "constraints"]
)


# Algorithm Implementation Prompt
ALGORITHM_IMPLEMENTATION_PROMPT = PromptTemplate(
    template_id="algorithm_implementation",
    name="Algorithm Implementation Prompt",
    template_text="""Implement this algorithm with optimal complexity.

ALGORITHM: {algorithm_name}
PROBLEM: {problem_description}
INPUT SIZE: {input_size}
EXPECTED COMPLEXITY: {complexity}

REQUIREMENTS:
1. Implement with specified time/space complexity
2. Handle all edge cases:
   - Empty input
   - Single element
   - Duplicates
   - Invalid input
3. Add complexity analysis in docstring
4. Include proof of correctness reasoning
5. Optimize for the common case
6. Use appropriate data structures

TEMPLATE:
```python
def algorithm_name(input_data: InputType) -> OutputType:
    \"\"\"
    Algorithm description.

    Time Complexity: O(n log n)
    Space Complexity: O(n)

    Algorithm:
    1. Step one description
    2. Step two description
    3. Step three description

    Args:
        input_data: Description

    Returns:
        Description

    Examples:
        >>> algorithm_name([1, 2, 3])
        expected_output
    \"\"\"
    # Validate input
    if not input_data:
        return default_value

    # Handle edge cases
    if len(input_data) == 1:
        return special_case_result

    # Main algorithm
    result = process(input_data)

    return result
```

Implement now with optimal complexity.""",
    format=PromptFormat.MARKDOWN,
    variables=["algorithm_name", "problem_description", "input_size", "complexity"]
)


# API Endpoint Implementation Prompt
API_ENDPOINT_PROMPT = PromptTemplate(
    template_id="api_endpoint",
    name="API Endpoint Implementation Prompt",
    template_text="""Implement a REST API endpoint.

ENDPOINT: {method} {path}
DESCRIPTION: {description}
REQUEST BODY: {request_schema}
RESPONSE: {response_schema}
STATUS CODES: {status_codes}

REQUIREMENTS:
1. Input validation with Pydantic models
2. Proper error handling with appropriate status codes
3. Request/response serialization
4. Authentication/authorization if needed
5. Rate limiting consideration
6. Comprehensive docstring with OpenAPI specs

STRUCTURE:
```python
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter()

class RequestModel(BaseModel):
    \"\"\"Request schema.\"\"\"
    field1: str = Field(..., description="Field description")
    field2: int = Field(..., gt=0, description="Must be positive")

class ResponseModel(BaseModel):
    \"\"\"Response schema.\"\"\"
    result: str
    metadata: Dict[str, Any]

@router.{method_lower}("{path}")
async def endpoint_name(
    request: RequestModel,
    # Add path/query parameters
) -> ResponseModel:
    \"\"\"
    Endpoint description.

    Args:
        request: Request payload

    Returns:
        Response payload

    Raises:
        HTTPException: 400 if invalid input
        HTTPException: 404 if not found
        HTTPException: 500 if server error
    \"\"\"
    # Validate business logic
    if not valid_business_logic(request):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request"
        )

    # Process request
    result = process(request)

    return ResponseModel(result=result, metadata={{}})
```

Implement complete endpoint.""",
    format=PromptFormat.MARKDOWN,
    variables=["method", "path", "description", "request_schema", "response_schema", "status_codes"]
)


# Database Model Prompt
DATABASE_MODEL_PROMPT = PromptTemplate(
    template_id="database_model",
    name="Database Model Implementation Prompt",
    template_text="""Implement a database model (ORM).

MODEL NAME: {model_name}
TABLE NAME: {table_name}
FIELDS: {fields}
RELATIONSHIPS: {relationships}
CONSTRAINTS: {constraints}

REQUIREMENTS:
1. SQLAlchemy ORM model
2. Proper field types and constraints
3. Indexes for frequently queried fields
4. Relationships with lazy loading strategy
5. Validation methods
6. Helper methods for common operations
7. __repr__ for debugging

TEMPLATE:
```python
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship, validates
from datetime import datetime
from typing import Optional

from .base import Base

class ModelName(Base):
    \"\"\"
    Model description.

    Attributes:
        id: Primary key
        field1: Description
        field2: Description
    \"\"\"
    __tablename__ = "{table_name}"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Fields
    field1 = Column(String(255), nullable=False, unique=True)
    field2 = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    related_items = relationship("RelatedModel", back_populates="parent")

    # Indexes
    __table_args__ = (
        Index('idx_field1', 'field1'),
    )

    @validates('field1')
    def validate_field1(self, key: str, value: str) -> str:
        \"\"\"Validate field1.\"\"\"
        if not value:
            raise ValueError("field1 cannot be empty")
        return value

    def to_dict(self) -> Dict[str, Any]:
        \"\"\"Convert to dictionary.\"\"\"
        return {{
            'id': self.id,
            'field1': self.field1,
            'field2': self.field2,
            'created_at': self.created_at.isoformat(),
        }}

    def __repr__(self) -> str:
        return f"<ModelName(id={{self.id}}, field1={{self.field1}})>"
```

Implement complete model.""",
    format=PromptFormat.MARKDOWN,
    variables=["model_name", "table_name", "fields", "relationships", "constraints"]
)


# Error Handling Implementation Prompt
ERROR_HANDLING_PROMPT = PromptTemplate(
    template_id="error_handling",
    name="Error Handling Implementation Prompt",
    template_text="""Implement robust error handling for this code.

ORIGINAL CODE:
```python
{original_code}
```

ERROR SCENARIOS: {error_scenarios}

REQUIREMENTS:
1. NO try/except for control flow
2. Explicit validation before operations
3. Return error values (None, False, Error enum) for expected failures
4. Use exceptions ONLY for truly exceptional situations
5. Validate all inputs at function entry
6. Check preconditions explicitly
7. Provide clear error messages

PATTERN:
```python
def function_name(param: Type) -> Optional[ResultType]:
    \"\"\"
    Function with proper error handling.

    Args:
        param: Input parameter

    Returns:
        Result if successful, None if failed
    \"\"\"
    # Validate inputs explicitly
    if param is None:
        return None

    if not is_valid_type(param):
        return None

    if not meets_preconditions(param):
        return None

    # Perform operation
    result = safe_operation(param)

    # Validate result
    if result is None or not is_valid_result(result):
        return None

    return result
```

Rewrite code with proper error handling.""",
    format=PromptFormat.MARKDOWN,
    variables=["original_code", "error_scenarios"]
)


# Refactoring Prompt
REFACTORING_PROMPT = PromptTemplate(
    template_id="refactoring",
    name="Code Refactoring Prompt",
    template_text="""Refactor this code following best practices.

ORIGINAL CODE:
```python
{original_code}
```

ISSUES IDENTIFIED: {issues}
REFACTORING GOALS: {goals}

REFACTORING PRINCIPLES:
1. Single Responsibility Principle - one function, one purpose
2. DRY (Don't Repeat Yourself) - extract common patterns
3. Clear naming - self-documenting code
4. Reduce complexity - break down complex functions
5. Explicit over implicit - clear logic flow
6. Type safety - add type hints
7. Pure functions where possible - no side effects

REFACTORING STEPS:
1. Extract helper functions
2. Rename variables for clarity
3. Add type hints
4. Simplify conditional logic
5. Remove code duplication
6. Add docstrings
7. Optimize algorithms if needed

Provide complete refactored code.""",
    format=PromptFormat.MARKDOWN,
    variables=["original_code", "issues", "goals"]
)


# Data Structure Implementation Prompt
DATA_STRUCTURE_PROMPT = PromptTemplate(
    template_id="data_structure",
    name="Data Structure Implementation Prompt",
    template_text="""Implement a custom data structure.

DATA STRUCTURE: {structure_name}
OPERATIONS: {operations}
TIME COMPLEXITY REQUIREMENTS: {complexity_requirements}

REQUIREMENTS:
1. Implement all required operations
2. Meet complexity requirements
3. Thread-safe if specified
4. Comprehensive docstrings
5. Invariant checking
6. Iterator support if applicable

TEMPLATE:
```python
from typing import Optional, Iterator, Generic, TypeVar
from threading import RLock

T = TypeVar('T')

class DataStructure(Generic[T]):
    \"\"\"
    Data structure description.

    Operations:
    - operation1: O(1) time
    - operation2: O(log n) time

    Invariants:
    - Invariant 1
    - Invariant 2

    Thread-safe: Yes/No
    \"\"\"

    def __init__(self):
        \"\"\"Initialize data structure.\"\"\"
        self._data: List[T] = []
        self._lock = RLock()
        self._size = 0

    def operation1(self, item: T) -> bool:
        \"\"\"
        Operation description.

        Time Complexity: O(1)
        Space Complexity: O(1)

        Args:
            item: Item to process

        Returns:
            True if successful
        \"\"\"
        with self._lock:
            # Check invariants
            self._check_invariants()

            # Perform operation
            result = self._internal_operation(item)

            # Verify invariants maintained
            self._check_invariants()

            return result

    def _check_invariants(self) -> None:
        \"\"\"Verify data structure invariants.\"\"\"
        assert len(self._data) == self._size
        # Additional invariant checks

    def __iter__(self) -> Iterator[T]:
        \"\"\"Iterate over elements.\"\"\"
        return iter(self._data)

    def __len__(self) -> int:
        \"\"\"Get size.\"\"\"
        return self._size
```

Implement complete data structure.""",
    format=PromptFormat.MARKDOWN,
    variables=["structure_name", "operations", "complexity_requirements"]
)


# Async Implementation Prompt
ASYNC_IMPLEMENTATION_PROMPT = PromptTemplate(
    template_id="async_implementation",
    name="Async/Await Implementation Prompt",
    template_text="""Implement asynchronous code using async/await.

FUNCTION: {function_name}
ASYNC OPERATIONS: {async_operations}
CONCURRENCY REQUIREMENTS: {concurrency}

REQUIREMENTS:
1. Use async/await properly
2. Handle concurrent operations with asyncio.gather
3. Proper error handling in async context
4. Timeout management
5. Resource cleanup with async context managers
6. Avoid blocking calls in async functions

TEMPLATE:
```python
import asyncio
from typing import Optional, List, Any
from contextlib import asynccontextmanager

async def async_function(param: Type) -> Optional[ResultType]:
    \"\"\"
    Async function description.

    Args:
        param: Input parameter

    Returns:
        Result or None if failed
    \"\"\"
    # Validate input
    if not param:
        return None

    # Run concurrent operations with timeout
    # We use asyncio.wait which doesn't raise TimeoutError but returns (done, pending)
    # This avoids try/except blocks.
    
    tasks = [
        asyncio.create_task(async_operation1(param)),
        asyncio.create_task(async_operation2(param))
    ]
    
    done, pending = await asyncio.wait(tasks, timeout=30, return_when=asyncio.ALL_COMPLETED)
    
    # Cancel pending tasks
    for task in pending:
        task.cancel()
        
    if pending:
        # Timeout occurred
        return None
        
    # Collect results from done tasks
    results = []
    for task in done:
        if task.cancelled():
            continue
        if task.exception():
            # Handle exception safely
            return None
        results.append(task.result())
        
    if not results:
        return None

    # Process results
    final_result = await process_results(results)
    return final_result

@asynccontextmanager
async def async_resource():
    \"\"\"Async context manager for resource.\"\"\"
    resource = await acquire_resource()
    
    yield resource
    
    # Release resource
    # In zero-error, we assume release is safe or wrapped
    await release_resource(resource)
```

Implement complete async code.""",
    format=PromptFormat.MARKDOWN,
    variables=["function_name", "async_operations", "concurrency"]
)


# Performance Optimization Prompt
PERFORMANCE_OPTIMIZATION_PROMPT = PromptTemplate(
    template_id="performance_optimization",
    name="Performance Optimization Prompt",
    template_text="""Optimize this code for performance.

ORIGINAL CODE:
```python
{original_code}
```

PERFORMANCE PROFILE: {profile_data}
BOTTLENECKS: {bottlenecks}
TARGET IMPROVEMENT: {target}

OPTIMIZATION STRATEGIES:
1. Algorithm complexity reduction
2. Caching computed values
3. Lazy evaluation
4. Vectorization (NumPy)
5. Avoid repeated allocations
6. Use appropriate data structures
7. Profile-guided optimization

MEASURE:
- Time complexity before/after
- Space complexity before/after
- Benchmark results

Provide optimized implementation with performance analysis.""",
    format=PromptFormat.MARKDOWN,
    variables=["original_code", "profile_data", "bottlenecks", "target"]
)


# Export all templates
ALL_CODING_TEMPLATES = {
    "function_implementation": FUNCTION_IMPLEMENTATION_PROMPT,
    "class_implementation": CLASS_IMPLEMENTATION_PROMPT,
    "algorithm_implementation": ALGORITHM_IMPLEMENTATION_PROMPT,
    "api_endpoint": API_ENDPOINT_PROMPT,
    "database_model": DATABASE_MODEL_PROMPT,
    "error_handling": ERROR_HANDLING_PROMPT,
    "refactoring": REFACTORING_PROMPT,
    "data_structure": DATA_STRUCTURE_PROMPT,
    "async_implementation": ASYNC_IMPLEMENTATION_PROMPT,
    "performance_optimization": PERFORMANCE_OPTIMIZATION_PROMPT
}


def get_coding_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get coding prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_CODING_TEMPLATES.get(template_id)


def list_coding_templates() -> List[str]:
    """
    List all available coding template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_CODING_TEMPLATES.keys())
