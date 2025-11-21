"""
Verification Prompts Module.

This module provides comprehensive prompt templates for verifier agents in the
zero-error system. Verifier agents validate code through 8 verification layers.

Verification Layers:
1. Syntax - Syntactically correct Python
2. Type Checking - Type annotations and consistency
3. Contracts - Preconditions, postconditions, invariants
4. Unit Tests - Function-level correctness
5. Property Tests - Universal properties hold
6. Static Analysis - Code quality and patterns
7. Security - Vulnerability scanning
8. Performance - Efficiency and optimization

All verification follows zero-error philosophy with explicit pass/fail criteria.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .base_prompts import PromptTemplate, PromptFormat


# Syntax Verification Prompt
SYNTAX_VERIFICATION_PROMPT = PromptTemplate(
    template_id="syntax_verification",
    name="Syntax Verification Prompt",
    template_text="""Verify Python syntax correctness.

CODE TO VERIFY:
```python
{code}
```

VERIFICATION CHECKLIST:
1. Valid Python syntax (no SyntaxError)
2. Proper indentation
3. Balanced parentheses/brackets/braces
4. Valid string literals
5. Correct operator usage
6. No invalid tokens
7. Proper statement structure

VERIFICATION PROCESS:
1. Attempt to parse code with ast.parse()
2. Check for syntax errors
3. Identify line and character of any errors
4. Provide specific error message

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "errors": [
        {{
            "line": 10,
            "column": 5,
            "message": "Error description",
            "code": "relevant code snippet"
        }}
    ],
    "summary": "Brief summary of verification"
}}
```

Verify now and return JSON result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Type Checking Verification Prompt
TYPE_CHECKING_PROMPT = PromptTemplate(
    template_id="type_checking",
    name="Type Checking Verification Prompt",
    template_text="""Verify type annotations and type correctness.

CODE TO VERIFY:
```python
{code}
```

TYPE CHECKING REQUIREMENTS:
1. All function parameters have type hints
2. All function return types specified
3. Class attributes have type hints
4. Type consistency throughout code
5. No type: ignore comments (fix types instead)
6. Generic types properly parameterized
7. Union/Optional types used correctly

VERIFICATION PROCESS:
1. Check for missing type annotations
2. Verify type consistency
3. Check type compatibility in assignments
4. Validate generic type parameters
5. Ensure proper Optional/Union usage

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "errors": [
        {{
            "line": 15,
            "issue": "Missing type hint",
            "suggestion": "Add type hint: param: str",
            "severity": "error"
        }}
    ],
    "coverage": 95,  // percentage of typed code
    "summary": "Type checking summary"
}}
```

Perform type checking and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Contract Verification Prompt
CONTRACT_VERIFICATION_PROMPT = PromptTemplate(
    template_id="contract_verification",
    name="Contract Verification Prompt",
    template_text="""Verify formal contracts (preconditions, postconditions, invariants).

CODE TO VERIFY:
```python
{code}
```

CONTRACT REQUIREMENTS:
1. Preconditions: Input validation at function start
2. Postconditions: Output guarantees before return
3. Invariants: Conditions that always hold
4. Clear contract documentation

VERIFICATION CHECKLIST:
1. All inputs validated before use
2. All edge cases have guards
3. Return values validated before return
4. Invariants checked after mutations
5. Contracts documented in docstring

EXAMPLE VALID CONTRACT:
```python
def divide(a: float, b: float) -> Optional[float]:
    \"\"\"
    Divide a by b.

    Preconditions:
    - b != 0

    Postconditions:
    - result is None if b == 0
    - result == a / b if b != 0

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result or None if invalid
    \"\"\"
    # Precondition check
    if b == 0:
        return None

    result = a / b

    # Postcondition check
    assert result == a / b

    return result
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "missing_preconditions": ["function1: no input validation"],
    "missing_postconditions": ["function2: no output validation"],
    "missing_invariants": ["class: no invariant checks"],
    "violations": [],
    "summary": "Contract verification summary"
}}
```

Verify contracts and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Unit Test Verification Prompt
UNIT_TEST_VERIFICATION_PROMPT = PromptTemplate(
    template_id="unit_test_verification",
    name="Unit Test Verification Prompt",
    template_text="""Verify unit test completeness and correctness.

CODE UNDER TEST:
```python
{code}
```

TEST CODE:
```python
{test_code}
```

TEST REQUIREMENTS:
1. Test all public functions/methods
2. Test happy path scenarios
3. Test edge cases
4. Test error conditions
5. Test boundary values
6. Achieve 100% code coverage
7. Tests are independent
8. Tests are deterministic

COVERAGE ANALYSIS:
1. Line coverage percentage
2. Branch coverage percentage
3. Untested code paths
4. Missing edge case tests

EXAMPLE COMPLETE TESTS:
```python
import pytest

def test_function_happy_path():
    \"\"\"Test normal operation.\"\"\"
    result = function(valid_input)
    assert result == expected_output

def test_function_empty_input():
    \"\"\"Test empty input edge case.\"\"\"
    result = function([])
    assert result == expected_empty_result

def test_function_invalid_input():
    \"\"\"Test invalid input handling.\"\"\"
    result = function(None)
    assert result is None

def test_function_boundary():
    \"\"\"Test boundary values.\"\"\"
    result = function(MAX_VALUE)
    assert result is not None
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "line_coverage": 95,
    "branch_coverage": 90,
    "untested_functions": ["function1", "function2"],
    "missing_tests": [
        "function1: no edge case tests",
        "function2: no error condition tests"
    ],
    "test_quality_issues": [],
    "summary": "Test verification summary"
}}
```

Verify tests and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "test_code"]
)


# Property Test Verification Prompt
PROPERTY_TEST_VERIFICATION_PROMPT = PromptTemplate(
    template_id="property_test_verification",
    name="Property Test Verification Prompt",
    template_text="""Verify property-based tests for universal properties.

CODE TO VERIFY:
```python
{code}
```

PROPERTY TEST REQUIREMENTS:
1. Universal properties identified
2. Hypothesis tests for properties
3. Properties hold for all valid inputs
4. Counterexamples found if properties violated

COMMON PROPERTIES:
- Idempotence: f(f(x)) == f(x)
- Commutativity: f(a, b) == f(b, a)
- Associativity: f(a, f(b, c)) == f(f(a, b), c)
- Inverse: f(inverse_f(x)) == x
- Invariants: property(before) == property(after)

EXAMPLE PROPERTY TESTS:
```python
from hypothesis import given, strategies as st

@given(st.integers())
def test_absolute_value_idempotent(x):
    \"\"\"abs(abs(x)) should equal abs(x).\"\"\"
    assert abs(abs(x)) == abs(x)

@given(st.lists(st.integers()))
def test_sort_idempotent(lst):
    \"\"\"Sorting twice should equal sorting once.\"\"\"
    assert sorted(sorted(lst)) == sorted(lst)

@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    \"\"\"Addition should be commutative.\"\"\"
    assert a + b == b + a
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "properties_identified": [
        "function1: idempotence",
        "function2: commutativity"
    ],
    "properties_tested": ["function1: idempotence"],
    "untested_properties": ["function2: commutativity"],
    "property_violations": [],
    "counterexamples": [],
    "summary": "Property verification summary"
}}
```

Verify properties and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Static Analysis Verification Prompt
STATIC_ANALYSIS_PROMPT = PromptTemplate(
    template_id="static_analysis",
    name="Static Analysis Verification Prompt",
    template_text="""Perform static code analysis for quality and patterns.

CODE TO ANALYZE:
```python
{code}
```

ANALYSIS DIMENSIONS:
1. Code complexity (cyclomatic complexity)
2. Code duplication
3. Dead code detection
4. Unused variables/imports
5. Code smells
6. Design pattern violations
7. Maintainability index

QUALITY METRICS:
- Cyclomatic Complexity: < 10 per function
- Lines per Function: < 50
- Parameters per Function: < 5
- Nesting Depth: < 4
- Maintainability Index: > 70

CODE SMELLS TO DETECT:
- Long functions (> 50 lines)
- Long parameter lists (> 5 params)
- Deep nesting (> 4 levels)
- Duplicated code
- God classes (too many responsibilities)
- Magic numbers
- Complex conditionals

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "complexity_violations": [
        {{
            "function": "process_data",
            "complexity": 15,
            "threshold": 10,
            "line": 50
        }}
    ],
    "code_smells": [
        {{
            "type": "long_function",
            "location": "function_name:100",
            "severity": "warning"
        }}
    ],
    "dead_code": ["unused_function"],
    "duplications": [],
    "maintainability_index": 75,
    "summary": "Static analysis summary"
}}
```

Perform analysis and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Security Verification Prompt
SECURITY_VERIFICATION_PROMPT = PromptTemplate(
    template_id="security_verification",
    name="Security Verification Prompt",
    template_text="""Verify code security and identify vulnerabilities.

CODE TO VERIFY:
```python
{code}
```

SECURITY CHECKS:
1. SQL Injection vulnerabilities
2. Command Injection vulnerabilities
3. Path Traversal vulnerabilities
4. Insecure Deserialization
5. Hardcoded credentials/secrets
6. Weak cryptography
7. Insecure random number generation
8. Unvalidated redirects
9. XML External Entity (XXE)
10. SSRF vulnerabilities

OWASP TOP 10:
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A04: Insecure Design
- A05: Security Misconfiguration
- A06: Vulnerable Components
- A07: Authentication Failures
- A08: Software Integrity Failures
- A09: Logging Failures
- A10: SSRF

EXAMPLE VULNERABILITIES:
```python
# VULNERABLE: SQL Injection
query = f"SELECT * FROM users WHERE id = {{user_id}}"

# SECURE: Parameterized query
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_id,))

# VULNERABLE: Command Injection
os.system(f"ping {{user_input}}")

# SECURE: Validate and use array
subprocess.run(["ping", "-c", "1", validated_host])
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "vulnerabilities": [
        {{
            "type": "SQL Injection",
            "severity": "critical",
            "line": 45,
            "description": "Unsanitized user input in SQL query",
            "remediation": "Use parameterized queries"
        }}
    ],
    "hardcoded_secrets": [],
    "insecure_functions": [],
    "risk_score": 75,  // 0-100, lower is better
    "summary": "Security verification summary"
}}
```

Verify security and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Performance Verification Prompt
PERFORMANCE_VERIFICATION_PROMPT = PromptTemplate(
    template_id="performance_verification",
    name="Performance Verification Prompt",
    template_text="""Verify code performance and identify bottlenecks.

CODE TO VERIFY:
```python
{code}
```

PERFORMANCE CRITERIA:
1. Time complexity analysis
2. Space complexity analysis
3. Algorithm efficiency
4. Memory allocations
5. Loop optimizations
6. Data structure selection
7. Caching opportunities

PERFORMANCE ISSUES:
- O(n^2) algorithms where O(n log n) possible
- Repeated expensive operations in loops
- Unnecessary memory allocations
- Missing caching for computed values
- Inefficient data structures
- String concatenation in loops
- Recursive functions without memoization

EXAMPLE OPTIMIZATIONS:
```python
# INEFFICIENT: O(n^2)
for i in range(len(lst)):
    if item in lst:  # O(n) each iteration
        process(item)

# EFFICIENT: O(n)
item_set = set(lst)  # O(n) once
for i in range(len(lst)):
    if item in item_set:  # O(1) each iteration
        process(item)

# INEFFICIENT: Repeated allocations
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# EFFICIENT: Single allocation
result = "".join(str(item) for item in items)
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "time_complexity": {{
        "function1": "O(n^2)",
        "function2": "O(n log n)"
    }},
    "space_complexity": {{
        "function1": "O(n)",
        "function2": "O(1)"
    }},
    "bottlenecks": [
        {{
            "function": "process_data",
            "issue": "Nested loops creating O(n^2) complexity",
            "line": 50,
            "suggestion": "Use hash table for O(n) lookup"
        }}
    ],
    "optimization_opportunities": [],
    "performance_score": 85,  // 0-100, higher is better
    "summary": "Performance verification summary"
}}
```

Verify performance and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Comprehensive Verification Prompt
COMPREHENSIVE_VERIFICATION_PROMPT = PromptTemplate(
    template_id="comprehensive_verification",
    name="Comprehensive Verification Prompt",
    template_text="""Perform comprehensive verification across all 8 layers.

CODE TO VERIFY:
```python
{code}
```

VERIFICATION LAYERS:
1. Syntax Verification
2. Type Checking
3. Contract Verification
4. Unit Test Verification
5. Property Test Verification
6. Static Analysis
7. Security Verification
8. Performance Verification

For each layer, verify and provide:
- Pass/Fail status
- List of issues found
- Severity of issues
- Remediation suggestions

OUTPUT FORMAT:
```json
{{
    "overall_passed": true/false,
    "layers": {{
        "syntax": {{
            "passed": true,
            "errors": [],
            "score": 100
        }},
        "type_checking": {{
            "passed": true,
            "errors": [],
            "score": 95
        }},
        "contracts": {{
            "passed": false,
            "errors": ["Missing precondition check"],
            "score": 70
        }},
        "unit_tests": {{"passed": true, "errors": [], "score": 100}},
        "property_tests": {{"passed": true, "errors": [], "score": 90}},
        "static_analysis": {{"passed": true, "errors": [], "score": 85}},
        "security": {{"passed": true, "errors": [], "score": 100}},
        "performance": {{"passed": true, "errors": [], "score": 90}}
    }},
    "overall_score": 91,
    "blocking_issues": [],
    "summary": "Verification summary"
}}
```

Perform comprehensive verification and return result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Export all templates
ALL_VERIFICATION_TEMPLATES = {
    "syntax_verification": SYNTAX_VERIFICATION_PROMPT,
    "type_checking": TYPE_CHECKING_PROMPT,
    "contract_verification": CONTRACT_VERIFICATION_PROMPT,
    "unit_test_verification": UNIT_TEST_VERIFICATION_PROMPT,
    "property_test_verification": PROPERTY_TEST_VERIFICATION_PROMPT,
    "static_analysis": STATIC_ANALYSIS_PROMPT,
    "security_verification": SECURITY_VERIFICATION_PROMPT,
    "performance_verification": PERFORMANCE_VERIFICATION_PROMPT,
    "comprehensive_verification": COMPREHENSIVE_VERIFICATION_PROMPT
}


def get_verification_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get verification prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_VERIFICATION_TEMPLATES.get(template_id)


def list_verification_templates() -> List[str]:
    """
    List all available verification template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_VERIFICATION_TEMPLATES.keys())
