"""
Review Prompts Module.

This module provides comprehensive prompt templates for reviewer agents in the
zero-error system. Reviewer agents assess code quality, architecture, security,
and adherence to best practices.

Key Responsibilities:
- Evaluate code quality and maintainability
- Verify architectural patterns and design decisions
- Check security vulnerabilities and risks
- Assess documentation completeness
- Review test coverage and quality
- Validate API design and interfaces
- Ensure performance optimization
- Verify SOLID principles adherence

All review prompts enforce zero-error philosophy with structured assessments.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from .base_prompts import PromptTemplate, PromptFormat


# Code Review Quality Assessment Prompt
CODE_REVIEW_QUALITY_PROMPT = PromptTemplate(
    template_id="code_review_quality",
    name="Code Review Quality Assessment Prompt",
    template_text="""Perform comprehensive code quality review.

CODE TO REVIEW:
```python
{code}
```

REVIEW DIMENSIONS:

1. CORRECTNESS (0-10):
   - Logic correctness
   - Edge case handling
   - Error handling
   - Return value validation

2. READABILITY (0-10):
   - Variable naming clarity
   - Code organization
   - Comment quality
   - Function/class structure

3. MAINTAINABILITY (0-10):
   - Code complexity
   - Modularity
   - Single Responsibility Principle
   - DRY principle adherence

4. PERFORMANCE (0-10):
   - Algorithm efficiency
   - Resource usage
   - Memory management
   - Optimization opportunities

5. SECURITY (0-10):
   - Input validation
   - SQL injection prevention
   - XSS prevention
   - Authentication/authorization

6. TESTING (0-10):
   - Test coverage
   - Test quality
   - Edge case tests
   - Mock/stub usage

ASSESSMENT CRITERIA:
- 9-10: Exceptional - Production ready, best practices
- 7-8: Good - Minor improvements needed
- 5-6: Acceptable - Several improvements needed
- 3-4: Needs work - Major issues present
- 0-2: Unacceptable - Critical issues, not production ready

OUTPUT FORMAT:
```json
{{
    "overall_score": 85,
    "dimensions": {{
        "correctness": {{
            "score": 9,
            "issues": [],
            "strengths": ["Comprehensive input validation", "Clear error handling"],
            "improvements": []
        }},
        "readability": {{
            "score": 8,
            "issues": ["Variable name 'x' is unclear at line 45"],
            "strengths": ["Good function documentation"],
            "improvements": ["Rename 'x' to 'user_count'"]
        }},
        "maintainability": {{"score": 7, "issues": [...], "strengths": [...], "improvements": [...]}},
        "performance": {{"score": 8, "issues": [...], "strengths": [...], "improvements": [...]}},
        "security": {{"score": 9, "issues": [...], "strengths": [...], "improvements": [...]}},
        "testing": {{"score": 7, "issues": [...], "strengths": [...], "improvements": [...]}}
    }},
    "blocking_issues": [],
    "non_blocking_issues": ["Consider adding type hints to helper functions"],
    "recommendations": [
        "Add docstring examples",
        "Consider extracting validation logic to separate function"
    ],
    "approval_status": "APPROVED_WITH_SUGGESTIONS"
}}
```

Perform detailed review and return JSON result.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Best Practices Review Prompt
BEST_PRACTICES_REVIEW_PROMPT = PromptTemplate(
    template_id="best_practices_review",
    name="Best Practices Review Prompt",
    template_text="""Review code adherence to Python best practices.

CODE TO REVIEW:
```python
{code}
```

PYTHON BEST PRACTICES CHECKLIST:

1. PEP 8 STYLE GUIDE:
   - Line length <= 79 characters (code) / 72 (comments)
   - 4 spaces for indentation
   - Blank lines (2 before classes, 1 before methods)
   - Naming conventions (snake_case, CamelCase, UPPER_CASE)
   - Import organization (stdlib, third-party, local)

2. TYPE HINTS:
   - All function parameters typed
   - Return types specified
   - Complex types properly annotated
   - Optional/Union used correctly
   - No type: ignore comments

3. DOCSTRINGS:
   - Module-level docstring present
   - Class docstrings with attributes
   - Function docstrings with Args/Returns/Raises
   - Examples in docstrings
   - Sphinx/Google/NumPy format consistency

4. ERROR HANDLING:
   - Specific exceptions (not bare except)
   - No try/except for control flow
   - Proper exception hierarchy
   - Context managers for resources
   - Cleanup in finally blocks

5. SOLID PRINCIPLES:
   - Single Responsibility (one purpose per function/class)
   - Open/Closed (extensible without modification)
   - Liskov Substitution (subclasses usable as base)
   - Interface Segregation (small, focused interfaces)
   - Dependency Inversion (depend on abstractions)

6. PYTHONIC IDIOMS:
   - List comprehensions where appropriate
   - Context managers (with statement)
   - Generators for large sequences
   - enumerate() instead of range(len())
   - zip() for parallel iteration
   - dict.get() with defaults
   - Truthiness checks (if x: not if x == True:)

7. CODE ORGANIZATION:
   - Imports at top
   - Constants in UPPER_CASE
   - Private methods with underscore prefix
   - @property for computed attributes
   - @dataclass for data containers

EXAMPLE VIOLATIONS:

```python
# BAD: Missing type hints, unclear names
def f(x, y):
    return x + y

# GOOD: Type hints, clear names
def calculate_total(price: float, tax: float) -> float:
    \"\"\"Calculate total price including tax.\"\"\"
    return price + tax

# BAD: Mutable default argument
def add_item(item, items=[]):
    items.append(item)
    return items

# GOOD: None default with initialization
def add_item(item: str, items: Optional[List[str]] = None) -> List[str]:
    \"\"\"Add item to list.\"\"\"
    if items is None:
        items = []
    items.append(item)
    return items
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "pep8_violations": [
        {{"line": 10, "issue": "Line too long (95 > 79 characters)", "severity": "warning"}}
    ],
    "type_hint_issues": [
        {{"line": 25, "issue": "Missing return type", "severity": "error"}}
    ],
    "docstring_issues": [
        {{"line": 15, "issue": "Missing Args section", "severity": "warning"}}
    ],
    "error_handling_issues": [],
    "solid_violations": [
        {{"principle": "SRP", "location": "MyClass", "issue": "Class has 3 responsibilities"}}
    ],
    "pythonic_improvements": [
        {{"line": 40, "current": "for i in range(len(items)):", "suggested": "for i, item in enumerate(items):"}}
    ],
    "score": 75,
    "summary": "Good adherence to best practices with minor improvements needed"
}}
```

Review code and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Architecture Review Prompt
ARCHITECTURE_REVIEW_PROMPT = PromptTemplate(
    template_id="architecture_review",
    name="Architecture Review Prompt",
    template_text="""Review architectural design and patterns.

CODE/DESIGN TO REVIEW:
```python
{code}
```

ARCHITECTURE CONTEXT:
{architecture_context}

ARCHITECTURE REVIEW CRITERIA:

1. DESIGN PATTERNS:
   - Appropriate pattern selection
   - Correct pattern implementation
   - Pattern consistency across codebase
   - Avoiding anti-patterns

2. SEPARATION OF CONCERNS:
   - Clear layer boundaries
   - Minimal coupling between components
   - High cohesion within components
   - Dependency direction (inward)

3. SCALABILITY:
   - Horizontal scaling capability
   - Stateless design where appropriate
   - Efficient resource usage
   - Bottleneck identification

4. MODULARITY:
   - Clear module boundaries
   - Well-defined interfaces
   - Minimal module dependencies
   - Reusable components

5. EXTENSIBILITY:
   - Easy to add new features
   - Plugin architecture if needed
   - Open/Closed principle
   - Configuration-driven behavior

6. DATA FLOW:
   - Clear data flow direction
   - Immutability where appropriate
   - Side effect isolation
   - State management strategy

COMMON ARCHITECTURAL PATTERNS:
- MVC/MVVM (Model-View-Controller/ViewModel)
- Repository Pattern (data access abstraction)
- Service Layer (business logic)
- Factory Pattern (object creation)
- Strategy Pattern (algorithm selection)
- Observer Pattern (event handling)
- Dependency Injection (loose coupling)
- CQRS (Command Query Responsibility Segregation)

ANTI-PATTERNS TO DETECT:
- God Object (class with too many responsibilities)
- Spaghetti Code (complex dependencies)
- Circular Dependencies
- Magic Numbers/Strings
- Hard-coded Configuration
- Tight Coupling
- Global State
- Premature Optimization

OUTPUT FORMAT:
```json
{{
    "overall_assessment": "GOOD",
    "architecture_score": 80,
    "patterns_used": [
        {{"pattern": "Repository", "location": "data/repositories.py", "quality": "well-implemented"}},
        {{"pattern": "Factory", "location": "models/factory.py", "quality": "appropriate"}}
    ],
    "anti_patterns_detected": [
        {{"anti_pattern": "God Object", "location": "services/manager.py", "severity": "high", "description": "Class has 15 methods with 8 different responsibilities"}}
    ],
    "separation_of_concerns": {{
        "score": 7,
        "issues": ["Business logic mixed with data access in UserService"]
    }},
    "scalability_assessment": {{
        "score": 8,
        "strengths": ["Stateless services", "Async I/O"],
        "concerns": ["Single database connection pool"]
    }},
    "modularity": {{
        "score": 9,
        "strengths": ["Clear module boundaries", "Minimal dependencies"]
    }},
    "extensibility": {{
        "score": 8,
        "strengths": ["Plugin system", "Configuration-driven"],
        "improvements": ["Add more extension points"]
    }},
    "recommendations": [
        "Split Manager class into separate services",
        "Introduce service layer to separate business logic",
        "Consider connection pooling for database"
    ],
    "blocking_issues": [],
    "summary": "Solid architecture with good pattern usage. Main concern is God Object anti-pattern in Manager class."
}}
```

Perform architecture review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "architecture_context"]
)


# Documentation Review Prompt
DOCUMENTATION_REVIEW_PROMPT = PromptTemplate(
    template_id="documentation_review",
    name="Documentation Review Prompt",
    template_text="""Review documentation completeness and quality.

CODE TO REVIEW:
```python
{code}
```

DOCUMENTATION REQUIREMENTS:

1. MODULE DOCUMENTATION:
   - Module-level docstring present
   - Purpose clearly stated
   - Key concepts explained
   - Usage examples provided
   - Author and license info

2. CLASS DOCUMENTATION:
   - Class purpose and responsibility
   - Attributes documented with types
   - Usage examples
   - Related classes mentioned
   - Design pattern if applicable

3. FUNCTION DOCUMENTATION:
   - Brief one-line summary
   - Detailed description if complex
   - Args section with types and descriptions
   - Returns section with type and description
   - Raises section for exceptions
   - Examples section with doctests
   - Complexity analysis if relevant

4. INLINE COMMENTS:
   - Complex logic explained
   - Non-obvious decisions justified
   - Not redundant with code
   - Up-to-date and accurate
   - TODO/FIXME tracked

5. TYPE ANNOTATIONS:
   - All parameters typed
   - Return types specified
   - Complex types explained in docstring
   - Generic types parameterized
   - No type: ignore without explanation

6. EXAMPLES AND USAGE:
   - Doctest examples that pass
   - Real-world usage scenarios
   - Edge case examples
   - Integration examples
   - Performance considerations

DOCUMENTATION STYLES:
```python
# GOOGLE STYLE (Preferred)
def function(arg1: str, arg2: int) -> bool:
    \"\"\"Brief description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input

    Examples:
        >>> function("test", 42)
        True
    \"\"\"
    pass

# NUMPY STYLE
def function(arg1, arg2):
    \"\"\"Brief description.

    Parameters
    ----------
    arg1 : str
        Description of arg1
    arg2 : int
        Description of arg2

    Returns
    -------
    bool
        Description of return
    \"\"\"
    pass
```

OUTPUT FORMAT:
```json
{{
    "passed": true/false,
    "documentation_score": 75,
    "module_docs": {{
        "present": true,
        "quality": 8,
        "missing": []
    }},
    "class_docs": [
        {{
            "class": "UserManager",
            "quality": 9,
            "has_attributes": true,
            "has_examples": true,
            "issues": []
        }},
        {{
            "class": "Helper",
            "quality": 4,
            "has_attributes": false,
            "has_examples": false,
            "issues": ["Missing docstring", "No attribute documentation"]
        }}
    ],
    "function_docs": [
        {{
            "function": "process_data",
            "quality": 7,
            "has_args": true,
            "has_returns": true,
            "has_raises": false,
            "has_examples": true,
            "issues": ["Missing Raises section"]
        }}
    ],
    "type_coverage": 85,
    "missing_type_hints": [
        {{"location": "helper.py:45", "item": "parameter 'data'"}}
    ],
    "undocumented_items": [
        "Helper class",
        "_internal_method function",
        "CACHE_SIZE constant"
    ],
    "recommendations": [
        "Add docstring to Helper class",
        "Document _internal_method for maintainability",
        "Add Examples section to process_data",
        "Complete type hints for all parameters"
    ],
    "summary": "Good documentation overall. Missing class docstring for Helper and some type hints."
}}
```

Review documentation and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Security Review Prompt
SECURITY_REVIEW_PROMPT = PromptTemplate(
    template_id="security_review",
    name="Security Review Prompt",
    template_text="""Perform security-focused code review.

CODE TO REVIEW:
```python
{code}
```

SECURITY REVIEW CHECKLIST:

1. INPUT VALIDATION:
   - All user input validated
   - Whitelist validation preferred
   - Type checking enforced
   - Length/range limits applied
   - Special characters handled

2. INJECTION VULNERABILITIES:
   - SQL Injection (parameterized queries)
   - Command Injection (avoid shell=True)
   - LDAP Injection (input sanitization)
   - XPath Injection (parameterized queries)
   - Code Injection (no eval/exec on user input)

3. AUTHENTICATION & AUTHORIZATION:
   - Password hashing (bcrypt, argon2)
   - No hardcoded credentials
   - Secure session management
   - Proper permission checks
   - Multi-factor authentication support

4. CRYPTOGRAPHY:
   - Strong algorithms (AES-256, RSA-2048+)
   - No custom crypto implementations
   - Secure random number generation
   - Proper key management
   - TLS/SSL for data in transit

5. DATA PROTECTION:
   - Sensitive data encrypted at rest
   - Secure data transmission
   - No sensitive data in logs
   - Secure temporary file handling
   - Proper data sanitization

6. ACCESS CONTROL:
   - Principle of least privilege
   - Role-based access control
   - Resource-level permissions
   - No insecure direct object references
   - Path traversal prevention

7. ERROR HANDLING:
   - No sensitive info in error messages
   - Generic error messages to users
   - Detailed logging for debugging
   - No stack traces to users
   - Proper exception handling

8. DEPENDENCY SECURITY:
   - Up-to-date dependencies
   - Known vulnerabilities checked
   - Minimal dependency footprint
   - Trusted sources only
   - License compliance

EXAMPLE VULNERABILITIES:

```python
# CRITICAL: SQL Injection
query = f"SELECT * FROM users WHERE id = {{user_id}}"  # VULNERABLE
cursor.execute(query, (user_id,))  # SAFE

# CRITICAL: Command Injection
os.system(f"ping {{host}}")  # VULNERABLE
subprocess.run(["ping", "-c", "1", validated_host])  # SAFE

# HIGH: Hardcoded Credentials
password = "admin123"  # VULNERABLE
password = os.environ.get("DB_PASSWORD")  # SAFE

# HIGH: Weak Cryptography
hash = hashlib.md5(password)  # VULNERABLE
hash = bcrypt.hashpw(password, bcrypt.gensalt())  # SAFE

# MEDIUM: Path Traversal
file_path = f"/uploads/{{filename}}"  # VULNERABLE
file_path = os.path.join(UPLOAD_DIR, secure_filename(filename))  # SAFE
```

OWASP TOP 10 COVERAGE:
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A04: Insecure Design
- A05: Security Misconfiguration
- A06: Vulnerable and Outdated Components
- A07: Identification and Authentication Failures
- A08: Software and Data Integrity Failures
- A09: Security Logging and Monitoring Failures
- A10: Server-Side Request Forgery (SSRF)

OUTPUT FORMAT:
```json
{{
    "security_score": 65,
    "risk_level": "MEDIUM",
    "critical_vulnerabilities": [
        {{
            "type": "SQL Injection",
            "location": "users.py:45",
            "severity": "CRITICAL",
            "description": "Unsanitized user input in SQL query",
            "remediation": "Use parameterized queries",
            "cwe": "CWE-89"
        }}
    ],
    "high_vulnerabilities": [
        {{
            "type": "Hardcoded Credentials",
            "location": "config.py:12",
            "severity": "HIGH",
            "description": "Database password hardcoded in source",
            "remediation": "Use environment variables or secrets manager",
            "cwe": "CWE-798"
        }}
    ],
    "medium_vulnerabilities": [],
    "low_vulnerabilities": [],
    "input_validation": {{
        "score": 6,
        "issues": ["Missing validation on 'email' parameter"]
    }},
    "authentication": {{
        "score": 8,
        "strengths": ["Password hashing with bcrypt"],
        "issues": ["No rate limiting on login attempts"]
    }},
    "cryptography": {{
        "score": 9,
        "strengths": ["Strong algorithms used"]
    }},
    "access_control": {{
        "score": 7,
        "issues": ["Missing permission check on delete operation"]
    }},
    "recommendations": [
        "Implement parameterized queries for all database operations",
        "Move credentials to environment variables",
        "Add rate limiting to authentication endpoints",
        "Implement permission checks on all write operations"
    ],
    "blocking_issues": ["SQL Injection vulnerability"],
    "must_fix_before_production": true,
    "summary": "Critical SQL injection vulnerability found. Must be fixed before deployment. Authentication is well-implemented."
}}
```

Perform security review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


# Performance Review Prompt
PERFORMANCE_REVIEW_PROMPT = PromptTemplate(
    template_id="performance_review",
    name="Performance Review Prompt",
    template_text="""Review code performance and optimization opportunities.

CODE TO REVIEW:
```python
{code}
```

PERFORMANCE CONTEXT:
Expected Scale: {expected_scale}
Performance Requirements: {performance_requirements}

PERFORMANCE REVIEW AREAS:

1. ALGORITHM COMPLEXITY:
   - Time complexity analysis
   - Space complexity analysis
   - Best/average/worst case scenarios
   - Big-O notation for key operations
   - Comparison with optimal solutions

2. DATA STRUCTURE SELECTION:
   - Appropriate data structure choice
   - Hash tables for O(1) lookup
   - Sets for membership testing
   - Heaps for priority queues
   - Trees for hierarchical data

3. DATABASE OPTIMIZATION:
   - Query efficiency (N+1 problem)
   - Index usage
   - Batch operations
   - Connection pooling
   - Query result caching

4. MEMORY MANAGEMENT:
   - Memory leaks detection
   - Large object allocation
   - Generators for large datasets
   - Object pooling
   - Lazy loading

5. I/O OPTIMIZATION:
   - Async I/O where appropriate
   - Batch operations
   - Caching strategies
   - Connection reuse
   - Streaming for large files

6. CACHING OPPORTUNITIES:
   - Memoization for expensive functions
   - Result caching
   - Static data caching
   - Cache invalidation strategy
   - Cache hit rate optimization

7. CONCURRENCY:
   - Parallelization opportunities
   - Thread safety
   - Lock contention
   - Async/await usage
   - CPU vs I/O bound analysis

EXAMPLE PERFORMANCE ISSUES:

```python
# BAD: O(n^2) - Nested iteration
for i in range(len(items)):
    for j in range(len(items)):
        if items[i] == items[j]:
            process(items[i])

# GOOD: O(n) - Set for deduplication
unique_items = set(items)
for item in unique_items:
    process(item)

# BAD: N+1 Query Problem
users = User.query.all()
for user in users:
    posts = Post.query.filter_by(user_id=user.id).all()  # N queries

# GOOD: Eager loading
users = User.query.options(joinedload(User.posts)).all()  # 1 query

# BAD: Loading entire file into memory
with open('large_file.txt') as f:
    lines = f.readlines()  # Loads all lines
    for line in lines:
        process(line)

# GOOD: Streaming/Generator
with open('large_file.txt') as f:
    for line in f:  # One line at a time
        process(line)

# BAD: Repeated string concatenation
result = ""
for item in items:
    result += str(item)  # Creates new string each time

# GOOD: Join operation
result = "".join(str(item) for item in items)
```

OUTPUT FORMAT:
```json
{{
    "performance_score": 70,
    "overall_assessment": "ACCEPTABLE",
    "complexity_analysis": {{
        "time_complexity": "O(n^2)",
        "space_complexity": "O(n)",
        "optimal_time": "O(n log n)",
        "gap": "Can be improved by one order of magnitude"
    }},
    "bottlenecks": [
        {{
            "location": "process_data:45",
            "issue": "Nested loops creating O(n^2) complexity",
            "impact": "HIGH",
            "current_time": "5000ms for n=1000",
            "expected_after_fix": "50ms",
            "suggestion": "Use hash table for O(1) lookup"
        }}
    ],
    "data_structure_issues": [
        {{
            "location": "find_user:23",
            "current": "List with O(n) search",
            "suggested": "Dictionary with O(1) lookup",
            "impact": "MEDIUM"
        }}
    ],
    "database_issues": [
        {{
            "issue": "N+1 query problem",
            "location": "get_user_posts",
            "current_queries": 101,
            "suggested_queries": 1,
            "fix": "Use eager loading with joinedload()"
        }}
    ],
    "memory_issues": [
        {{
            "issue": "Loading entire 1GB file into memory",
            "location": "process_log_file:10",
            "current_memory": "1GB",
            "suggested_memory": "1MB",
            "fix": "Use generator to stream file"
        }}
    ],
    "caching_opportunities": [
        {{
            "function": "calculate_stats",
            "reason": "Expensive computation with static input",
            "expected_improvement": "95% reduction in compute time",
            "suggestion": "Add @lru_cache decorator"
        }}
    ],
    "concurrency_opportunities": [
        {{
            "location": "fetch_all_data",
            "type": "I/O bound",
            "current": "Sequential network calls",
            "suggested": "Async with asyncio.gather()",
            "expected_improvement": "10x faster"
        }}
    ],
    "optimization_priority": [
        "Fix O(n^2) algorithm in process_data (HIGH)",
        "Resolve N+1 query problem (HIGH)",
        "Add caching to calculate_stats (MEDIUM)",
        "Stream large file instead of loading (MEDIUM)"
    ],
    "estimated_improvement": "80% reduction in execution time",
    "blocking_issues": [],
    "summary": "Main bottleneck is O(n^2) algorithm. Fixing to O(n log n) will provide major improvement. Also resolve N+1 query problem."
}}
```

Perform performance review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "expected_scale", "performance_requirements"]
)


# Pull Request Review Prompt
PULL_REQUEST_REVIEW_PROMPT = PromptTemplate(
    template_id="pull_request_review",
    name="Pull Request Review Prompt",
    template_text="""Review pull request for merge readiness.

PULL REQUEST INFORMATION:
Title: {pr_title}
Description: {pr_description}
Author: {pr_author}
Files Changed: {files_changed}

CODE CHANGES:
```diff
{diff}
```

PULL REQUEST REVIEW CHECKLIST:

1. CHANGE SCOPE:
   - Changes match PR description
   - No unrelated changes included
   - Appropriate scope (not too large)
   - Breaking changes documented
   - Migration path provided if needed

2. CODE QUALITY:
   - Follows coding standards
   - No code smells introduced
   - Proper error handling
   - Test coverage adequate
   - Documentation updated

3. TESTS:
   - New tests added for new features
   - Existing tests still pass
   - Edge cases covered
   - Test quality is good
   - No flaky tests introduced

4. DOCUMENTATION:
   - README updated if needed
   - API documentation updated
   - Breaking changes documented
   - Migration guide provided
   - Changelog updated

5. SECURITY:
   - No security vulnerabilities introduced
   - Input validation present
   - No secrets in code
   - Dependencies are safe
   - Security implications considered

6. PERFORMANCE:
   - No performance regressions
   - Scalability considered
   - Resource usage acceptable
   - Benchmarks provided if relevant
   - Caching appropriate

7. COMPATIBILITY:
   - Backward compatibility maintained
   - Breaking changes justified
   - Deprecation warnings added
   - Version bump appropriate
   - Migration path clear

8. REVIEW FEEDBACK:
   - Previous review comments addressed
   - Discussion resolved
   - Requested changes made
   - Approval from required reviewers
   - CI/CD checks passing

REVIEW OUTCOMES:
- APPROVE: Ready to merge, excellent quality
- APPROVE_WITH_SUGGESTIONS: Can merge, minor improvements suggested
- REQUEST_CHANGES: Must address issues before merge
- REJECT: Fundamental problems, needs redesign

OUTPUT FORMAT:
```json
{{
    "decision": "APPROVE_WITH_SUGGESTIONS",
    "overall_score": 85,
    "change_scope": {{
        "score": 9,
        "matches_description": true,
        "has_unrelated_changes": false,
        "size_appropriate": true
    }},
    "code_quality": {{
        "score": 8,
        "issues": ["Variable naming could be clearer in user_service.py:45"],
        "strengths": ["Good error handling", "Well-structured code"]
    }},
    "tests": {{
        "score": 7,
        "coverage_change": "+5%",
        "new_tests": 12,
        "issues": ["Missing edge case test for empty input"],
        "strengths": ["Comprehensive happy path tests"]
    }},
    "documentation": {{
        "score": 9,
        "updated": ["README.md", "API.md"],
        "issues": [],
        "strengths": ["Clear examples added"]
    }},
    "security": {{
        "score": 10,
        "vulnerabilities": [],
        "strengths": ["Input validation added", "SQL injection prevented"]
    }},
    "performance": {{
        "score": 8,
        "regressions": [],
        "improvements": ["Reduced query count by 50%"],
        "concerns": []
    }},
    "blocking_issues": [],
    "suggestions": [
        "Consider renaming 'data' variable to 'user_data' for clarity",
        "Add test case for empty input scenario",
        "Update CHANGELOG.md with this feature"
    ],
    "required_changes": [],
    "praise": [
        "Excellent test coverage increase",
        "Good documentation with clear examples",
        "Significant performance improvement"
    ],
    "review_comments": [
        {{
            "file": "user_service.py",
            "line": 45,
            "comment": "Consider renaming 'data' to 'user_data' for clarity",
            "severity": "suggestion"
        }}
    ],
    "merge_recommendation": "APPROVE_WITH_SUGGESTIONS",
    "summary": "High-quality PR with good tests and documentation. Minor naming improvement suggested. Ready to merge after addressing suggestions."
}}
```

Perform PR review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["pr_title", "pr_description", "pr_author", "files_changed", "diff"]
)


# API Design Review Prompt
API_DESIGN_REVIEW_PROMPT = PromptTemplate(
    template_id="api_design_review",
    name="API Design Review Prompt",
    template_text="""Review API design for consistency and best practices.

API CODE:
```python
{api_code}
```

API SPECIFICATION:
{api_spec}

API DESIGN REVIEW CRITERIA:

1. RESTful PRINCIPLES:
   - Resource-based URLs (/users, not /getUsers)
   - Proper HTTP methods (GET, POST, PUT, DELETE)
   - Idempotent operations
   - Stateless requests
   - HATEOAS consideration

2. URL DESIGN:
   - Noun-based resources
   - Consistent naming (plural vs singular)
   - Hierarchical structure
   - Query parameters for filtering
   - Versioning strategy

3. REQUEST/RESPONSE FORMAT:
   - Consistent JSON structure
   - Proper status codes
   - Comprehensive error responses
   - Pagination support
   - Field filtering support

4. VALIDATION:
   - Input validation with clear errors
   - Schema validation (Pydantic, JSON Schema)
   - Required vs optional fields
   - Data type enforcement
   - Business rule validation

5. ERROR HANDLING:
   - Consistent error format
   - Appropriate HTTP status codes
   - Detailed error messages
   - Error codes for client handling
   - Stack traces only in debug mode

6. AUTHENTICATION & AUTHORIZATION:
   - Secure authentication scheme
   - Token-based auth (JWT, OAuth2)
   - Proper authorization checks
   - Role-based access control
   - API key management

7. VERSIONING:
   - Clear versioning strategy
   - Backward compatibility
   - Deprecation policy
   - Version in URL or header
   - Changelog maintained

8. DOCUMENTATION:
   - OpenAPI/Swagger spec
   - Example requests/responses
   - Error response documentation
   - Authentication documentation
   - Rate limiting documentation

EXAMPLE GOOD API DESIGN:

```python
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter(prefix="/api/v1")

class UserCreate(BaseModel):
    \"\"\"User creation request.\"\"\"
    email: str = Field(..., description="User email")
    name: str = Field(..., min_length=1, max_length=100)

class UserResponse(BaseModel):
    \"\"\"User response.\"\"\"
    id: int
    email: str
    name: str

class ErrorResponse(BaseModel):
    \"\"\"Error response.\"\"\"
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None

@router.post("/users", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def create_user(user: UserCreate) -> UserResponse:
    \"\"\"Create new user.\"\"\"
    # Implementation
    pass

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page")
) -> List[UserResponse]:
    \"\"\"List users with pagination.\"\"\"
    # Implementation
    pass
```

OUTPUT FORMAT:
```json
{{
    "api_design_score": 80,
    "restful_compliance": {{
        "score": 9,
        "issues": [],
        "strengths": ["Proper HTTP methods", "Resource-based URLs"]
    }},
    "url_design": {{
        "score": 8,
        "issues": [
            {{"url": "/getUser", "issue": "Not noun-based", "suggestion": "/users/{{id}}"}}
        ],
        "strengths": ["Consistent plural naming", "Good hierarchy"]
    }},
    "request_response": {{
        "score": 7,
        "issues": ["Inconsistent pagination format across endpoints"],
        "strengths": ["Clear JSON structure", "Good error responses"]
    }},
    "validation": {{
        "score": 9,
        "strengths": ["Pydantic models for validation", "Clear error messages"],
        "issues": []
    }},
    "error_handling": {{
        "score": 8,
        "strengths": ["Consistent error format", "Proper status codes"],
        "issues": ["Missing error codes for client handling"]
    }},
    "authentication": {{
        "score": 9,
        "strengths": ["JWT-based auth", "Proper authorization checks"],
        "issues": []
    }},
    "versioning": {{
        "score": 10,
        "strategy": "URL-based (/api/v1)",
        "strengths": ["Clear version in URL", "Deprecation policy documented"]
    }},
    "documentation": {{
        "score": 8,
        "has_openapi": true,
        "has_examples": true,
        "issues": ["Rate limiting not documented"]
    }},
    "consistency_issues": [
        "Pagination format differs between /users and /posts",
        "Error response structure not consistent"
    ],
    "recommendations": [
        "Standardize pagination format across all endpoints",
        "Add error codes to all error responses",
        "Document rate limiting policy",
        "Change /getUser to /users/{{id}}"
    ],
    "breaking_changes": [],
    "backward_compatibility": true,
    "summary": "Well-designed API with good RESTful compliance. Main issues are URL naming and pagination consistency."
}}
```

Perform API design review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["api_code", "api_spec"]
)


# Test Coverage Review Prompt
TEST_COVERAGE_REVIEW_PROMPT = PromptTemplate(
    template_id="test_coverage_review",
    name="Test Coverage Review Prompt",
    template_text="""Review test coverage and test quality.

CODE UNDER TEST:
```python
{code}
```

TEST CODE:
```python
{test_code}
```

COVERAGE REPORT:
{coverage_report}

TEST COVERAGE REVIEW CRITERIA:

1. COVERAGE METRICS:
   - Line coverage percentage
   - Branch coverage percentage
   - Function coverage percentage
   - Uncovered critical paths
   - Coverage trend

2. TEST COMPLETENESS:
   - All public functions tested
   - All branches covered
   - Edge cases tested
   - Error conditions tested
   - Boundary values tested

3. TEST QUALITY:
   - Clear test names
   - Independent tests
   - Deterministic tests
   - Fast execution
   - Proper assertions

4. TEST ORGANIZATION:
   - Logical grouping
   - Setup/teardown usage
   - Fixtures for common data
   - Parameterized tests
   - Clear test structure

5. TEST TYPES:
   - Unit tests for functions
   - Integration tests for components
   - End-to-end tests for workflows
   - Performance tests if needed
   - Security tests if needed

6. MOCKING STRATEGY:
   - Appropriate mocking
   - External dependencies mocked
   - Database mocking
   - Time mocking where needed
   - Network mocking

7. TEST MAINTAINABILITY:
   - DRY principle in tests
   - Helper functions extracted
   - Clear test data
   - Easy to update
   - Documentation

OUTPUT FORMAT:
```json
{{
    "test_coverage_score": 85,
    "coverage_metrics": {{
        "line_coverage": 92,
        "branch_coverage": 85,
        "function_coverage": 95,
        "trend": "+5% from last version"
    }},
    "uncovered_critical_code": [
        {{
            "file": "user_service.py",
            "lines": "45-52",
            "function": "delete_user",
            "reason": "No test for user deletion",
            "criticality": "HIGH"
        }}
    ],
    "missing_tests": [
        "No test for empty input in process_data",
        "No test for database connection failure",
        "No test for concurrent access to cache"
    ],
    "test_quality": {{
        "score": 8,
        "strengths": ["Clear test names", "Good use of fixtures"],
        "issues": ["Some tests are interdependent", "Test data could be clearer"]
    }},
    "test_completeness": {{
        "happy_path": 100,
        "edge_cases": 80,
        "error_conditions": 70,
        "boundary_values": 85
    }},
    "test_organization": {{
        "score": 9,
        "strengths": ["Good fixture usage", "Logical grouping"],
        "issues": []
    }},
    "mocking_strategy": {{
        "score": 8,
        "appropriate_mocking": true,
        "issues": ["Database could be mocked in some tests"]
    }},
    "test_types_present": {{
        "unit_tests": true,
        "integration_tests": true,
        "e2e_tests": false,
        "performance_tests": false
    }},
    "recommendations": [
        "Add tests for delete_user function (critical)",
        "Add edge case test for empty input",
        "Add error handling test for database failure",
        "Consider adding E2E tests for critical workflows",
        "Mock database in user_service tests for faster execution"
    ],
    "critical_gaps": [
        "User deletion not tested"
    ],
    "summary": "Good test coverage overall (92% line coverage). Critical gap: user deletion not tested. Add edge case and error handling tests."
}}
```

Perform test coverage review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "test_code", "coverage_report"]
)


# Maintainability Review Prompt
MAINTAINABILITY_REVIEW_PROMPT = PromptTemplate(
    template_id="maintainability_review",
    name="Maintainability Review Prompt",
    template_text="""Review code maintainability and long-term sustainability.

CODE TO REVIEW:
```python
{code}
```

PROJECT CONTEXT:
{project_context}

MAINTAINABILITY REVIEW CRITERIA:

1. CODE COMPLEXITY:
   - Cyclomatic complexity < 10
   - Nesting depth < 4 levels
   - Function length < 50 lines
   - Class size appropriate
   - Parameter count < 5

2. CODE READABILITY:
   - Self-documenting code
   - Clear variable names
   - Consistent formatting
   - Logical organization
   - Minimal cognitive load

3. MODULARITY:
   - Single Responsibility Principle
   - Loose coupling
   - High cohesion
   - Clear interfaces
   - Minimal dependencies

4. EXTENSIBILITY:
   - Easy to add features
   - Open/Closed principle
   - Plugin architecture
   - Configuration over code
   - Strategy pattern usage

5. TESTABILITY:
   - Dependency injection
   - Pure functions
   - Minimal side effects
   - Clear interfaces
   - Mockable dependencies

6. TECHNICAL DEBT:
   - No TODO/FIXME accumulation
   - No deprecated API usage
   - No workarounds
   - Clean git history
   - Regular refactoring

7. CODE DUPLICATION:
   - DRY principle
   - Extracted common logic
   - Shared utilities
   - Inheritance/composition
   - Template methods

8. CONFIGURATION:
   - Externalized configuration
   - Environment-based config
   - No magic numbers
   - Clear defaults
   - Validation present

OUTPUT FORMAT:
```json
{{
    "maintainability_score": 75,
    "maintainability_index": 72,
    "complexity": {{
        "average_cyclomatic": 6,
        "max_cyclomatic": 15,
        "functions_over_threshold": [
            {{"function": "process_data", "complexity": 15, "threshold": 10}}
        ]
    }},
    "readability": {{
        "score": 8,
        "strengths": ["Clear naming", "Good comments"],
        "issues": ["Some functions too long"]
    }},
    "modularity": {{
        "score": 7,
        "coupling": "MEDIUM",
        "cohesion": "HIGH",
        "issues": ["UserService has multiple responsibilities"]
    }},
    "extensibility": {{
        "score": 8,
        "strengths": ["Plugin system", "Config-driven"],
        "issues": ["Hard to add new payment providers"]
    }},
    "testability": {{
        "score": 9,
        "strengths": ["Dependency injection", "Pure functions"],
        "issues": []
    }},
    "technical_debt": {{
        "todo_count": 5,
        "fixme_count": 2,
        "deprecated_usage": ["old_api in payment.py:45"],
        "debt_ratio": "LOW"
    }},
    "code_duplication": {{
        "duplication_percentage": 3,
        "duplicated_blocks": [
            {{"locations": ["user.py:45", "admin.py:67"], "lines": 8}}
        ]
    }},
    "recommendations": [
        "Refactor process_data to reduce complexity from 15 to <10",
        "Split UserService into separate services",
        "Extract duplicated validation logic",
        "Address TODO items in auth.py",
        "Replace deprecated old_api usage"
    ],
    "refactoring_opportunities": [
        {{"location": "UserService", "type": "Split class", "priority": "HIGH"}},
        {{"location": "Validation logic", "type": "Extract to utility", "priority": "MEDIUM"}}
    ],
    "long_term_concerns": [
        "High coupling to legacy payment API",
        "Growing technical debt in auth module"
    ],
    "summary": "Moderate maintainability. Main concerns: high complexity in process_data, UserService has multiple responsibilities. Regular refactoring needed."
}}
```

Perform maintainability review and return assessment.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "project_context"]
)


# Export all templates
ALL_REVIEW_TEMPLATES = {
    "code_review_quality": CODE_REVIEW_QUALITY_PROMPT,
    "best_practices_review": BEST_PRACTICES_REVIEW_PROMPT,
    "architecture_review": ARCHITECTURE_REVIEW_PROMPT,
    "documentation_review": DOCUMENTATION_REVIEW_PROMPT,
    "security_review": SECURITY_REVIEW_PROMPT,
    "performance_review": PERFORMANCE_REVIEW_PROMPT,
    "pull_request_review": PULL_REQUEST_REVIEW_PROMPT,
    "api_design_review": API_DESIGN_REVIEW_PROMPT,
    "test_coverage_review": TEST_COVERAGE_REVIEW_PROMPT,
    "maintainability_review": MAINTAINABILITY_REVIEW_PROMPT
}


def get_review_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get review prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_REVIEW_TEMPLATES.get(template_id)


def list_review_templates() -> List[str]:
    """
    List all available review template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_REVIEW_TEMPLATES.keys())
