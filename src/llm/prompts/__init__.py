"""
Comprehensive prompt template library for zero-error LLM-based code generation.

This module provides a complete set of prompt templates organized into categories:

1. Base Prompts (base_prompts.py):
   - Core system prompts and foundational templates
   - Zero-error system prompt
   - Task specification, contract generation, error analysis
   - Helper functions for few-shot learning, chain-of-thought, self-consistency

2. Coding Prompts (coding_prompts.py):
   - Function, class, and algorithm implementation
   - API endpoints, database models
   - Error handling, refactoring, data structures
   - Async implementation, performance optimization

3. Verification Prompts (verification_prompts.py):
   - 8-layer verification stack
   - Syntax, type checking, contract verification
   - Unit tests, property tests, static analysis
   - Security and performance verification

4. Review Prompts (review_prompts.py):
   - Code quality assessment
   - Best practices, architecture, documentation
   - Security, performance, maintainability reviews
   - Pull request and API design reviews

5. Domain Prompts (domain_prompts/):
   - Web development (React, REST APIs, authentication, security)
   - Database operations (SQL, ORM, migrations, indexing)
   - Game development (game loops, ECS, pathfinding, collision)
   - Operating systems (processes, file I/O, IPC, daemons)

Architecture:
    All prompts are built on the PromptTemplate base class which provides:
    - Template variable substitution with {variable} syntax
    - Validation of required variables
    - Consistent rendering interface
    - Support for multiple output formats (Markdown, JSON, XML)

Usage:
    from llm.prompts import (
        ALL_PROMPTS,
        get_prompt,
        list_prompts,

        # Base prompts
        ZERO_ERROR_SYSTEM_PROMPT,
        TASK_SPECIFICATION_PROMPT,

        # Coding prompts
        FUNCTION_IMPLEMENTATION_PROMPT,
        CLASS_IMPLEMENTATION_PROMPT,

        # Verification prompts
        COMPREHENSIVE_VERIFICATION_PROMPT,

        # Review prompts
        CODE_REVIEW_QUALITY_PROMPT,

        # Domain prompts
        REACT_COMPONENT_PROMPT,
        SQL_QUERY_OPTIMIZATION_PROMPT,
    )

    # Get a specific prompt
    template = get_prompt("function_implementation")

    # Render with variables
    code = template.render(
        function_name="calculate_fibonacci",
        description="Calculate nth Fibonacci number",
        inputs="n: int (non-negative)",
        outputs="int (nth Fibonacci number)",
        constraints="n >= 0, return 0 for n=0, 1 for n=1"
    )

    # List all available prompts
    all_template_ids = list_prompts()

    # List by category
    categories = list_prompts_by_category()
    print(f"Base prompts: {categories['base']}")
    print(f"Coding prompts: {categories['coding']}")
"""

from typing import Dict, List, Optional

# Import base classes and enums
from llm.prompts.base_prompts import (
    PromptTemplate,
    PromptFormat,
)

# Import all base templates
from llm.prompts.base_prompts import (
    ALL_BASE_TEMPLATES,
    ZERO_ERROR_SYSTEM_PROMPT,
    TASK_SPECIFICATION_PROMPT,
    CONTRACT_GENERATION_PROMPT,
    ERROR_ANALYSIS_PROMPT,
    TEST_GENERATION_PROMPT,
    DECOMPOSITION_PROMPT,
    QUALITY_ASSESSMENT_PROMPT,
    create_few_shot_prompt,
    create_chain_of_thought_prompt,
    create_self_consistency_prompt,
    get_template as get_base_template,
    list_templates as list_base_templates,
)

# Import all coding templates
from llm.prompts.coding_prompts import (
    ALL_CODING_TEMPLATES,
    FUNCTION_IMPLEMENTATION_PROMPT,
    CLASS_IMPLEMENTATION_PROMPT,
    ALGORITHM_IMPLEMENTATION_PROMPT,
    API_ENDPOINT_PROMPT,
    DATABASE_MODEL_PROMPT,
    ERROR_HANDLING_PROMPT,
    REFACTORING_PROMPT,
    DATA_STRUCTURE_PROMPT,
    ASYNC_IMPLEMENTATION_PROMPT,
    PERFORMANCE_OPTIMIZATION_PROMPT,
    get_coding_template,
    list_coding_templates,
)

# Import all verification templates
from llm.prompts.verification_prompts import (
    ALL_VERIFICATION_TEMPLATES,
    SYNTAX_VERIFICATION_PROMPT,
    TYPE_CHECKING_PROMPT,
    CONTRACT_VERIFICATION_PROMPT,
    UNIT_TEST_VERIFICATION_PROMPT,
    PROPERTY_TEST_VERIFICATION_PROMPT,
    STATIC_ANALYSIS_PROMPT,
    SECURITY_VERIFICATION_PROMPT,
    PERFORMANCE_VERIFICATION_PROMPT,
    COMPREHENSIVE_VERIFICATION_PROMPT,
    get_verification_template,
    list_verification_templates,
)

# Import all review templates
from llm.prompts.review_prompts import (
    ALL_REVIEW_TEMPLATES,
    CODE_REVIEW_QUALITY_PROMPT,
    BEST_PRACTICES_REVIEW_PROMPT,
    ARCHITECTURE_REVIEW_PROMPT,
    DOCUMENTATION_REVIEW_PROMPT,
    SECURITY_REVIEW_PROMPT,
    PERFORMANCE_REVIEW_PROMPT,
    PULL_REQUEST_REVIEW_PROMPT,
    API_DESIGN_REVIEW_PROMPT,
    TEST_COVERAGE_REVIEW_PROMPT,
    MAINTAINABILITY_REVIEW_PROMPT,
    get_review_template,
    list_review_templates,
)

# Import all domain templates
from llm.prompts.domain_prompts import (
    ALL_DOMAIN_TEMPLATES,
    # Web
    ALL_WEB_TEMPLATES,
    REACT_COMPONENT_PROMPT,
    REST_API_PROMPT,
    AUTHENTICATION_SYSTEM_PROMPT,
    FORM_VALIDATION_PROMPT,
    WEB_SECURITY_PROMPT,
    get_web_template,
    list_web_templates,
    # Database
    ALL_DB_TEMPLATES,
    SQL_QUERY_OPTIMIZATION_PROMPT,
    ORM_MODEL_DESIGN_PROMPT,
    DATABASE_MIGRATION_PROMPT,
    DATABASE_INDEXING_PROMPT,
    TRANSACTION_MANAGEMENT_PROMPT,
    get_db_template,
    list_db_templates,
    # Game
    ALL_GAME_TEMPLATES,
    GAME_LOOP_PROMPT,
    ECS_PROMPT,
    PATHFINDING_PROMPT,
    COLLISION_DETECTION_PROMPT,
    get_game_template,
    list_game_templates,
    # OS
    ALL_OS_TEMPLATES,
    PROCESS_MANAGEMENT_PROMPT,
    FILE_IO_PROMPT,
    IPC_PROMPT,
    DAEMON_PROCESS_PROMPT,
    get_os_template,
    list_os_templates,
    # Scientific
    ALL_SCIENTIFIC_TEMPLATES,
    NUMERICAL_COMPUTING_PROMPT,
    SCIENTIFIC_SIMULATION_PROMPT,
    STATISTICAL_ANALYSIS_PROMPT,
    MACHINE_LEARNING_PIPELINE_PROMPT,
    OPTIMIZATION_ALGORITHM_PROMPT,
    get_scientific_template,
    list_scientific_templates,
    # Domain utilities
    get_domain_template,
    list_domain_templates,
    list_templates_by_domain,
)


# Combine all prompts into a single comprehensive dictionary
ALL_PROMPTS: Dict[str, PromptTemplate] = {
    **ALL_BASE_TEMPLATES,
    **ALL_CODING_TEMPLATES,
    **ALL_VERIFICATION_TEMPLATES,
    **ALL_REVIEW_TEMPLATES,
    **ALL_DOMAIN_TEMPLATES,
}


def get_prompt(template_id: str) -> Optional[PromptTemplate]:
    """
    Get any prompt template by ID.

    Searches across all categories (base, coding, verification, review, domain).

    Args:
        template_id: Template identifier (e.g., 'function_implementation', 'react_component')

    Returns:
        PromptTemplate if found, None otherwise

    Example:
        >>> template = get_prompt("function_implementation")
        >>> if template:
        ...     code = template.render(
        ...         function_name="factorial",
        ...         description="Calculate factorial",
        ...         inputs="n: int",
        ...         outputs="int",
        ...         constraints="n >= 0"
        ...     )
    """
    return ALL_PROMPTS.get(template_id)


def list_prompts() -> List[str]:
    """
    List all available prompt template IDs across all categories.

    Returns:
        List of all template IDs

    Example:
        >>> templates = list_prompts()
        >>> print(f"Total templates: {len(templates)}")
        Total templates: 45
    """
    return list(ALL_PROMPTS.keys())


def list_prompts_by_category() -> Dict[str, List[str]]:
    """
    List prompts organized by category.

    Returns:
        Dictionary mapping category names to lists of template IDs

    Example:
        >>> by_category = list_prompts_by_category()
        >>> print(f"Base: {len(by_category['base'])}")
        >>> print(f"Coding: {len(by_category['coding'])}")
        >>> print(f"Verification: {len(by_category['verification'])}")
        >>> print(f"Review: {len(by_category['review'])}")
        >>> print(f"Domain: {len(by_category['domain'])}")
    """
    return {
        "base": list_base_templates(),
        "coding": list_coding_templates(),
        "verification": list_verification_templates(),
        "review": list_review_templates(),
        "domain": list_domain_templates(),
    }


def get_prompt_statistics() -> Dict[str, int]:
    """
    Get statistics about the prompt library.

    Returns:
        Dictionary with counts of templates in each category

    Example:
        >>> stats = get_prompt_statistics()
        >>> print(f"Total prompts: {stats['total']}")
        >>> print(f"Base: {stats['base']}")
        >>> print(f"Coding: {stats['coding']}")
    """
    categories = list_prompts_by_category()
    return {
        "total": len(ALL_PROMPTS),
        "base": len(categories["base"]),
        "coding": len(categories["coding"]),
        "verification": len(categories["verification"]),
        "review": len(categories["review"]),
        "domain": len(categories["domain"]),
        "web": len(list_web_templates()),
        "database": len(list_db_templates()),
        "game": len(list_game_templates()),
        "os": len(list_os_templates()),
    }


# Export all public symbols
__all__ = [
    # Base classes
    "PromptTemplate",
    "PromptFormat",

    # Combined dictionaries
    "ALL_PROMPTS",
    "ALL_BASE_TEMPLATES",
    "ALL_CODING_TEMPLATES",
    "ALL_VERIFICATION_TEMPLATES",
    "ALL_REVIEW_TEMPLATES",
    "ALL_DOMAIN_TEMPLATES",
    "ALL_WEB_TEMPLATES",
    "ALL_DB_TEMPLATES",
    "ALL_GAME_TEMPLATES",
    "ALL_OS_TEMPLATES",
    "ALL_SCIENTIFIC_TEMPLATES",

    # Base templates
    "ZERO_ERROR_SYSTEM_PROMPT",
    "TASK_SPECIFICATION_PROMPT",
    "CONTRACT_GENERATION_PROMPT",
    "ERROR_ANALYSIS_PROMPT",
    "TEST_GENERATION_PROMPT",
    "DECOMPOSITION_PROMPT",
    "QUALITY_ASSESSMENT_PROMPT",

    # Coding templates
    "FUNCTION_IMPLEMENTATION_PROMPT",
    "CLASS_IMPLEMENTATION_PROMPT",
    "ALGORITHM_IMPLEMENTATION_PROMPT",
    "API_ENDPOINT_PROMPT",
    "DATABASE_MODEL_PROMPT",
    "ERROR_HANDLING_PROMPT",
    "REFACTORING_PROMPT",
    "DATA_STRUCTURE_PROMPT",
    "ASYNC_IMPLEMENTATION_PROMPT",
    "PERFORMANCE_OPTIMIZATION_PROMPT",

    # Verification templates
    "SYNTAX_VERIFICATION_PROMPT",
    "TYPE_CHECKING_PROMPT",
    "CONTRACT_VERIFICATION_PROMPT",
    "UNIT_TEST_VERIFICATION_PROMPT",
    "PROPERTY_TEST_VERIFICATION_PROMPT",
    "STATIC_ANALYSIS_PROMPT",
    "SECURITY_VERIFICATION_PROMPT",
    "PERFORMANCE_VERIFICATION_PROMPT",
    "COMPREHENSIVE_VERIFICATION_PROMPT",

    # Review templates
    "CODE_REVIEW_QUALITY_PROMPT",
    "BEST_PRACTICES_REVIEW_PROMPT",
    "ARCHITECTURE_REVIEW_PROMPT",
    "DOCUMENTATION_REVIEW_PROMPT",
    "SECURITY_REVIEW_PROMPT",
    "PERFORMANCE_REVIEW_PROMPT",
    "PULL_REQUEST_REVIEW_PROMPT",
    "API_DESIGN_REVIEW_PROMPT",
    "TEST_COVERAGE_REVIEW_PROMPT",
    "MAINTAINABILITY_REVIEW_PROMPT",

    # Domain: Web templates
    "REACT_COMPONENT_PROMPT",
    "REST_API_PROMPT",
    "AUTHENTICATION_SYSTEM_PROMPT",
    "FORM_VALIDATION_PROMPT",
    "WEB_SECURITY_PROMPT",

    # Domain: Database templates
    "SQL_QUERY_OPTIMIZATION_PROMPT",
    "ORM_MODEL_DESIGN_PROMPT",
    "DATABASE_MIGRATION_PROMPT",
    "DATABASE_INDEXING_PROMPT",
    "TRANSACTION_MANAGEMENT_PROMPT",

    # Domain: Game templates
    "GAME_LOOP_PROMPT",
    "ECS_PROMPT",
    "PATHFINDING_PROMPT",
    "COLLISION_DETECTION_PROMPT",

    # Domain: OS templates
    "PROCESS_MANAGEMENT_PROMPT",
    "FILE_IO_PROMPT",
    "IPC_PROMPT",
    "DAEMON_PROCESS_PROMPT",

    # Domain: Scientific templates
    "NUMERICAL_COMPUTING_PROMPT",
    "SCIENTIFIC_SIMULATION_PROMPT",
    "STATISTICAL_ANALYSIS_PROMPT",
    "MACHINE_LEARNING_PIPELINE_PROMPT",
    "OPTIMIZATION_ALGORITHM_PROMPT",

    # Helper functions
    "create_few_shot_prompt",
    "create_chain_of_thought_prompt",
    "create_self_consistency_prompt",

    # Getter functions - General
    "get_prompt",
    "list_prompts",
    "list_prompts_by_category",
    "get_prompt_statistics",

    # Getter functions - Category specific
    "get_base_template",
    "list_base_templates",
    "get_coding_template",
    "list_coding_templates",
    "get_verification_template",
    "list_verification_templates",
    "get_review_template",
    "list_review_templates",

    # Getter functions - Domain specific
    "get_domain_template",
    "list_domain_templates",
    "list_templates_by_domain",
    "get_web_template",
    "list_web_templates",
    "get_db_template",
    "list_db_templates",
    "get_game_template",
    "list_game_templates",
    "get_os_template",
    "list_os_templates",
    "get_scientific_template",
    "list_scientific_templates",
]
