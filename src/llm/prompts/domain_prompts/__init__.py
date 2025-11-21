"""
Domain-specific prompt templates for specialized code generation.

This module provides domain-specific prompts for:
- Web development (React, REST APIs, authentication, security)
- Database operations (SQL optimization, ORM design, migrations, indexing)
- Game development (game loops, ECS, pathfinding, collision detection)
- Operating systems (process management, file I/O, IPC, daemon processes)
- Scientific computing (numerical methods, simulations, ML pipelines, optimization)

Usage:
    from llm.prompts.domain_prompts import ALL_DOMAIN_TEMPLATES, get_domain_template

    # Get a specific template
    template = get_domain_template("react_component")

    # List all available templates
    template_ids = list_domain_templates()

    # Access by category
    from llm.prompts.domain_prompts import (
        ALL_WEB_TEMPLATES,
        ALL_DB_TEMPLATES,
        ALL_GAME_TEMPLATES,
        ALL_OS_TEMPLATES,
        ALL_SCIENTIFIC_TEMPLATES
    )
"""

from typing import Dict, List, Optional

from ..base_prompts import PromptTemplate

# Import all domain-specific template dictionaries
from .web_prompts import (
    ALL_WEB_TEMPLATES,
    REACT_COMPONENT_PROMPT,
    REST_API_PROMPT,
    AUTHENTICATION_SYSTEM_PROMPT,
    FORM_VALIDATION_PROMPT,
    WEB_SECURITY_PROMPT,
    get_web_template,
    list_web_templates
)

from .db_prompts import (
    ALL_DB_TEMPLATES,
    SQL_QUERY_OPTIMIZATION_PROMPT,
    ORM_MODEL_DESIGN_PROMPT,
    DATABASE_MIGRATION_PROMPT,
    DATABASE_INDEXING_PROMPT,
    TRANSACTION_MANAGEMENT_PROMPT,
    get_db_template,
    list_db_templates
)

from .game_prompts import (
    ALL_GAME_TEMPLATES,
    GAME_LOOP_PROMPT,
    ECS_PROMPT,
    PATHFINDING_PROMPT,
    COLLISION_DETECTION_PROMPT,
    get_game_template,
    list_game_templates
)

from .os_prompts import (
    ALL_OS_TEMPLATES,
    PROCESS_MANAGEMENT_PROMPT,
    FILE_IO_PROMPT,
    IPC_PROMPT,
    DAEMON_PROCESS_PROMPT,
    get_os_template,
    list_os_templates
)

from .scientific_prompts import (
    ALL_SCIENTIFIC_TEMPLATES,
    NUMERICAL_COMPUTING_PROMPT,
    SCIENTIFIC_SIMULATION_PROMPT,
    STATISTICAL_ANALYSIS_PROMPT,
    MACHINE_LEARNING_PIPELINE_PROMPT,
    OPTIMIZATION_ALGORITHM_PROMPT,
    get_scientific_template,
    list_scientific_templates
)


# Combine all domain templates into a single dictionary
ALL_DOMAIN_TEMPLATES: Dict[str, PromptTemplate] = {
    **ALL_WEB_TEMPLATES,
    **ALL_DB_TEMPLATES,
    **ALL_GAME_TEMPLATES,
    **ALL_OS_TEMPLATES,
    **ALL_SCIENTIFIC_TEMPLATES
}


def get_domain_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get a domain-specific prompt template by ID.

    Searches across all domain categories (web, database, game, OS, scientific).

    Args:
        template_id: Template identifier (e.g., 'react_component', 'sql_query_optimization', 'numerical_computing')

    Returns:
        PromptTemplate if found, None otherwise

    Example:
        >>> template = get_domain_template("react_component")
        >>> if template:
        ...     code = template.render(component_name="Button", props="onClick, disabled")
    """
    return ALL_DOMAIN_TEMPLATES.get(template_id)


def list_domain_templates() -> List[str]:
    """
    List all available domain-specific template IDs.

    Returns:
        List of all template IDs across all domain categories

    Example:
        >>> templates = list_domain_templates()
        >>> print(f"Available templates: {len(templates)}")
        Available templates: 23
    """
    return list(ALL_DOMAIN_TEMPLATES.keys())


def list_templates_by_domain() -> Dict[str, List[str]]:
    """
    List templates organized by domain category.

    Returns:
        Dictionary mapping domain names to lists of template IDs

    Example:
        >>> by_domain = list_templates_by_domain()
        >>> print(f"Web templates: {by_domain['web']}")
        Web templates: ['react_component', 'rest_api_endpoint', ...]
    """
    return {
        "web": list_web_templates(),
        "database": list_db_templates(),
        "game": list_game_templates(),
        "os": list_os_templates(),
        "scientific": list_scientific_templates()
    }


# Export all public symbols
__all__ = [
    # Combined dictionary
    "ALL_DOMAIN_TEMPLATES",

    # Category dictionaries
    "ALL_WEB_TEMPLATES",
    "ALL_DB_TEMPLATES",
    "ALL_GAME_TEMPLATES",
    "ALL_OS_TEMPLATES",
    "ALL_SCIENTIFIC_TEMPLATES",

    # Web templates
    "REACT_COMPONENT_PROMPT",
    "REST_API_PROMPT",
    "AUTHENTICATION_SYSTEM_PROMPT",
    "FORM_VALIDATION_PROMPT",
    "WEB_SECURITY_PROMPT",

    # Database templates
    "SQL_QUERY_OPTIMIZATION_PROMPT",
    "ORM_MODEL_DESIGN_PROMPT",
    "DATABASE_MIGRATION_PROMPT",
    "DATABASE_INDEXING_PROMPT",
    "TRANSACTION_MANAGEMENT_PROMPT",

    # Game templates
    "GAME_LOOP_PROMPT",
    "ECS_PROMPT",
    "PATHFINDING_PROMPT",
    "COLLISION_DETECTION_PROMPT",

    # OS templates
    "PROCESS_MANAGEMENT_PROMPT",
    "FILE_IO_PROMPT",
    "IPC_PROMPT",
    "DAEMON_PROCESS_PROMPT",

    # Scientific templates
    "NUMERICAL_COMPUTING_PROMPT",
    "SCIENTIFIC_SIMULATION_PROMPT",
    "STATISTICAL_ANALYSIS_PROMPT",
    "MACHINE_LEARNING_PIPELINE_PROMPT",
    "OPTIMIZATION_ALGORITHM_PROMPT",

    # Getter functions
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
