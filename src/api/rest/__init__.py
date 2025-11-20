"""
REST API Module.

This module provides the FastAPI-based REST API for the zero-error system.

Components:
- app: Main FastAPI application
- routes: API endpoint handlers
- middleware: Request/response middleware

Routes:
- /api/v1/tasks: Task management endpoints
- /api/v1/agents: Agent coordination endpoints
- /api/v1/verification: Code verification endpoints
- /api/v1/monitoring: System monitoring endpoints

Usage:
    from api.rest import create_application, APIConfig

    config = APIConfig(
        host="0.0.0.0",
        port=8000,
        debug=False
    )

    app = create_application(config)
"""

from .app import (
    APIApplication,
    APIConfig,
    create_application,
    create_app_with_dependencies
)

from . import routes
from . import middleware

__all__ = [
    # App
    'APIApplication',
    'APIConfig',
    'create_application',
    'create_app_with_dependencies',

    # Submodules
    'routes',
    'middleware',
]
