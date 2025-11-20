"""
API generation and specification patterns.

This module provides comprehensive knowledge for API design and generation
including REST, GraphQL, gRPC specifications, versioning, documentation,
and best practices for building robust APIs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class APIType(Enum):
    """API types."""
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    SOAP = "soap"


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class VersioningStrategy(Enum):
    """API versioning strategies."""
    URL_PATH = "url_path"  # /v1/users
    QUERY_PARAM = "query_param"  # /users?version=1
    HEADER = "header"  # Accept: application/vnd.api.v1+json
    CONTENT_NEGOTIATION = "content_negotiation"


class AuthScheme(Enum):
    """API authentication schemes."""
    BEARER_TOKEN = "bearer"
    API_KEY = "apikey"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    JWT = "jwt"


@dataclass
class APIEndpoint:
    """
    API endpoint specification.
    
    Attributes:
        path: Endpoint path
        method: HTTP method
        summary: Brief description
        description: Detailed description
        parameters: Path/query parameters
        request_body: Request body schema
        responses: Response schemas by status code
        tags: Endpoint tags for grouping
        deprecated: Is deprecated
    """
    path: str
    method: HTTPMethod
    summary: str
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    
    def is_valid(self) -> bool:
        """Check if endpoint is valid."""
        return bool(self.path and self.method and self.summary)


@dataclass
class RateLimitConfig:
    """
    Rate limiting configuration.
    
    Attributes:
        requests_per_minute: Requests allowed per minute
        requests_per_hour: Requests allowed per hour
        burst_size: Burst allowance
        key_type: Rate limit key (ip/user/api_key)
    """
    requests_per_minute: int
    requests_per_hour: int
    burst_size: int = 10
    key_type: str = "ip"


class APIKnowledge:
    """
    API design and generation knowledge base.
    
    Provides patterns, best practices, and specifications for
    building well-designed, documented, and secure APIs.
    """
    
    def __init__(self):
        """Initialize API knowledge."""
        self.rest_patterns: Dict[str, str] = {}
        self.status_codes: Dict[int, str] = {}
        self.best_practices: List[str] = []
        self._initialize_rest_patterns()
        self._initialize_status_codes()
        self._initialize_best_practices()
    
    def _initialize_rest_patterns(self) -> None:
        """Initialize REST API patterns."""
        self.rest_patterns = {
            "resource_naming": "Use nouns, not verbs (users, not getUsers)",
            "plural_resources": "Use plural names for collections",
            "nested_resources": "Limit nesting to 2 levels max",
            "filtering": "Use query parameters for filtering",
            "sorting": "Use ?sort=field:asc/desc",
            "pagination": "Use limit/offset or cursor-based",
            "versioning": "Version your API from day one",
            "idempotency": "POST/PUT/DELETE should be idempotent"
        }
    
    def _initialize_status_codes(self) -> None:
        """Initialize HTTP status codes."""
        self.status_codes = {
            200: "OK - Successful GET/PUT/PATCH",
            201: "Created - Successful POST",
            204: "No Content - Successful DELETE",
            400: "Bad Request - Invalid input",
            401: "Unauthorized - Authentication required",
            403: "Forbidden - Insufficient permissions",
            404: "Not Found - Resource doesn't exist",
            409: "Conflict - Resource conflict",
            422: "Unprocessable Entity - Validation failed",
            429: "Too Many Requests - Rate limit exceeded",
            500: "Internal Server Error - Server error",
            503: "Service Unavailable - Server overloaded"
        }
    
    def _initialize_best_practices(self) -> None:
        """Initialize API best practices."""
        self.best_practices = [
            "Use HTTPS everywhere",
            "Version your API",
            "Use proper HTTP methods",
            "Return appropriate status codes",
            "Provide comprehensive error messages",
            "Implement rate limiting",
            "Use pagination for large datasets",
            "Support filtering and sorting",
            "Provide API documentation",
            "Use consistent naming conventions",
            "Validate all inputs",
            "Use authentication and authorization",
            "Log all API requests",
            "Implement CORS properly",
            "Use caching where appropriate"
        ]
    
    def validate_endpoint(
        self,
        endpoint: APIEndpoint
    ) -> Tuple[bool, List[str], str]:
        """
        Validate an API endpoint.
        
        Args:
            endpoint: API endpoint specification
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        if not endpoint.is_valid():
            errors.append("Invalid endpoint specification")
        
        # Check path format
        if not endpoint.path.startswith('/'):
            errors.append("Path must start with /")
        
        # Check for verbs in path (REST anti-pattern)
        verbs = ["get", "create", "update", "delete", "fetch", "add", "remove"]
        path_lower = endpoint.path.lower()
        for verb in verbs:
            if verb in path_lower:
                errors.append(f"Path contains verb '{verb}' - use nouns instead")
        
        # Check response codes
        if not endpoint.responses:
            errors.append("No response schemas defined")
        else:
            # Check for success response
            has_success = any(200 <= code < 300 for code in endpoint.responses.keys())
            if not has_success:
                errors.append("No success response (2xx) defined")
            
            # Check for error responses
            has_error = any(400 <= code < 600 for code in endpoint.responses.keys())
            if not has_error:
                errors.append("No error response (4xx/5xx) defined")
        
        # Check request body for POST/PUT/PATCH
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
            if not endpoint.request_body:
                errors.append(f"{endpoint.method.value} should have request body schema")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "Endpoint valid")
    
    def generate_openapi_spec(
        self,
        endpoints: List[APIEndpoint],
        title: str,
        version: str,
        description: str = ""
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Generate OpenAPI 3.0 specification.
        
        Args:
            endpoints: List of endpoints
            title: API title
            version: API version
            description: API description
        
        Returns:
            Tuple of (success, spec dict, message)
        """
        if not endpoints:
            return (False, {}, "No endpoints provided")
        
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": description
            },
            "paths": {}
        }
        
        # Group endpoints by path
        for endpoint in endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}
            
            method_lower = endpoint.method.value.lower()
            spec["paths"][endpoint.path][method_lower] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "responses": endpoint.responses
            }
            
            if endpoint.parameters:
                spec["paths"][endpoint.path][method_lower]["parameters"] = endpoint.parameters
            
            if endpoint.request_body:
                spec["paths"][endpoint.path][method_lower]["requestBody"] = endpoint.request_body
            
            if endpoint.tags:
                spec["paths"][endpoint.path][method_lower]["tags"] = endpoint.tags
            
            if endpoint.deprecated:
                spec["paths"][endpoint.path][method_lower]["deprecated"] = True
        
        return (True, spec, f"OpenAPI spec generated with {len(endpoints)} endpoints")
    
    def generate_rest_endpoint_code(
        self,
        endpoint: APIEndpoint,
        language: str = "python"
    ) -> Tuple[bool, str, str]:
        """
        Generate REST endpoint code.
        
        Args:
            endpoint: Endpoint specification
            language: Target language
        
        Returns:
            Tuple of (success, code, message)
        """
        if not endpoint.is_valid():
            return (False, "", "Invalid endpoint specification")
        
        if language == "python":
            return self._generate_python_endpoint(endpoint)
        elif language == "javascript":
            return self._generate_js_endpoint(endpoint)
        else:
            return (False, "", f"Language {language} not supported")
    
    def _generate_python_endpoint(
        self,
        endpoint: APIEndpoint
    ) -> Tuple[bool, str, str]:
        """Generate Python FastAPI endpoint."""
        code = "from fastapi import APIRouter, HTTPException\n"
        code += "from pydantic import BaseModel\n\n"
        code += "router = APIRouter()\n\n"
        
        # Generate request model if needed
        if endpoint.request_body:
            code += "class RequestModel(BaseModel):\n"
            code += "    # TODO: Define request fields\n"
            code += "    pass\n\n"
        
        # Generate endpoint
        method = endpoint.method.value.lower()
        code += f"@router.{method}('{endpoint.path}')\n"
        code += f"async def {endpoint.summary.lower().replace(' ', '_')}("
        
        if endpoint.request_body:
            code += "data: RequestModel"
        
        code += "):\n"
        code += f'    """{endpoint.description}"""\n'
        code += "    # TODO: Implement endpoint logic\n"
        code += "    return {}\n"
        
        return (True, code, "Python endpoint generated")
    
    def _generate_js_endpoint(
        self,
        endpoint: APIEndpoint
    ) -> Tuple[bool, str, str]:
        """Generate JavaScript Express endpoint."""
        code = "const express = require('express');\n"
        code += "const router = express.Router();\n\n"
        
        method = endpoint.method.value.lower()
        code += f"router.{method}('{endpoint.path}', async (req, res) => {{\n"
        code += f"  // {endpoint.description}\n"
        code += "  try {\n"
        code += "    // TODO: Implement endpoint logic\n"
        code += "    res.json({});\n"
        code += "  } catch (error) {\n"
        code += "    res.status(500).json({ error: error.message });\n"
        code += "  }\n"
        code += "});\n\n"
        code += "module.exports = router;\n"
        
        return (True, code, "JavaScript endpoint generated")
    
    def recommend_rate_limits(
        self,
        is_public: bool,
        is_authenticated: bool,
        is_premium: bool = False
    ) -> RateLimitConfig:
        """
        Recommend rate limit configuration.
        
        Args:
            is_public: Is public API
            is_authenticated: Requires authentication
            is_premium: Is premium tier
        
        Returns:
            Rate limit configuration
        """
        if is_premium:
            return RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                burst_size=50,
                key_type="api_key"
            )
        elif is_authenticated:
            return RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=5000,
                burst_size=20,
                key_type="user"
            )
        elif is_public:
            return RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=500,
                burst_size=5,
                key_type="ip"
            )
        else:
            return RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=3000,
                burst_size=10,
                key_type="ip"
            )
    
    def generate_error_response(
        self,
        status_code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate standardized error response.
        
        Args:
            status_code: HTTP status code
            message: Error message
            details: Additional error details
        
        Returns:
            Error response dictionary
        """
        response = {
            "error": {
                "code": status_code,
                "message": message,
                "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
            }
        }
        
        if details:
            response["error"]["details"] = details
        
        return response
    
    def validate_pagination_params(
        self,
        limit: Optional[int],
        offset: Optional[int]
    ) -> Tuple[bool, str]:
        """
        Validate pagination parameters.
        
        Args:
            limit: Page size limit
            offset: Offset
        
        Returns:
            Tuple of (is_valid, message)
        """
        if limit is not None:
            if limit < 1:
                return (False, "limit must be at least 1")
            if limit > 100:
                return (False, "limit cannot exceed 100")
        
        if offset is not None:
            if offset < 0:
                return (False, "offset cannot be negative")
        
        return (True, "Pagination parameters valid")
    
    def generate_pagination_response(
        self,
        items: List[Any],
        total: int,
        limit: int,
        offset: int
    ) -> Dict[str, Any]:
        """
        Generate paginated response.
        
        Args:
            items: Page items
            total: Total items
            limit: Page size
            offset: Current offset
        
        Returns:
            Paginated response dictionary
        """
        return {
            "data": items,
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total
            }
        }
    
    def estimate_api_size(
        self,
        num_resources: int,
        crud_operations: bool = True,
        custom_endpoints: int = 0
    ) -> Tuple[bool, int, str]:
        """
        Estimate number of API endpoints.
        
        Args:
            num_resources: Number of resources
            crud_operations: Include CRUD operations
            custom_endpoints: Number of custom endpoints
        
        Returns:
            Tuple of (success, num_endpoints, message)
        """
        endpoints = 0
        
        if crud_operations:
            # List, Create, Read, Update, Delete per resource
            endpoints += num_resources * 5
        
        endpoints += custom_endpoints
        
        return (True, endpoints, f"Estimated {endpoints} endpoints")
    
    def get_versioning_recommendation(
        self,
        is_public: bool,
        breaking_changes_expected: bool
    ) -> VersioningStrategy:
        """
        Recommend versioning strategy.
        
        Args:
            is_public: Is public API
            breaking_changes_expected: Expect breaking changes
        
        Returns:
            Recommended versioning strategy
        """
        if is_public and breaking_changes_expected:
            return VersioningStrategy.URL_PATH
        elif is_public:
            return VersioningStrategy.HEADER
        else:
            return VersioningStrategy.URL_PATH
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API knowledge statistics."""
        return {
            "rest_patterns": len(self.rest_patterns),
            "status_codes": len(self.status_codes),
            "best_practices": len(self.best_practices),
            "supported_api_types": len(APIType)
        }


__all__ = [
    # Enums
    "APIType",
    "HTTPMethod",
    "VersioningStrategy",
    "AuthScheme",
    # Data classes
    "APIEndpoint",
    "RateLimitConfig",
    # Main class
    "APIKnowledge",
]
