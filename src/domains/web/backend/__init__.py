"""
Backend framework specifications and patterns.

This module provides comprehensive knowledge for backend development
including Django, Flask, FastAPI, Express, NestJS and associated patterns
for APIs, middleware, authentication, and database integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class BackendFramework(Enum):
    """Backend frameworks."""
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    EXPRESS = "express"
    NESTJS = "nestjs"
    SPRING = "spring"
    RAILS = "rails"
    LARAVEL = "laravel"


class APIStyle(Enum):
    """API architectural styles."""
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    SOAP = "soap"
    WEBSOCKET = "websocket"


class MiddlewareType(Enum):
    """Middleware types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    LOGGING = "logging"
    CORS = "cors"
    RATE_LIMITING = "rate_limiting"
    COMPRESSION = "compression"
    CACHING = "caching"
    ERROR_HANDLING = "error_handling"


@dataclass
class EndpointSpec:
    """
    API endpoint specification.
    
    Attributes:
        path: URL path
        method: HTTP method
        handler: Handler function name
        middleware: Middleware to apply
        auth_required: Requires authentication
        rate_limit: Rate limit config
        request_schema: Request validation schema
        response_schema: Response schema
    """
    path: str
    method: str
    handler: str
    middleware: List[str] = field(default_factory=list)
    auth_required: bool = False
    rate_limit: Optional[Dict[str, int]] = None
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    
    def is_valid(self) -> bool:
        """Check if endpoint spec is valid."""
        valid_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
        return bool(
            self.path and
            self.method and
            self.handler and
            self.method.upper() in valid_methods
        )


@dataclass
class MiddlewareSpec:
    """
    Middleware specification.
    
    Attributes:
        name: Middleware name
        type: Middleware type
        order: Execution order
        config: Configuration options
    """
    name: str
    type: MiddlewareType
    order: int
    config: Dict[str, Any] = field(default_factory=dict)


class BackendKnowledge:
    """
    Backend development knowledge base.
    
    Provides patterns, best practices, and specifications for
    building robust backend applications and APIs.
    """
    
    def __init__(self):
        """Initialize backend knowledge."""
        self.design_patterns: Dict[str, str] = {}
        self.security_patterns: List[str] = []
        self.performance_patterns: List[str] = []
        self._initialize_patterns()
        self._initialize_security()
        self._initialize_performance()
    
    def _initialize_patterns(self) -> None:
        """Initialize design patterns."""
        self.design_patterns = {
            "repository": "Abstract data access layer",
            "service_layer": "Business logic separation",
            "dependency_injection": "Inversion of control",
            "middleware_pipeline": "Request/response processing chain",
            "factory": "Object creation abstraction",
            "strategy": "Algorithm selection at runtime",
            "observer": "Event-driven architecture",
            "decorator": "Extend functionality dynamically"
        }
    
    def _initialize_security(self) -> None:
        """Initialize security patterns."""
        self.security_patterns = [
            "Input validation on all endpoints",
            "Parameterized queries for SQL",
            "Password hashing with bcrypt/argon2",
            "JWT token validation",
            "CORS configuration",
            "Rate limiting per IP/user",
            "HTTPS enforcement",
            "Security headers (CSP, HSTS, etc.)",
            "SQL injection prevention",
            "XSS protection",
            "CSRF tokens for state-changing operations",
            "Secure session management"
        ]
    
    def _initialize_performance(self) -> None:
        """Initialize performance patterns."""
        self.performance_patterns = [
            "Database connection pooling",
            "Query optimization and indexing",
            "Caching with Redis/Memcached",
            "Async/await for I/O operations",
            "Background task queues (Celery/Bull)",
            "Load balancing",
            "Database read replicas",
            "CDN for static assets",
            "Response compression (gzip)",
            "API response pagination"
        ]
    
    def validate_endpoint_spec(
        self,
        spec: EndpointSpec
    ) -> Tuple[bool, List[str], str]:
        """
        Validate an endpoint specification.
        
        Args:
            spec: Endpoint specification
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        if not spec.is_valid():
            errors.append("Invalid endpoint specification")
        
        # Validate path format
        if not spec.path.startswith('/'):
            errors.append("Path must start with /")
        
        # Check authentication for sensitive operations
        if spec.method.upper() in ["POST", "PUT", "DELETE", "PATCH"]:
            if not spec.auth_required:
                errors.append(f"{spec.method} endpoint should require authentication")
        
        # Check rate limiting for public endpoints
        if not spec.auth_required and not spec.rate_limit:
            errors.append("Public endpoint should have rate limiting")
        
        # Validate request schema for POST/PUT/PATCH
        if spec.method.upper() in ["POST", "PUT", "PATCH"]:
            if not spec.request_schema:
                errors.append(f"{spec.method} endpoint should have request schema")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "Endpoint specification valid")
    
    def generate_endpoint_code(
        self,
        spec: EndpointSpec,
        framework: BackendFramework
    ) -> Tuple[bool, str, str]:
        """
        Generate endpoint code from specification.
        
        Args:
            spec: Endpoint specification
            framework: Target framework
        
        Returns:
            Tuple of (success, code, message)
        """
        if not spec.is_valid():
            return (False, "", "Invalid endpoint specification")
        
        if framework == BackendFramework.FASTAPI:
            return self._generate_fastapi_endpoint(spec)
        elif framework == BackendFramework.FLASK:
            return self._generate_flask_endpoint(spec)
        elif framework == BackendFramework.EXPRESS:
            return self._generate_express_endpoint(spec)
        else:
            return (False, "", f"Framework {framework.value} not supported")
    
    def _generate_fastapi_endpoint(
        self,
        spec: EndpointSpec
    ) -> Tuple[bool, str, str]:
        """Generate FastAPI endpoint code."""
        code = "from fastapi import APIRouter, Depends, HTTPException\n"
        code += "from typing import Optional\n\n"
        code += "router = APIRouter()\n\n"
        
        # Generate endpoint
        method = spec.method.lower()
        code += f"@router.{method}('{spec.path}'"
        
        if spec.response_schema:
            code += ", response_model=ResponseSchema"
        
        code += ")\n"
        
        # Add authentication dependency
        if spec.auth_required:
            code += "async def "
            code += f"{spec.handler}(current_user: User = Depends(get_current_user)):\n"
        else:
            code += f"async def {spec.handler}():\n"
        
        code += "    # TODO: Implement endpoint logic\n"
        code += "    pass\n"
        
        return (True, code, "FastAPI endpoint generated")
    
    def _generate_flask_endpoint(
        self,
        spec: EndpointSpec
    ) -> Tuple[bool, str, str]:
        """Generate Flask endpoint code."""
        code = "from flask import Blueprint, request, jsonify\n\n"
        code += "bp = Blueprint('api', __name__)\n\n"
        
        # Generate endpoint
        code += f"@bp.route('{spec.path}', methods=['{spec.method.upper()}'])\n"
        
        if spec.auth_required:
            code += "@login_required\n"
        
        code += f"def {spec.handler}():\n"
        code += "    # TODO: Implement endpoint logic\n"
        code += "    return jsonify({{}}), 200\n"
        
        return (True, code, "Flask endpoint generated")
    
    def _generate_express_endpoint(
        self,
        spec: EndpointSpec
    ) -> Tuple[bool, str, str]:
        """Generate Express endpoint code."""
        code = "const express = require('express');\n"
        code += "const router = express.Router();\n\n"
        
        # Generate endpoint
        method = spec.method.lower()
        code += f"router.{method}('{spec.path}'"
        
        # Add middleware
        if spec.auth_required:
            code += ", authMiddleware"
        
        code += ", async (req, res) => {\n"
        code += "  try {\n"
        code += "    // TODO: Implement endpoint logic\n"
        code += "    res.json({});\n"
        code += "  } catch (error) {\n"
        code += "    res.status(500).json({ error: error.message });\n"
        code += "  }\n"
        code += "});\n\n"
        code += "module.exports = router;\n"
        
        return (True, code, "Express endpoint generated")
    
    def generate_middleware_code(
        self,
        spec: MiddlewareSpec,
        framework: BackendFramework
    ) -> Tuple[bool, str, str]:
        """
        Generate middleware code.
        
        Args:
            spec: Middleware specification
            framework: Target framework
        
        Returns:
            Tuple of (success, code, message)
        """
        if framework == BackendFramework.FASTAPI:
            return self._generate_fastapi_middleware(spec)
        elif framework == BackendFramework.EXPRESS:
            return self._generate_express_middleware(spec)
        else:
            return (False, "", f"Framework {framework.value} not supported")
    
    def _generate_fastapi_middleware(
        self,
        spec: MiddlewareSpec
    ) -> Tuple[bool, str, str]:
        """Generate FastAPI middleware code."""
        code = "from fastapi import Request\n"
        code += "from starlette.middleware.base import BaseHTTPMiddleware\n\n"
        code += f"class {spec.name.capitalize()}Middleware(BaseHTTPMiddleware):\n"
        code += "    async def dispatch(self, request: Request, call_next):\n"
        code += "        # Pre-processing\n"
        code += "        response = await call_next(request)\n"
        code += "        # Post-processing\n"
        code += "        return response\n"
        
        return (True, code, "FastAPI middleware generated")
    
    def _generate_express_middleware(
        self,
        spec: MiddlewareSpec
    ) -> Tuple[bool, str, str]:
        """Generate Express middleware code."""
        code = f"const {spec.name}Middleware = (req, res, next) => {{\n"
        code += "  // Middleware logic\n"
        code += "  next();\n"
        code += "};\n\n"
        code += f"module.exports = {spec.name}Middleware;\n"
        
        return (True, code, "Express middleware generated")
    
    def estimate_api_complexity(
        self,
        num_endpoints: int,
        has_auth: bool,
        has_database: bool,
        has_file_upload: bool,
        has_real_time: bool
    ) -> Tuple[bool, int, str]:
        """
        Estimate API complexity.
        
        Args:
            num_endpoints: Number of endpoints
            has_auth: Has authentication
            has_database: Has database
            has_file_upload: Has file upload
            has_real_time: Has real-time features
        
        Returns:
            Tuple of (success, complexity_score, message)
        """
        complexity = 0
        
        # Base complexity from endpoints
        if num_endpoints > 50:
            complexity += 5
        elif num_endpoints > 20:
            complexity += 4
        elif num_endpoints > 10:
            complexity += 3
        elif num_endpoints > 5:
            complexity += 2
        else:
            complexity += 1
        
        # Add feature complexity
        if has_auth:
            complexity += 3
        if has_database:
            complexity += 2
        if has_file_upload:
            complexity += 2
        if has_real_time:
            complexity += 4
        
        return (True, complexity, f"Estimated complexity: {complexity}")
    
    def get_recommended_middleware(
        self,
        has_auth: bool,
        is_public_api: bool,
        has_file_upload: bool
    ) -> List[MiddlewareType]:
        """
        Get recommended middleware.
        
        Args:
            has_auth: Has authentication
            is_public_api: Is public API
            has_file_upload: Has file upload
        
        Returns:
            List of recommended middleware types
        """
        middleware = [
            MiddlewareType.LOGGING,
            MiddlewareType.ERROR_HANDLING,
            MiddlewareType.COMPRESSION
        ]
        
        if has_auth:
            middleware.append(MiddlewareType.AUTHENTICATION)
            middleware.append(MiddlewareType.AUTHORIZATION)
        
        if is_public_api:
            middleware.append(MiddlewareType.CORS)
            middleware.append(MiddlewareType.RATE_LIMITING)
        
        if has_file_upload:
            # File upload middleware would be added
            pass
        
        return middleware
    
    def validate_security_config(
        self,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str], str]:
        """
        Validate security configuration.
        
        Args:
            config: Security configuration
        
        Returns:
            Tuple of (is_valid, violations list, message)
        """
        violations = []
        
        # Check HTTPS
        if not config.get("https_only", False):
            violations.append("HTTPS not enforced")
        
        # Check CORS
        if "cors" in config:
            if config["cors"].get("allow_all_origins", False):
                violations.append("CORS allows all origins (security risk)")
        
        # Check rate limiting
        if not config.get("rate_limiting", {}):
            violations.append("No rate limiting configured")
        
        # Check authentication
        if config.get("has_auth", False):
            if "jwt_secret" not in config:
                violations.append("JWT secret not configured")
            
            if config.get("password_min_length", 0) < 8:
                violations.append("Password minimum length too short")
        
        if violations:
            return (False, violations, f"Found {len(violations)} security violations")
        
        return (True, [], "Security configuration valid")
    
    def get_performance_recommendations(
        self,
        num_endpoints: int,
        expected_rps: int,
        has_database: bool
    ) -> List[str]:
        """
        Get performance recommendations.
        
        Args:
            num_endpoints: Number of endpoints
            expected_rps: Expected requests per second
            has_database: Has database
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if expected_rps > 1000:
            recommendations.append("Implement load balancing")
            recommendations.append("Use connection pooling")
            recommendations.append("Enable response caching")
        
        if expected_rps > 100:
            recommendations.append("Use async/await for I/O")
            recommendations.append("Implement rate limiting")
        
        if has_database:
            recommendations.append("Use database connection pooling")
            recommendations.append("Optimize database queries")
            recommendations.append("Add database indexes")
            
            if expected_rps > 500:
                recommendations.append("Consider read replicas")
                recommendations.append("Implement query caching")
        
        if num_endpoints > 20:
            recommendations.append("Implement API versioning")
            recommendations.append("Use API documentation (OpenAPI)")
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend knowledge statistics."""
        return {
            "design_patterns": len(self.design_patterns),
            "security_patterns": len(self.security_patterns),
            "performance_patterns": len(self.performance_patterns),
            "supported_frameworks": len(BackendFramework),
            "api_styles": len(APIStyle)
        }


__all__ = [
    # Enums
    "BackendFramework",
    "APIStyle",
    "MiddlewareType",
    # Data classes
    "EndpointSpec",
    "MiddlewareSpec",
    # Main class
    "BackendKnowledge",
]
