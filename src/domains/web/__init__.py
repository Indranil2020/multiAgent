"""
Web development domain knowledge and specifications.

This module provides comprehensive web development patterns, frameworks,
and best practices for generating zero-error web applications including
frontend, backend, APIs, and database integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum


class WebArchitectureType(Enum):
    """Web application architecture types."""
    SPA = "single_page_application"
    MPA = "multi_page_application"
    SSR = "server_side_rendering"
    SSG = "static_site_generation"
    HYBRID = "hybrid"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"


class WebFramework(Enum):
    """Popular web frameworks."""
    # Frontend
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    SVELTE = "svelte"
    # Backend
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    EXPRESS = "express"
    NESTJS = "nestjs"
    SPRING = "spring"
    # Full-stack
    NEXTJS = "nextjs"
    NUXTJS = "nuxtjs"
    REMIX = "remix"


class AuthenticationMethod(Enum):
    """Authentication methods for web apps."""
    JWT = "jwt"
    SESSION = "session"
    OAUTH2 = "oauth2"
    SAML = "saml"
    API_KEY = "api_key"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"


class StateManagementPattern(Enum):
    """State management patterns."""
    REDUX = "redux"
    MOBX = "mobx"
    VUEX = "vuex"
    CONTEXT_API = "context_api"
    RECOIL = "recoil"
    ZUSTAND = "zustand"
    PINIA = "pinia"


@dataclass
class WebPattern:
    """
    Web-specific design pattern.
    
    Attributes:
        name: Pattern name
        category: Pattern category (frontend/backend/api)
        description: Pattern description
        use_cases: When to use this pattern
        implementation: Implementation details
        pros: Advantages
        cons: Disadvantages
        code_example: Code example
    """
    name: str
    category: str
    description: str
    use_cases: List[str] = field(default_factory=list)
    implementation: str = ""
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    code_example: str = ""
    
    def is_valid(self) -> bool:
        """Check if pattern is valid."""
        return bool(self.name and self.category and self.description)


@dataclass
class WebSecurityRequirement:
    """
    Web security requirement.
    
    Attributes:
        name: Requirement name
        description: Description
        severity: Severity level
        mitigation: Mitigation strategy
        validation_rules: Validation rules
    """
    name: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    mitigation: str
    validation_rules: List[str] = field(default_factory=list)


class WebDomainKnowledge:
    """
    Comprehensive web development knowledge base.
    
    Provides patterns, best practices, and specifications for
    building zero-error web applications.
    """
    
    def __init__(self):
        """Initialize web domain knowledge."""
        self.patterns: List[WebPattern] = []
        self.security_requirements: List[WebSecurityRequirement] = []
        self._initialize_patterns()
        self._initialize_security()
    
    def _initialize_patterns(self) -> None:
        """Initialize common web patterns."""
        # MVC Pattern
        mvc = WebPattern(
            name="Model-View-Controller",
            category="architecture",
            description="Separation of data, presentation, and logic",
            use_cases=[
                "Traditional web applications",
                "Server-side rendering",
                "Complex business logic"
            ],
            pros=[
                "Clear separation of concerns",
                "Easier testing",
                "Maintainable codebase"
            ],
            cons=[
                "Can be overkill for simple apps",
                "Learning curve for beginners"
            ]
        )
        self.patterns.append(mvc)
        
        # REST API Pattern
        rest = WebPattern(
            name="RESTful API",
            category="api",
            description="Resource-based API design using HTTP methods",
            use_cases=[
                "Public APIs",
                "Microservices communication",
                "Mobile app backends"
            ],
            pros=[
                "Stateless and scalable",
                "Well-understood conventions",
                "Cacheable responses"
            ],
            cons=[
                "Over-fetching/under-fetching",
                "Multiple round trips"
            ]
        )
        self.patterns.append(rest)
        
        # Component Pattern
        component = WebPattern(
            name="Component-Based Architecture",
            category="frontend",
            description="UI built from reusable components",
            use_cases=[
                "Modern SPAs",
                "Design systems",
                "Reusable UI libraries"
            ],
            pros=[
                "Reusability",
                "Easier testing",
                "Better organization"
            ],
            cons=[
                "Component overhead",
                "Prop drilling issues"
            ]
        )
        self.patterns.append(component)
    
    def _initialize_security(self) -> None:
        """Initialize security requirements."""
        # XSS Protection
        xss = WebSecurityRequirement(
            name="XSS Protection",
            description="Prevent cross-site scripting attacks",
            severity="critical",
            mitigation="Sanitize all user input, use Content Security Policy",
            validation_rules=[
                "No unescaped user input in HTML",
                "CSP headers present",
                "Input validation on all forms"
            ]
        )
        self.security_requirements.append(xss)
        
        # CSRF Protection
        csrf = WebSecurityRequirement(
            name="CSRF Protection",
            description="Prevent cross-site request forgery",
            severity="high",
            mitigation="Use CSRF tokens, SameSite cookies",
            validation_rules=[
                "CSRF tokens on state-changing requests",
                "SameSite cookie attribute set",
                "Origin/Referer header validation"
            ]
        )
        self.security_requirements.append(csrf)
        
        # SQL Injection Protection
        sqli = WebSecurityRequirement(
            name="SQL Injection Protection",
            description="Prevent SQL injection attacks",
            severity="critical",
            mitigation="Use parameterized queries, ORM",
            validation_rules=[
                "No string concatenation in queries",
                "Parameterized queries only",
                "Input validation and sanitization"
            ]
        )
        self.security_requirements.append(sqli)
    
    def get_pattern(self, name: str) -> Tuple[bool, Optional[WebPattern], str]:
        """
        Get a pattern by name.
        
        Args:
            name: Pattern name
        
        Returns:
            Tuple of (success, pattern or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        for pattern in self.patterns:
            if pattern.name.lower() == name.lower():
                return (True, pattern, "Pattern found")
        
        return (False, None, f"Pattern '{name}' not found")
    
    def get_patterns_by_category(self, category: str) -> List[WebPattern]:
        """
        Get all patterns in a category.
        
        Args:
            category: Category name
        
        Returns:
            List of patterns
        """
        if not category:
            return []
        
        return [p for p in self.patterns if p.category.lower() == category.lower()]
    
    def get_security_requirement(
        self,
        name: str
    ) -> Tuple[bool, Optional[WebSecurityRequirement], str]:
        """
        Get a security requirement by name.
        
        Args:
            name: Requirement name
        
        Returns:
            Tuple of (success, requirement or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        for req in self.security_requirements:
            if req.name.lower() == name.lower():
                return (True, req, "Requirement found")
        
        return (False, None, f"Requirement '{name}' not found")
    
    def get_critical_security_requirements(self) -> List[WebSecurityRequirement]:
        """Get all critical security requirements."""
        return [r for r in self.security_requirements if r.severity == "critical"]
    
    def validate_web_spec(
        self,
        spec: Dict[str, Any]
    ) -> Tuple[bool, List[str], str]:
        """
        Validate a web application specification.
        
        Args:
            spec: Specification dictionary
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        # Check required fields
        required_fields = ["name", "type", "framework"]
        for field in required_fields:
            if field not in spec:
                errors.append(f"Missing required field: {field}")
        
        # Validate architecture type
        if "type" in spec:
            valid_types = [t.value for t in WebArchitectureType]
            if spec["type"] not in valid_types:
                errors.append(f"Invalid architecture type: {spec['type']}")
        
        # Validate framework
        if "framework" in spec:
            valid_frameworks = [f.value for f in WebFramework]
            if spec["framework"] not in valid_frameworks:
                errors.append(f"Invalid framework: {spec['framework']}")

        
        # Check security requirements
        if "security" not in spec:
            errors.append("Missing security configuration")
        elif isinstance(spec["security"], dict):
            if "authentication" not in spec["security"]:
                errors.append("Missing authentication method")
            
            if "https_only" not in spec["security"]:
                errors.append("Missing HTTPS requirement")
        
        # Check API specification if present
        if spec.get("has_api", False):
            if "api" not in spec:
                errors.append("API specification missing")
            elif isinstance(spec["api"], dict):
                if "version" not in spec["api"]:
                    errors.append("API version not specified")
                
                if "endpoints" not in spec["api"]:
                    errors.append("API endpoints not specified")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "Specification valid")
    
    def estimate_complexity(
        self,
        spec: Dict[str, Any]
    ) -> Tuple[bool, int, str]:
        """
        Estimate complexity of a web application.
        
        Args:
            spec: Specification dictionary
        
        Returns:
            Tuple of (success, complexity_score, message)
        """
        if not spec:
            return (False, 0, "spec cannot be empty")
        
        complexity = 0
        
        # Base complexity by architecture
        arch_complexity = {
            "single_page_application": 3,
            "multi_page_application": 2,
            "server_side_rendering": 4,
            "static_site_generation": 2,
            "hybrid": 5,
            "microservices": 7,
            "serverless": 6
        }
        
        if "type" in spec:
            complexity += arch_complexity.get(spec["type"], 3)
        
        # Add complexity for features
        if spec.get("has_api", False):
            complexity += 2
        
        if spec.get("has_database", False):
            complexity += 2
        
        if spec.get("has_authentication", False):
            complexity += 3
        
        if spec.get("has_real_time", False):
            complexity += 4
        
        if spec.get("has_file_upload", False):
            complexity += 2
        
        # Add complexity for number of pages/routes
        num_routes = spec.get("num_routes", 0)
        if num_routes > 50:
            complexity += 4
        elif num_routes > 20:
            complexity += 3
        elif num_routes > 10:
            complexity += 2
        elif num_routes > 5:
            complexity += 1
        
        # Add complexity for integrations
        num_integrations = len(spec.get("integrations", []))
        complexity += num_integrations
        
        return (True, complexity, f"Estimated complexity: {complexity}")
    
    def generate_validation_rules(
        self,
        spec: Dict[str, Any]
    ) -> List[str]:
        """
        Generate validation rules for a web spec.
        
        Args:
            spec: Specification dictionary
        
        Returns:
            List of validation rules
        """
        rules = []
        
        # Always include basic security
        rules.append("All user input must be validated")
        rules.append("All output must be escaped")
        rules.append("HTTPS must be enforced")
        
        # Add authentication rules
        if spec.get("has_authentication", False):
            rules.append("Passwords must be hashed with bcrypt/argon2")
            rules.append("Session tokens must be cryptographically secure")
            rules.append("Failed login attempts must be rate-limited")
        
        # Add API rules
        if spec.get("has_api", False):
            rules.append("API must have rate limiting")
            rules.append("API must validate all inputs")
            rules.append("API must use proper HTTP status codes")
        
        # Add database rules
        if spec.get("has_database", False):
            rules.append("Database queries must be parameterized")
            rules.append("Database connections must be pooled")
            rules.append("Database migrations must be reversible")
        
        # Add file upload rules
        if spec.get("has_file_upload", False):
            rules.append("File uploads must validate file type")
            rules.append("File uploads must have size limits")
            rules.append("Uploaded files must be scanned for malware")
        
        return rules
    
    def get_recommended_stack(
        self,
        requirements: str
    ) -> Tuple[bool, Dict[str, str], str]:
        """
        Recommend technology stack based on requirements.
        
        Args:
            requirements: Natural language requirements
        
        Returns:
            Tuple of (success, stack dict, message)
        """
        if not requirements:
            return (False, {}, "requirements cannot be empty")
        
        requirements_lower = requirements.lower()
        
        stack = {}
        
        # Detect frontend needs
        if any(word in requirements_lower for word in ["spa", "single page", "interactive"]):
            if "typescript" in requirements_lower:
                stack["frontend"] = "react_typescript"
            else:
                stack["frontend"] = "react"
        elif "static" in requirements_lower:
            stack["frontend"] = "nextjs_ssg"
        else:
            stack["frontend"] = "react"
        
        # Detect backend needs
        if "python" in requirements_lower:
            if "async" in requirements_lower or "high performance" in requirements_lower:
                stack["backend"] = "fastapi"
            else:
                stack["backend"] = "django"
        elif "node" in requirements_lower or "javascript" in requirements_lower:
            if "typescript" in requirements_lower:
                stack["backend"] = "nestjs"
            else:
                stack["backend"] = "express"
        else:
            stack["backend"] = "fastapi"  # Default
        
        # Detect database needs
        if "nosql" in requirements_lower or "mongodb" in requirements_lower:
            stack["database"] = "mongodb"
        elif "postgres" in requirements_lower:
            stack["database"] = "postgresql"
        elif "mysql" in requirements_lower:
            stack["database"] = "mysql"
        else:
            stack["database"] = "postgresql"  # Default
        
        # Detect caching needs
        if "cache" in requirements_lower or "redis" in requirements_lower:
            stack["cache"] = "redis"
        
        # Detect real-time needs
        if "real-time" in requirements_lower or "websocket" in requirements_lower:
            stack["realtime"] = "websocket"
        
        return (True, stack, f"Recommended stack with {len(stack)} components")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get web domain knowledge statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            "total_patterns": len(self.patterns),
            "patterns_by_category": {
                "architecture": len(self.get_patterns_by_category("architecture")),
                "frontend": len(self.get_patterns_by_category("frontend")),
                "backend": len(self.get_patterns_by_category("backend")),
                "api": len(self.get_patterns_by_category("api"))
            },
            "total_security_requirements": len(self.security_requirements),
            "critical_security_requirements": len(self.get_critical_security_requirements())
        }


__all__ = [
    # Enums
    "WebArchitectureType",
    "WebFramework",
    "AuthenticationMethod",
    "StateManagementPattern",
    # Data classes
    "WebPattern",
    "WebSecurityRequirement",
    # Main class
    "WebDomainKnowledge",
]
