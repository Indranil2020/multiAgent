"""
Domain module for zero-error software development system.

This module provides domain-specific knowledge, patterns, and specifications
for generating zero-error code across different software domains including
web development, operating systems, databases, games, and ML/AI systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import re


class DomainType(Enum):
    """Types of software domains supported."""
    WEB = "web"
    OPERATING_SYSTEMS = "operating_systems"
    DATABASES = "databases"
    GAMES = "games"
    ML_AI = "ml_ai"
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    CLOUD = "cloud"
    SECURITY = "security"
    NETWORKING = "networking"


class ComplexityLevel(Enum):
    """Complexity levels for domain tasks."""
    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    VERY_COMPLEX = 5


@dataclass
class DomainPattern:
    """
    A domain-specific pattern or best practice.
    
    Attributes:
        name: Pattern name
        description: Pattern description
        domain: Associated domain
        complexity: Pattern complexity
        code_template: Optional code template
        validation_rules: Validation rules for this pattern
        anti_patterns: Common anti-patterns to avoid
        examples: Usage examples
    """
    name: str
    description: str
    domain: DomainType
    complexity: ComplexityLevel
    code_template: str = ""
    validation_rules: List[str] = field(default_factory=list)
    anti_patterns: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if pattern is valid."""
        return bool(self.name and self.description)


@dataclass
class DomainKnowledge:
    """
    Domain-specific knowledge base.
    
    Attributes:
        domain: Domain type
        patterns: Common patterns
        frameworks: Supported frameworks
        languages: Preferred languages
        best_practices: Best practices
        complexity_factors: Factors affecting complexity
        verification_criteria: Domain-specific verification
    """
    domain: DomainType
    patterns: List[DomainPattern] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    complexity_factors: Dict[str, float] = field(default_factory=dict)
    verification_criteria: List[str] = field(default_factory=list)
    
    def get_pattern(self, name: str) -> Tuple[bool, Optional[DomainPattern], str]:
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
    
    def add_pattern(self, pattern: DomainPattern) -> Tuple[bool, str]:
        """
        Add a pattern to knowledge base.
        
        Args:
            pattern: Pattern to add
        
        Returns:
            Tuple of (success, message)
        """
        if not pattern.is_valid():
            return (False, "Invalid pattern")
        
        # Check for duplicates
        for existing in self.patterns:
            if existing.name.lower() == pattern.name.lower():
                return (False, f"Pattern '{pattern.name}' already exists")
        
        self.patterns.append(pattern)
        return (True, f"Pattern '{pattern.name}' added")


class DomainDetector:
    """
    Detects domain from requirements or code.
    
    Uses keyword matching and pattern recognition to identify
    the primary domain of a software project.
    """
    
    def __init__(self):
        """Initialize domain detector with keyword mappings."""
        self.domain_keywords: Dict[DomainType, Set[str]] = {
            DomainType.WEB: {
                "web", "http", "rest", "api", "frontend", "backend",
                "react", "vue", "angular", "django", "flask", "express",
                "html", "css", "javascript", "typescript", "server",
                "client", "browser", "endpoint", "route", "middleware"
            },
            DomainType.OPERATING_SYSTEMS: {
                "kernel", "driver", "filesystem", "process", "thread",
                "memory", "scheduler", "interrupt", "system call",
                "device", "boot", "init", "daemon", "syscall",
                "virtual memory", "page", "segment", "inode"
            },
            DomainType.DATABASES: {
                "database", "sql", "nosql", "query", "table", "index",
                "transaction", "acid", "schema", "migration", "orm",
                "postgresql", "mysql", "mongodb", "cassandra", "redis",
                "join", "foreign key", "primary key", "constraint"
            },
            DomainType.GAMES: {
                "game", "engine", "graphics", "render", "physics",
                "collision", "entity", "component", "scene", "sprite",
                "texture", "shader", "mesh", "animation", "particle",
                "opengl", "vulkan", "directx", "unity", "unreal"
            },
            DomainType.ML_AI: {
                "machine learning", "neural network", "deep learning",
                "model", "training", "inference", "dataset", "feature",
                "tensorflow", "pytorch", "keras", "transformer", "cnn",
                "rnn", "lstm", "attention", "embedding", "gradient"
            }
        }
    
    def detect_domain(
        self,
        requirements: str,
        confidence_threshold: float = 0.3
    ) -> Tuple[bool, Optional[DomainType], float, str]:
        """
        Detect domain from requirements.
        
        Args:
            requirements: Natural language requirements
            confidence_threshold: Minimum confidence to return a domain
        
        Returns:
            Tuple of (success, domain or None, confidence, message)
        """
        if not requirements:
            return (False, None, 0.0, "requirements cannot be empty")
        
        requirements_lower = requirements.lower()
        
        # Count keyword matches for each domain
        domain_scores: Dict[DomainType, int] = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in requirements_lower:
                    score += 1
            
            domain_scores[domain] = score
        
        # Find domain with highest score
        if not domain_scores:
            return (False, None, 0.0, "No domain keywords found")
        
        max_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[max_domain]
        
        if max_score == 0:
            return (False, None, 0.0, "No domain keywords matched")
        
        # Calculate confidence
        total_keywords = sum(len(kw) for kw in self.domain_keywords.values())
        confidence = max_score / total_keywords
        
        if confidence < confidence_threshold:
            return (False, None, confidence, f"Confidence {confidence:.2f} below threshold")
        
        return (True, max_domain, confidence, f"Detected {max_domain.value} with {confidence:.2f} confidence")
    
    def detect_multiple_domains(
        self,
        requirements: str,
        min_confidence: float = 0.1
    ) -> List[Tuple[DomainType, float]]:
        """
        Detect multiple domains (for multi-domain projects).
        
        Args:
            requirements: Natural language requirements
            min_confidence: Minimum confidence to include
        
        Returns:
            List of (domain, confidence) tuples sorted by confidence
        """
        if not requirements:
            return []
        
        requirements_lower = requirements.lower()
        
        # Count keyword matches for each domain
        domain_scores: Dict[DomainType, int] = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in requirements_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        # Calculate confidences
        total_keywords = sum(len(kw) for kw in self.domain_keywords.values())
        
        results = []
        for domain, score in domain_scores.items():
            confidence = score / total_keywords
            if confidence >= min_confidence:
                results.append((domain, confidence))
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class DomainRegistry:
    """
    Central registry for all domain knowledge.
    
    Manages domain-specific knowledge bases and provides
    routing to appropriate domain handlers.
    """
    
    def __init__(self):
        """Initialize domain registry."""
        self.domains: Dict[DomainType, DomainKnowledge] = {}
        self.detector = DomainDetector()
        self._initialize_domains()
    
    def _initialize_domains(self) -> None:
        """Initialize all domain knowledge bases."""
        # Initialize each domain with empty knowledge base
        for domain_type in DomainType:
            self.domains[domain_type] = DomainKnowledge(domain=domain_type)
    
    def register_domain(
        self,
        domain: DomainType,
        knowledge: DomainKnowledge
    ) -> Tuple[bool, str]:
        """
        Register or update domain knowledge.
        
        Args:
            domain: Domain type
            knowledge: Domain knowledge
        
        Returns:
            Tuple of (success, message)
        """
        if knowledge.domain != domain:
            return (False, "Domain mismatch in knowledge base")
        
        self.domains[domain] = knowledge
        return (True, f"Domain {domain.value} registered")
    
    def get_domain_knowledge(
        self,
        domain: DomainType
    ) -> Tuple[bool, Optional[DomainKnowledge], str]:
        """
        Get knowledge for a domain.
        
        Args:
            domain: Domain type
        
        Returns:
            Tuple of (success, knowledge or None, message)
        """
        if domain not in self.domains:
            return (False, None, f"Domain {domain.value} not registered")
        
        return (True, self.domains[domain], "Knowledge retrieved")
    
    def detect_and_get_knowledge(
        self,
        requirements: str
    ) -> Tuple[bool, Optional[DomainKnowledge], str]:
        """
        Detect domain and get its knowledge.
        
        Args:
            requirements: Natural language requirements
        
        Returns:
            Tuple of (success, knowledge or None, message)
        """
        success, domain, confidence, msg = self.detector.detect_domain(requirements)
        
        if not success or not domain:
            return (False, None, msg)
        
        return self.get_domain_knowledge(domain)
    
    def get_all_domains(self) -> List[DomainType]:
        """Get list of all registered domains."""
        return list(self.domains.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with stats
        """
        total_patterns = sum(
            len(knowledge.patterns)
            for knowledge in self.domains.values()
        )
        
        total_frameworks = sum(
            len(knowledge.frameworks)
            for knowledge in self.domains.values()
        )
        
        return {
            "total_domains": len(self.domains),
            "total_patterns": total_patterns,
            "total_frameworks": total_frameworks,
            "domains": [d.value for d in self.domains.keys()]
        }


class DomainRouter:
    """
    Routes tasks to appropriate domain handlers.
    
    Analyzes requirements and routes to the correct domain
    module for specialized processing.
    """
    
    def __init__(self, registry: DomainRegistry):
        """
        Initialize domain router.
        
        Args:
            registry: Domain registry
        """
        self.registry = registry
        self.detector = registry.detector
    
    def route_task(
        self,
        requirements: str,
        preferred_domain: Optional[DomainType] = None
    ) -> Tuple[bool, Optional[DomainType], str]:
        """
        Route task to appropriate domain.
        
        Args:
            requirements: Task requirements
            preferred_domain: Optional preferred domain
        
        Returns:
            Tuple of (success, domain or None, message)
        """
        if not requirements:
            return (False, None, "requirements cannot be empty")
        
        # Use preferred domain if specified
        if preferred_domain:
            success, knowledge, msg = self.registry.get_domain_knowledge(preferred_domain)
            if success:
                return (True, preferred_domain, f"Using preferred domain: {preferred_domain.value}")
            return (False, None, f"Preferred domain not available: {msg}")
        
        # Auto-detect domain
        success, domain, confidence, msg = self.detector.detect_domain(requirements)
        
        if not success or not domain:
            return (False, None, f"Could not detect domain: {msg}")
        
        return (True, domain, f"Routed to {domain.value} (confidence: {confidence:.2f})")
    
    def route_multi_domain_task(
        self,
        requirements: str
    ) -> Tuple[bool, List[DomainType], str]:
        """
        Route task that spans multiple domains.
        
        Args:
            requirements: Task requirements
        
        Returns:
            Tuple of (success, domains list, message)
        """
        if not requirements:
            return (False, [], "requirements cannot be empty")
        
        domain_confidences = self.detector.detect_multiple_domains(requirements)
        
        if not domain_confidences:
            return (False, [], "No domains detected")
        
        domains = [d for d, _ in domain_confidences]
        
        return (True, domains, f"Routed to {len(domains)} domains")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "available_domains": len(self.registry.get_all_domains()),
            "detector_keywords": sum(
                len(keywords)
                for keywords in self.detector.domain_keywords.values()
            )
        }


# Global domain registry instance
_global_registry: Optional[DomainRegistry] = None


def get_domain_registry() -> DomainRegistry:
    """
    Get global domain registry instance.
    
    Returns:
        Global domain registry
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = DomainRegistry()
    
    return _global_registry


def detect_domain(requirements: str) -> Tuple[bool, Optional[DomainType], float, str]:
    """
    Convenience function to detect domain.
    
    Args:
        requirements: Natural language requirements
    
    Returns:
        Tuple of (success, domain or None, confidence, message)
    """
    registry = get_domain_registry()
    return registry.detector.detect_domain(requirements)


def get_domain_knowledge(domain: DomainType) -> Tuple[bool, Optional[DomainKnowledge], str]:
    """
    Convenience function to get domain knowledge.
    
    Args:
        domain: Domain type
    
    Returns:
        Tuple of (success, knowledge or None, message)
    """
    registry = get_domain_registry()
    return registry.get_domain_knowledge(domain)


__all__ = [
    # Enums
    "DomainType",
    "ComplexityLevel",
    # Data classes
    "DomainPattern",
    "DomainKnowledge",
    # Core classes
    "DomainDetector",
    "DomainRegistry",
    "DomainRouter",
    # Convenience functions
    "get_domain_registry",
    "detect_domain",
    "get_domain_knowledge",
]
