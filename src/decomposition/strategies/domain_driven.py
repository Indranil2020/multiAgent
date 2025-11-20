"""
Domain-driven decomposition strategy.

This module implements domain-driven decomposition, breaking tasks by
domain boundaries and bounded contexts.
"""

from dataclasses import dataclass
from typing import List, Tuple, Set
from enum import Enum


class DomainType(Enum):
    """Types of domain contexts."""
    CORE_DOMAIN = "core"
    SUPPORTING_DOMAIN = "supporting"
    GENERIC_DOMAIN = "generic"


@dataclass
class BoundedContext:
    """
    A bounded context in domain-driven design.
    
    Attributes:
        context_id: Unique identifier
        name: Context name
        description: Context description
        domain_type: Type of domain
        responsibilities: List of responsibilities
        entities: Domain entities in this context
        dependencies: Other contexts this depends on
    """
    context_id: str
    name: str
    description: str
    domain_type: DomainType
    responsibilities: List[str]
    entities: List[str]
    dependencies: Set[str]
    
    def is_valid(self) -> bool:
        """Check if bounded context is valid."""
        return bool(
            self.context_id and
            self.name and
            self.description and
            self.responsibilities
        )


class DomainDrivenStrategy:
    """
    Domain-driven decomposition strategy.
    
    This strategy decomposes tasks based on domain boundaries,
    identifying bounded contexts and separating domain logic.
    """
    
    def __init__(self):
        """Initialize domain-driven strategy."""
        self.context_counter = 0
    
    def decompose_by_domain(
        self,
        task_id: str,
        description: str
    ) -> Tuple[bool, List[BoundedContext], str]:
        """
        Decompose task by domain boundaries.
        
        Args:
            task_id: Task ID
            description: Task description
        
        Returns:
            Tuple of (success, list of bounded contexts, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if not description:
            return (False, [], "description cannot be empty")
        
        # Identify bounded contexts
        contexts = self._identify_contexts(task_id, description)
        
        if not contexts:
            return (False, [], "No bounded contexts identified")
        
        return (True, contexts, f"Identified {len(contexts)} bounded contexts")
    
    def _identify_contexts(
        self,
        task_id: str,
        description: str
    ) -> List[BoundedContext]:
        """
        Identify bounded contexts from description.
        
        Args:
            task_id: Task ID
            description: Task description
        
        Returns:
            List of bounded contexts
        """
        contexts = []
        desc_lower = description.lower()
        
        # Identify core domain context
        core_keywords = ['business', 'domain', 'model', 'entity', 'aggregate']
        if any(kw in desc_lower for kw in core_keywords):
            contexts.append(BoundedContext(
                context_id=f"{task_id}_core_domain",
                name="Core Domain",
                description="Core business logic and domain models",
                domain_type=DomainType.CORE_DOMAIN,
                responsibilities=["Business logic", "Domain rules"],
                entities=["DomainModel"],
                dependencies=set()
            ))
        
        # Identify infrastructure context
        infra_keywords = ['database', 'storage', 'persistence', 'repository']
        if any(kw in desc_lower for kw in infra_keywords):
            contexts.append(BoundedContext(
                context_id=f"{task_id}_infrastructure",
                name="Infrastructure",
                description="Infrastructure and persistence layer",
                domain_type=DomainType.SUPPORTING_DOMAIN,
                responsibilities=["Data persistence", "External integrations"],
                entities=["Repository", "DataAccess"],
                dependencies={f"{task_id}_core_domain"} if contexts else set()
            ))
        
        # Identify application context
        app_keywords = ['service', 'application', 'use case', 'workflow']
        if any(kw in desc_lower for kw in app_keywords):
            contexts.append(BoundedContext(
                context_id=f"{task_id}_application",
                name="Application",
                description="Application services and use cases",
                domain_type=DomainType.SUPPORTING_DOMAIN,
                responsibilities=["Orchestration", "Use case implementation"],
                entities=["ApplicationService", "UseCase"],
                dependencies={f"{task_id}_core_domain"} if contexts else set()
            ))
        
        # If no specific contexts identified, create a generic one
        if not contexts:
            contexts.append(BoundedContext(
                context_id=f"{task_id}_main_context",
                name="Main Context",
                description=description,
                domain_type=DomainType.GENERIC_DOMAIN,
                responsibilities=["Primary functionality"],
                entities=["MainEntity"],
                dependencies=set()
            ))
        
        return contexts
    
    def identify_domain_events(
        self,
        context: BoundedContext
    ) -> List[str]:
        """
        Identify domain events for a bounded context.
        
        Args:
            context: Bounded context
        
        Returns:
            List of domain event names
        """
        if not context.is_valid():
            return []
        
        events = []
        
        # Generate events based on context type
        if context.domain_type == DomainType.CORE_DOMAIN:
            for responsibility in context.responsibilities:
                event_name = f"{responsibility.replace(' ', '')}Completed"
                events.append(event_name)
        
        return events
    
    def validate_contexts(
        self,
        contexts: List[BoundedContext]
    ) -> Tuple[bool, str]:
        """
        Validate bounded contexts.
        
        Args:
            contexts: Contexts to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not contexts:
            return (False, "No contexts to validate")
        
        # Check each context is valid
        for context in contexts:
            if not context.is_valid():
                return (False, f"Invalid context: {context.context_id}")
        
        # Check for duplicate IDs
        context_ids = [c.context_id for c in contexts]
        if len(context_ids) != len(set(context_ids)):
            return (False, "Duplicate context IDs found")
        
        # Check dependencies reference existing contexts
        all_ids = set(context_ids)
        for context in contexts:
            for dep_id in context.dependencies:
                if dep_id not in all_ids:
                    return (False, f"Context {context.context_id} has invalid dependency: {dep_id}")
        
        return (True, "Contexts are valid")
    
    def reset_counter(self) -> None:
        """Reset the context counter."""
        self.context_counter = 0
