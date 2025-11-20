"""
Decomposition strategies for task breakdown.

This package provides various strategies for decomposing tasks including
hierarchical, functional, domain-driven, and atomic task creation.
"""

from .atomic import (
    AtomicTaskSpec,
    AtomicTaskCreator,
)

from .hierarchical import (
    DecompositionNode,
    HierarchicalStrategy,
)

from .functional import (
    FunctionalComponent,
    FunctionalStrategy,
)

from .domain_driven import (
    DomainType,
    BoundedContext,
    DomainDrivenStrategy,
)

__all__ = [
    # Atomic Tasks
    "AtomicTaskSpec",
    "AtomicTaskCreator",
    # Hierarchical Decomposition
    "DecompositionNode",
    "HierarchicalStrategy",
    # Functional Decomposition
    "FunctionalComponent",
    "FunctionalStrategy",
    # Domain-Driven Decomposition
    "DomainType",
    "BoundedContext",
    "DomainDrivenStrategy",
]
