"""
Decomposition module for hierarchical task breakdown.

This module provides comprehensive task decomposition capabilities including:
- Complexity analysis to determine if tasks need decomposition
- Dependency analysis for execution ordering
- Risk assessment for identifying high-risk components
- Multiple decomposition strategies (hierarchical, functional, domain-driven)
- Atomic task creation
- DAG building for parallel execution

The decomposition module implements the 7-layer hierarchical decomposition
from the architecture, breaking complex tasks into atomic units (≤20 lines,
complexity ≤5).
"""

# Analyzers
from .analyzers import (
    ComplexityMetrics,
    ComplexityAnalyzer,
    DependencyType,
    TaskDependency,
    DependencyGraph,
    DependencyAnalyzer,
    RiskLevel,
    RiskCategory,
    RiskMetrics,
    RiskAnalyzer,
)

# Strategies
from .strategies import (
    AtomicTaskSpec,
    AtomicTaskCreator,
    DecompositionNode,
    HierarchicalStrategy,
    FunctionalComponent,
    FunctionalStrategy,
    DomainType,
    BoundedContext,
    DomainDrivenStrategy,
)

# DAG Builder
from .dag_builder import (
    NodeStatus,
    TaskNode,
    TaskDAG,
    DAGBuilder,
)

# Main Engine
from .engine import (
    DecompositionStrategy,
    DecompositionTree,
    DecompositionResult,
    DecompositionEngine,
)

__all__ = [
    # Analyzers
    "ComplexityMetrics",
    "ComplexityAnalyzer",
    "DependencyType",
    "TaskDependency",
    "DependencyGraph",
    "DependencyAnalyzer",
    "RiskLevel",
    "RiskCategory",
    "RiskMetrics",
    "RiskAnalyzer",
    # Strategies
    "AtomicTaskSpec",
    "AtomicTaskCreator",
    "DecompositionNode",
    "HierarchicalStrategy",
    "FunctionalComponent",
    "FunctionalStrategy",
    "DomainType",
    "BoundedContext",
    "DomainDrivenStrategy",
    # DAG Builder
    "NodeStatus",
    "TaskNode",
    "TaskDAG",
    "DAGBuilder",
    # Main Engine
    "DecompositionStrategy",
    "DecompositionTree",
    "DecompositionResult",
    "DecompositionEngine",
]

__version__ = "0.1.0"
