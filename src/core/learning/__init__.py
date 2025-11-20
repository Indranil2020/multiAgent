"""
Continuous learning module for the zero-error system.

This module provides continuous learning capabilities including:
- Error pattern recognition and analysis
- Agent performance improvement and specialization
- Specification refinement and optimization

The learning module enables the system to improve over time by learning from
execution history and adapting agent configurations and specifications.
"""

from .pattern_recognizer import (
    PatternRecognizer,
    ErrorPattern,
    PatternStatistics,
    FailureRecord,
    PatternType,
    PatternSeverity,
)

from .agent_improver import (
    AgentImprover,
    AgentPerformanceMetrics,
    SpecializationProfile,
    ExecutionRecord,
    PerformanceMetric,
)

from .spec_refiner import (
    SpecRefiner,
    SpecificationIssue,
    RefinementSuggestion,
    SpecificationFailureRecord,
    IssueType,
    IssueSeverity,
    RefinementType,
)

__all__ = [
    # Pattern Recognition
    "PatternRecognizer",
    "ErrorPattern",
    "PatternStatistics",
    "FailureRecord",
    "PatternType",
    "PatternSeverity",
    # Agent Improvement
    "AgentImprover",
    "AgentPerformanceMetrics",
    "SpecializationProfile",
    "ExecutionRecord",
    "PerformanceMetric",
    # Specification Refinement
    "SpecRefiner",
    "SpecificationIssue",
    "RefinementSuggestion",
    "SpecificationFailureRecord",
    "IssueType",
    "IssueSeverity",
    "RefinementType",
]

__version__ = "0.1.0"
