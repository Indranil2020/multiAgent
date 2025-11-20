"""
Red Flag Detection Module

Provides comprehensive red flag detection for agent responses in the zero-error system.
Based on the MAKER paper's approach to identifying and filtering suspicious outputs.

This module includes:
- Pattern-based detection for problematic markers
- Uncertainty quantification and detection
- Format validation
- Length validation
- Escalation management for flagged responses

Usage:
    from src.core.red_flag import RedFlagDetector, EscalationManager
    
    detector = RedFlagDetector()
    result = detector.check(agent_response, task_spec)
    
    if result.is_flagged:
        escalation_manager = EscalationManager()
        decision = escalation_manager.handle_red_flag(task_id, result)
"""

from .detector import RedFlagDetector, RedFlagResult
from .patterns import (
    DetectionPattern,
    PatternRegistry,
    FormatValidator,
    LengthValidator
)
from .uncertainty import (
    UncertaintyDetector,
    UncertaintyScore,
    ConfidenceAnalyzer
)
from .escalation import (
    EscalationLevel,
    EscalationReason,
    EscalationDecision,
    EscalationPolicy,
    EscalationManager,
    EscalationRouter
)

__all__ = [
    # Main detector
    "RedFlagDetector",
    "RedFlagResult",
    
    # Pattern detection
    "DetectionPattern",
    "PatternRegistry",
    "FormatValidator",
    "LengthValidator",
    
    # Uncertainty detection
    "UncertaintyDetector",
    "UncertaintyScore",
    "ConfidenceAnalyzer",
    
    # Escalation
    "EscalationLevel",
    "EscalationReason",
    "EscalationDecision",
    "EscalationPolicy",
    "EscalationManager",
    "EscalationRouter",
]

__version__ = "1.0.0"
__author__ = "Zero-Error System Team"
