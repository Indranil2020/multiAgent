"""
Red Flag Detector - Main Orchestrator

Coordinates all red flag detection mechanisms to identify suspicious agent outputs.
This is the primary interface for the red-flagging system.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..task_spec.language import TaskSpecification
from ..task_spec.types import TaskType
from .patterns import PatternRegistry, FormatValidator, LengthValidator
from .uncertainty import UncertaintyDetector, ConfidenceAnalyzer


@dataclass
class AgentResponse:
    """Response from an agent execution"""
    task_id: str
    agent_id: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    
    def __str__(self) -> str:
        return f"AgentResponse(task={self.task_id}, agent={self.agent_id})"


@dataclass
class RedFlagResult:
    """Result of red flag detection"""
    is_flagged: bool
    reasons: List[str] = field(default_factory=list)
    severity: str = "none"  # none, low, medium, high, critical
    details: Dict[str, Any] = field(default_factory=dict)
    uncertainty_score: float = 0.0
    confidence_level: str = "high"
    
    def should_discard(self) -> bool:
        """Determine if response should be discarded"""
        return self.is_flagged and self.severity in ["high", "critical"]
    
    def should_escalate(self) -> bool:
        """Determine if response should be escalated for human review"""
        return self.is_flagged and self.severity == "medium"
    
    def __str__(self) -> str:
        if not self.is_flagged:
            return "RedFlag: PASSED"
        return f"RedFlag: FLAGGED ({self.severity}) - {', '.join(self.reasons)}"


class RedFlagDetector:
    """
    Main red flag detection system.
    
    Coordinates multiple detection mechanisms to identify suspicious or
    low-quality agent outputs before they enter the voting process.
    
    Based on MAKER paper's red-flagging approach.
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        enable_pattern_detection: bool = True,
        enable_format_validation: bool = True,
        enable_length_validation: bool = True,
        enable_uncertainty_detection: bool = True
    ):
        """
        Initialize the red flag detector.
        
        Args:
            uncertainty_threshold: Maximum acceptable uncertainty score
            enable_pattern_detection: Enable pattern-based detection
            enable_format_validation: Enable format validation
            enable_length_validation: Enable length validation
            enable_uncertainty_detection: Enable uncertainty detection
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_format_validation = enable_format_validation
        self.enable_length_validation = enable_length_validation
        self.enable_uncertainty_detection = enable_uncertainty_detection
        
        # Initialize detection components
        self.pattern_registry = PatternRegistry()
        self.uncertainty_detector = UncertaintyDetector()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.format_validator = FormatValidator()
        self.length_validator = LengthValidator()
    
    def check(
        self,
        response: AgentResponse,
        task_spec: TaskSpecification
    ) -> RedFlagResult:
        """
        Check if an agent response should be red-flagged.
        
        Args:
            response: The agent response to check
            task_spec: The task specification
            
        Returns:
            RedFlagResult with detailed analysis
        """
        reasons = []
        details = {}
        max_severity = "none"
        
        # Convert response content to string
        response_str = str(response.content)
        
        # 1. Check for empty or None response
        if not response.content or not response_str.strip():
            return RedFlagResult(
                is_flagged=True,
                reasons=["Empty or None response"],
                severity="critical",
                details={"response_length": 0}
            )
        
        # 2. Pattern-based detection
        if self.enable_pattern_detection:
            pattern_result = self._check_patterns(response_str)
            if pattern_result["flagged"]:
                reasons.extend(pattern_result["reasons"])
                details["pattern_matches"] = pattern_result["matches"]
                max_severity = self._max_severity(max_severity, pattern_result["severity"])
        
        # 3. Uncertainty detection
        if self.enable_uncertainty_detection:
            uncertainty_result = self._check_uncertainty(response_str)
            if uncertainty_result["flagged"]:
                reasons.append(uncertainty_result["reason"])
                details["uncertainty"] = uncertainty_result["details"]
                max_severity = self._max_severity(max_severity, uncertainty_result["severity"])
        
        # 4. Format validation
        if self.enable_format_validation:
            format_result = self._check_format(response_str)
            if format_result["flagged"]:
                reasons.extend(format_result["reasons"])
                details["format_errors"] = format_result["errors"]
                max_severity = self._max_severity(max_severity, format_result["severity"])
        
        # 5. Length validation
        if self.enable_length_validation:
            length_result = self._check_length(response_str, task_spec)
            if length_result["flagged"]:
                reasons.extend(length_result["reasons"])
                details["length_stats"] = length_result["stats"]
                max_severity = self._max_severity(max_severity, length_result["severity"])
        
        # 6. Task-specific checks
        task_result = self._check_task_specific(response_str, task_spec)
        if task_result["flagged"]:
            reasons.extend(task_result["reasons"])
            details["task_specific"] = task_result["details"]
            max_severity = self._max_severity(max_severity, task_result["severity"])
        
        # Build final result
        is_flagged = len(reasons) > 0
        
        # Get uncertainty score for metadata
        uncertainty_score_obj = self.uncertainty_detector.detect(response_str)
        
        return RedFlagResult(
            is_flagged=is_flagged,
            reasons=reasons,
            severity=max_severity,
            details=details,
            uncertainty_score=uncertainty_score_obj.score,
            confidence_level=uncertainty_score_obj.confidence_level
        )
    
    def _check_patterns(self, text: str) -> Dict[str, Any]:
        """Check for problematic patterns"""
        matches = self.pattern_registry.check_all_patterns(text)
        
        if not matches:
            return {"flagged": False}
        
        reasons = []
        max_severity = "low"
        
        for pattern_name, pattern_matches in matches.items():
            pattern = self.pattern_registry.get_pattern(pattern_name)
            reasons.append(f"Found {pattern.name}: {', '.join(pattern_matches[:3])}")
            max_severity = self._max_severity(max_severity, pattern.severity)
        
        return {
            "flagged": True,
            "reasons": reasons,
            "matches": matches,
            "severity": max_severity
        }
    
    def _check_uncertainty(self, text: str) -> Dict[str, Any]:
        """Check for uncertainty markers"""
        uncertainty_score = self.uncertainty_detector.detect(text)
        
        if uncertainty_score.is_acceptable(self.uncertainty_threshold):
            return {"flagged": False}
        
        return {
            "flagged": True,
            "reason": f"High uncertainty detected (score: {uncertainty_score.score:.2f})",
            "details": {
                "score": uncertainty_score.score,
                "markers": uncertainty_score.markers_found,
                "confidence_level": uncertainty_score.confidence_level
            },
            "severity": "high" if uncertainty_score.score > 0.7 else "medium"
        }
    
    def _check_format(self, text: str) -> Dict[str, Any]:
        """Check for format errors"""
        if not self.format_validator.has_format_errors(text):
            return {"flagged": False}
        
        errors = self.format_validator.get_format_error_details(text)
        
        return {
            "flagged": True,
            "reasons": [f"Format error: {e}" for e in errors],
            "errors": errors,
            "severity": "medium"
        }
    
    def _check_length(self, text: str, task_spec: TaskSpecification) -> Dict[str, Any]:
        """Check length constraints"""
        stats = self.length_validator.get_length_stats(text)
        reasons = []
        
        # Check maximum length based on task type
        if task_spec.task_type == TaskType.CODE_GENERATION:
            max_length = task_spec.max_lines * 100  # ~100 chars per line
            if self.length_validator.is_too_long(text, max_length):
                reasons.append(f"Response too long: {stats['total_chars']} chars (max: {max_length})")
            
            # Check line limit
            if self.length_validator.exceeds_line_limit(text, task_spec.max_lines):
                reasons.append(f"Too many lines: {stats['non_empty_lines']} (max: {task_spec.max_lines})")
        else:
            # General maximum
            if self.length_validator.is_too_long(text, 10000):
                reasons.append(f"Response too long: {stats['total_chars']} chars")
        
        # Check minimum length
        min_length = 10 if task_spec.task_type == TaskType.CODE_GENERATION else 5
        if self.length_validator.is_too_short(text, min_length):
            reasons.append(f"Response too short: {stats['total_chars']} chars (min: {min_length})")
        
        if not reasons:
            return {"flagged": False}
        
        return {
            "flagged": True,
            "reasons": reasons,
            "stats": stats,
            "severity": "medium"
        }
    
    def _check_task_specific(self, text: str, task_spec: TaskSpecification) -> Dict[str, Any]:
        """Perform task-specific validation"""
        reasons = []
        details = {}
        
        # Code generation specific checks
        if task_spec.task_type == TaskType.CODE_GENERATION:
            # Check for code markers
            has_code_markers = any(marker in text for marker in ["def ", "class ", "function ", "import ", "from "])
            if not has_code_markers:
                reasons.append("Code generation task but no code markers found")
        
        # Verification task specific checks
        elif task_spec.task_type == TaskType.VERIFICATION:
            # Should have verification-related content
            has_verification_markers = any(marker in text.lower() for marker in ["pass", "fail", "verify", "check", "test"])
            if not has_verification_markers:
                reasons.append("Verification task but no verification markers found")
        
        if not reasons:
            return {"flagged": False}
        
        return {
            "flagged": True,
            "reasons": reasons,
            "details": details,
            "severity": "medium"
        }
    
    def _max_severity(self, sev1: str, sev2: str) -> str:
        """Return the maximum severity level"""
        severity_order = ["none", "low", "medium", "high", "critical"]
        idx1 = severity_order.index(sev1) if sev1 in severity_order else 0
        idx2 = severity_order.index(sev2) if sev2 in severity_order else 0
        return severity_order[max(idx1, idx2)]
    
    def batch_check(
        self,
        responses: List[AgentResponse],
        task_spec: TaskSpecification
    ) -> List[RedFlagResult]:
        """Check multiple responses at once"""
        return [self.check(response, task_spec) for response in responses]
    
    def get_statistics(self, results: List[RedFlagResult]) -> Dict[str, Any]:
        """Get statistics from multiple red flag results"""
        total = len(results)
        flagged = sum(1 for r in results if r.is_flagged)
        
        severity_counts = {
            "none": sum(1 for r in results if r.severity == "none"),
            "low": sum(1 for r in results if r.severity == "low"),
            "medium": sum(1 for r in results if r.severity == "medium"),
            "high": sum(1 for r in results if r.severity == "high"),
            "critical": sum(1 for r in results if r.severity == "critical")
        }
        
        avg_uncertainty = sum(r.uncertainty_score for r in results) / max(total, 1)
        
        return {
            "total_checked": total,
            "flagged_count": flagged,
            "flagged_percentage": (flagged / max(total, 1)) * 100,
            "severity_distribution": severity_counts,
            "average_uncertainty": avg_uncertainty,
            "should_discard": sum(1 for r in results if r.should_discard()),
            "should_escalate": sum(1 for r in results if r.should_escalate())
        }
