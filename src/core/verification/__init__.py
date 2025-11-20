"""
Verification Stack Module

This module implements the 8-layer verification system for the zero-error architecture.
It provides comprehensive code verification through multiple layers of checks.

Layers:
    1. Syntax Verification - Parse and validate syntax
    2. Type Checking - Verify type correctness
    3. Contract Verification - Check pre/postconditions
    4. Unit Testing - Execute unit tests
    5. Property Testing - Run property-based tests
    6. Static Analysis - Analyze code quality
    7. Security Scanning - Detect vulnerabilities
    8. Performance Checking - Validate performance requirements

Additional Components:
    - Formal Prover - Mathematical correctness proofs
    - Compositional Verifier - Verify component composition
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class VerificationLayer(Enum):
    """Enumeration of verification layers"""
    SYNTAX = "syntax"
    TYPES = "types"
    CONTRACTS = "contracts"
    UNIT_TESTS = "unit_tests"
    PROPERTY_TESTS = "property_tests"
    STATIC_ANALYSIS = "static_analysis"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FORMAL_PROOF = "formal_proof"
    COMPOSITIONAL = "compositional"


class VerificationStatus(Enum):
    """Status of verification result"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class VerificationIssue:
    """Represents a single verification issue"""
    layer: VerificationLayer
    severity: str  # "critical", "high", "medium", "low", "info"
    message: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of a verification layer"""
    layer: VerificationLayer
    status: VerificationStatus
    passed: bool
    message: str = ""
    issues: List[VerificationIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: VerificationIssue) -> None:
        """Add an issue to this result"""
        self.issues.append(issue)
        if issue.severity in ["critical", "high"]:
            self.passed = False
            self.status = VerificationStatus.FAILED


@dataclass
class VerificationReport:
    """Complete verification report for code"""
    code_id: str
    overall_passed: bool
    results: List[VerificationResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    total_execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: VerificationResult) -> None:
        """Add a verification result to the report"""
        self.results.append(result)
        self.total_execution_time_ms += result.execution_time_ms
        
        # Update issue counts
        for issue in result.issues:
            self.total_issues += 1
            if issue.severity == "critical":
                self.critical_issues += 1
            elif issue.severity == "high":
                self.high_issues += 1
            elif issue.severity == "medium":
                self.medium_issues += 1
            elif issue.severity == "low":
                self.low_issues += 1
        
        # Update overall status
        if not result.passed:
            self.overall_passed = False

    def get_summary(self) -> str:
        """Get a summary of the verification report"""
        status = "PASSED" if self.overall_passed else "FAILED"
        return (
            f"Verification {status}\n"
            f"Total Issues: {self.total_issues} "
            f"(Critical: {self.critical_issues}, "
            f"High: {self.high_issues}, "
            f"Medium: {self.medium_issues}, "
            f"Low: {self.low_issues})\n"
            f"Execution Time: {self.total_execution_time_ms:.2f}ms"
        )


# Export all public classes and enums
__all__ = [
    "VerificationLayer",
    "VerificationStatus",
    "VerificationIssue",
    "VerificationResult",
    "VerificationReport",
]
