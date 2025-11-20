"""
Specification refinement and improvement system.

This module analyzes task specifications that lead to failures and suggests
improvements to make specifications more precise, complete, and effective.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class IssueType(Enum):
    """Types of specification issues that can be detected."""
    AMBIGUOUS_DESCRIPTION = "ambiguous_description"
    MISSING_PRECONDITION = "missing_precondition"
    MISSING_POSTCONDITION = "missing_postcondition"
    INCOMPLETE_TYPE_INFO = "incomplete_type_info"
    INSUFFICIENT_TEST_CASES = "insufficient_test_cases"
    UNCLEAR_CONSTRAINTS = "unclear_constraints"
    COMPLEXITY_MISMATCH = "complexity_mismatch"
    TIMEOUT_TOO_STRICT = "timeout_too_strict"


class IssueSeverity(Enum):
    """Severity levels for specification issues."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RefinementType(Enum):
    """Types of refinements that can be suggested."""
    ADD_PRECONDITION = "add_precondition"
    ADD_POSTCONDITION = "add_postcondition"
    CLARIFY_DESCRIPTION = "clarify_description"
    ADD_TYPE_ANNOTATION = "add_type_annotation"
    ADD_TEST_CASE = "add_test_case"
    RELAX_CONSTRAINT = "relax_constraint"
    TIGHTEN_CONSTRAINT = "tighten_constraint"
    INCREASE_TIMEOUT = "increase_timeout"
    DECREASE_COMPLEXITY_LIMIT = "decrease_complexity_limit"


@dataclass(frozen=True)
class SpecificationIssue:
    """
    Represents an identified issue in a task specification.
    
    Attributes:
        issue_id: Unique identifier for this issue
        task_id: ID of the task with the issue
        issue_type: Type of issue detected
        severity: Severity level of the issue
        description: Human-readable description of the issue
        affected_component: Which part of the spec is affected
        evidence: Evidence supporting this issue
        occurrence_count: Number of times this issue has been observed
        confidence: Confidence in this issue detection (0.0 to 1.0)
        first_detected_timestamp: When issue was first detected
        last_detected_timestamp: When issue was last observed
    """
    issue_id: str
    task_id: str
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    affected_component: str
    evidence: Dict[str, str]
    occurrence_count: int
    confidence: float
    first_detected_timestamp: float
    last_detected_timestamp: float
    
    def validate_confidence(self) -> bool:
        """Validate confidence score is in valid range."""
        return 0.0 <= self.confidence <= 1.0
    
    def validate_occurrence_count(self) -> bool:
        """Validate occurrence count is positive."""
        return self.occurrence_count > 0
    
    def validate_timestamps(self) -> bool:
        """Validate timestamp ordering."""
        return self.first_detected_timestamp <= self.last_detected_timestamp
    
    def is_valid(self) -> bool:
        """Check if issue is fully valid."""
        return (
            self.validate_confidence() and
            self.validate_occurrence_count() and
            self.validate_timestamps()
        )


@dataclass(frozen=True)
class RefinementSuggestion:
    """
    Represents a suggested improvement to a task specification.
    
    Attributes:
        suggestion_id: Unique identifier for this suggestion
        task_id: ID of the task to refine
        refinement_type: Type of refinement suggested
        description: Human-readable description of the suggestion
        specific_change: Specific change to make
        rationale: Why this refinement is suggested
        expected_improvement: Expected improvement from this change
        confidence: Confidence in this suggestion (0.0 to 1.0)
        priority: Priority level for this suggestion
        based_on_issues: Issues this suggestion addresses
        created_timestamp: When suggestion was created
    """
    suggestion_id: str
    task_id: str
    refinement_type: RefinementType
    description: str
    specific_change: str
    rationale: str
    expected_improvement: str
    confidence: float
    priority: IssueSeverity
    based_on_issues: List[str]
    created_timestamp: float
    
    def validate_confidence(self) -> bool:
        """Validate confidence score is in valid range."""
        return 0.0 <= self.confidence <= 1.0
    
    def is_valid(self) -> bool:
        """Check if suggestion is valid."""
        return self.validate_confidence()


@dataclass
class SpecificationFailureRecord:
    """
    Record of a specification-related failure.
    
    Attributes:
        task_id: ID of the task that failed
        task_type: Type of task
        failure_mode: How the task failed
        specification_snapshot: Snapshot of the specification
        failure_details: Details about the failure
        timestamp: When the failure occurred
    """
    task_id: str
    task_type: str
    failure_mode: str
    specification_snapshot: Dict[str, str]
    failure_details: Dict[str, str]
    timestamp: float


class SpecRefiner:
    """
    Refines and improves task specifications based on failure analysis.
    
    This class analyzes specifications that lead to failures and suggests
    improvements to make them more precise, complete, and effective.
    """
    
    def __init__(
        self,
        min_failures_for_issue: int = 3,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize specification refiner.
        
        Args:
            min_failures_for_issue: Minimum failures before identifying issue
            confidence_threshold: Minimum confidence for suggestions
        """
        self.min_failures_for_issue = min_failures_for_issue
        self.confidence_threshold = confidence_threshold
        self.issues: Dict[str, SpecificationIssue] = {}
        self.suggestions: Dict[str, RefinementSuggestion] = {}
        self.failure_records: List[SpecificationFailureRecord] = []
        self.issue_index: Dict[IssueType, Set[str]] = defaultdict(set)
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.min_failures_for_issue < 1:
            return (False, "min_failures_for_issue must be at least 1")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return (False, "confidence_threshold must be between 0.0 and 1.0")
        
        return (True, "")
    
    def record_failure(
        self,
        failure: SpecificationFailureRecord
    ) -> Tuple[bool, str]:
        """
        Record a specification-related failure.
        
        Args:
            failure: The failure record to add
        
        Returns:
            Tuple of (success, message)
        """
        if not failure.task_id:
            return (False, "task_id cannot be empty")
        
        if not failure.task_type:
            return (False, "task_type cannot be empty")
        
        if not failure.failure_mode:
            return (False, "failure_mode cannot be empty")
        
        if failure.timestamp <= 0:
            return (False, "timestamp must be positive")
        
        self.failure_records.append(failure)
        return (True, "Failure recorded successfully")
    
    def analyze_specifications(self) -> Tuple[bool, str]:
        """
        Analyze specifications to identify issues.
        
        Returns:
            Tuple of (success, message)
        """
        if len(self.failure_records) < self.min_failures_for_issue:
            return (
                False,
                f"Need at least {self.min_failures_for_issue} failures to analyze"
            )
        
        # Detect various types of issues
        self._detect_ambiguous_descriptions()
        self._detect_missing_preconditions()
        self._detect_missing_postconditions()
        self._detect_incomplete_type_info()
        self._detect_insufficient_test_cases()
        self._detect_timeout_issues()
        
        issues_found = len(self.issues)
        return (True, f"Analysis complete. Found {issues_found} issues.")
    
    def generate_suggestions(self) -> Tuple[bool, str]:
        """
        Generate refinement suggestions based on identified issues.
        
        Returns:
            Tuple of (success, message)
        """
        if len(self.issues) == 0:
            return (False, "No issues identified. Run analyze_specifications first.")
        
        suggestions_created = 0
        
        for issue in self.issues.values():
            # Generate suggestions based on issue type
            suggestion = self._create_suggestion_for_issue(issue)
            
            if suggestion and suggestion.confidence >= self.confidence_threshold:
                self.suggestions[suggestion.suggestion_id] = suggestion
                suggestions_created += 1
        
        return (True, f"Generated {suggestions_created} refinement suggestions")
    
    def _detect_ambiguous_descriptions(self) -> None:
        """Detect tasks with ambiguous descriptions."""
        # Group failures by task
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "ambiguous" in record.failure_mode.lower() or \
               "unclear" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        # Identify tasks with repeated ambiguity failures
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"ambiguous_desc_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.AMBIGUOUS_DESCRIPTION,
                        severity=IssueSeverity.HIGH,
                        description=f"Task {task_id} has ambiguous description",
                        affected_component="description",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.AMBIGUOUS_DESCRIPTION].add(issue_id)
    
    def _detect_missing_preconditions(self) -> None:
        """Detect tasks missing important preconditions."""
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "precondition" in record.failure_mode.lower() or \
               "invalid input" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"missing_precond_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.MISSING_PRECONDITION,
                        severity=IssueSeverity.HIGH,
                        description=f"Task {task_id} missing preconditions",
                        affected_component="preconditions",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.MISSING_PRECONDITION].add(issue_id)
    
    def _detect_missing_postconditions(self) -> None:
        """Detect tasks missing important postconditions."""
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "postcondition" in record.failure_mode.lower() or \
               "invalid output" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"missing_postcond_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.MISSING_POSTCONDITION,
                        severity=IssueSeverity.HIGH,
                        description=f"Task {task_id} missing postconditions",
                        affected_component="postconditions",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.MISSING_POSTCONDITION].add(issue_id)
    
    def _detect_incomplete_type_info(self) -> None:
        """Detect tasks with incomplete type information."""
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "type" in record.failure_mode.lower() or \
               "annotation" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"incomplete_types_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.INCOMPLETE_TYPE_INFO,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Task {task_id} has incomplete type information",
                        affected_component="type_annotations",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.INCOMPLETE_TYPE_INFO].add(issue_id)
    
    def _detect_insufficient_test_cases(self) -> None:
        """Detect tasks with insufficient test cases."""
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "test" in record.failure_mode.lower() or \
               "coverage" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"insufficient_tests_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.INSUFFICIENT_TEST_CASES,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Task {task_id} has insufficient test cases",
                        affected_component="test_cases",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.INSUFFICIENT_TEST_CASES].add(issue_id)
    
    def _detect_timeout_issues(self) -> None:
        """Detect tasks with timeout issues."""
        by_task: Dict[str, List[SpecificationFailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            if "timeout" in record.failure_mode.lower():
                by_task[record.task_id].append(record)
        
        for task_id, failures in by_task.items():
            if len(failures) >= self.min_failures_for_issue:
                confidence = min(
                    len(failures) / (self.min_failures_for_issue * 2),
                    1.0
                )
                
                issue_id = f"timeout_{task_id}"
                
                if issue_id not in self.issues:
                    issue = SpecificationIssue(
                        issue_id=issue_id,
                        task_id=task_id,
                        issue_type=IssueType.TIMEOUT_TOO_STRICT,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Task {task_id} has timeout issues",
                        affected_component="timeout",
                        evidence={
                            "failure_count": str(len(failures)),
                            "failure_mode": failures[0].failure_mode
                        },
                        occurrence_count=len(failures),
                        confidence=confidence,
                        first_detected_timestamp=failures[0].timestamp,
                        last_detected_timestamp=failures[-1].timestamp
                    )
                    
                    self.issues[issue_id] = issue
                    self.issue_index[IssueType.TIMEOUT_TOO_STRICT].add(issue_id)
    
    def _create_suggestion_for_issue(
        self,
        issue: SpecificationIssue
    ) -> Optional[RefinementSuggestion]:
        """
        Create a refinement suggestion for an issue.
        
        Args:
            issue: The issue to create suggestion for
        
        Returns:
            RefinementSuggestion or None
        """
        suggestion_id = f"suggestion_{issue.issue_id}"
        
        # Map issue types to refinement types and create suggestions
        if issue.issue_type == IssueType.AMBIGUOUS_DESCRIPTION:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.CLARIFY_DESCRIPTION,
                description="Clarify task description to reduce ambiguity",
                specific_change="Add more specific details about expected behavior",
                rationale=f"Task failed {issue.occurrence_count} times due to ambiguity",
                expected_improvement="Reduced agent confusion and better results",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        elif issue.issue_type == IssueType.MISSING_PRECONDITION:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.ADD_PRECONDITION,
                description="Add missing preconditions to specification",
                specific_change="Define input validation requirements",
                rationale=f"Task failed {issue.occurrence_count} times due to invalid inputs",
                expected_improvement="Better input validation and fewer failures",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        elif issue.issue_type == IssueType.MISSING_POSTCONDITION:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.ADD_POSTCONDITION,
                description="Add missing postconditions to specification",
                specific_change="Define output validation requirements",
                rationale=f"Task failed {issue.occurrence_count} times due to invalid outputs",
                expected_improvement="Better output validation and quality",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        elif issue.issue_type == IssueType.INCOMPLETE_TYPE_INFO:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.ADD_TYPE_ANNOTATION,
                description="Add complete type annotations",
                specific_change="Specify types for all parameters and return values",
                rationale=f"Task failed {issue.occurrence_count} times due to type issues",
                expected_improvement="Better type safety and fewer type errors",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        elif issue.issue_type == IssueType.INSUFFICIENT_TEST_CASES:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.ADD_TEST_CASE,
                description="Add more comprehensive test cases",
                specific_change="Include edge cases and boundary conditions",
                rationale=f"Task failed {issue.occurrence_count} times, more tests needed",
                expected_improvement="Better coverage and quality assurance",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        elif issue.issue_type == IssueType.TIMEOUT_TOO_STRICT:
            return RefinementSuggestion(
                suggestion_id=suggestion_id,
                task_id=issue.task_id,
                refinement_type=RefinementType.INCREASE_TIMEOUT,
                description="Increase timeout limit",
                specific_change="Double the current timeout value",
                rationale=f"Task timed out {issue.occurrence_count} times",
                expected_improvement="Allow sufficient time for completion",
                confidence=issue.confidence,
                priority=issue.severity,
                based_on_issues=[issue.issue_id],
                created_timestamp=issue.last_detected_timestamp
            )
        
        return None
    
    def get_issue(self, issue_id: str) -> Optional[SpecificationIssue]:
        """Get a specific issue by ID."""
        return self.issues.get(issue_id)
    
    def get_suggestion(self, suggestion_id: str) -> Optional[RefinementSuggestion]:
        """Get a specific suggestion by ID."""
        return self.suggestions.get(suggestion_id)
    
    def get_issues_for_task(self, task_id: str) -> List[SpecificationIssue]:
        """Get all issues for a specific task."""
        return [
            issue for issue in self.issues.values()
            if issue.task_id == task_id
        ]
    
    def get_suggestions_for_task(self, task_id: str) -> List[RefinementSuggestion]:
        """Get all suggestions for a specific task."""
        return [
            suggestion for suggestion in self.suggestions.values()
            if suggestion.task_id == task_id
        ]
    
    def get_high_priority_suggestions(self) -> List[RefinementSuggestion]:
        """Get high priority suggestions."""
        return [
            suggestion for suggestion in self.suggestions.values()
            if suggestion.priority in {IssueSeverity.HIGH, IssueSeverity.CRITICAL}
        ]
    
    def clear_data(self) -> None:
        """Clear all stored issues, suggestions, and failure records."""
        self.issues.clear()
        self.suggestions.clear()
        self.failure_records.clear()
        self.issue_index.clear()
