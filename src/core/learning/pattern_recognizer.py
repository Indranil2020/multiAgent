"""
Error pattern recognition and learning system.

This module implements pattern recognition for agent execution failures,
enabling the system to learn from recurring error patterns and improve
future agent selection and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict


class PatternType(Enum):
    """Types of error patterns that can be detected."""
    AGENT_CONFIG_FAILURE = "agent_config_failure"
    TASK_TYPE_MISMATCH = "task_type_mismatch"
    SEMANTIC_DIVERGENCE = "semantic_divergence"
    TIMEOUT_PATTERN = "timeout_pattern"
    QUALITY_DEGRADATION = "quality_degradation"
    RED_FLAG_CLUSTER = "red_flag_cluster"


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class ErrorPattern:
    """
    Represents a detected error pattern.
    
    Attributes:
        pattern_id: Unique identifier for this pattern
        pattern_type: Type of pattern detected
        description: Human-readable description of the pattern
        severity: Severity level of this pattern
        characteristics: Key characteristics that define this pattern
        occurrence_count: Number of times this pattern has been observed
        confidence: Confidence score for this pattern (0.0 to 1.0)
        first_seen_timestamp: When this pattern was first detected
        last_seen_timestamp: When this pattern was last observed
    """
    pattern_id: str
    pattern_type: PatternType
    description: str
    severity: PatternSeverity
    characteristics: Dict[str, str]
    occurrence_count: int
    confidence: float
    first_seen_timestamp: float
    last_seen_timestamp: float
    
    def validate_confidence(self) -> bool:
        """Validate confidence score is in valid range."""
        return 0.0 <= self.confidence <= 1.0
    
    def validate_occurrence_count(self) -> bool:
        """Validate occurrence count is positive."""
        return self.occurrence_count > 0
    
    def validate_timestamps(self) -> bool:
        """Validate timestamp ordering."""
        return self.first_seen_timestamp <= self.last_seen_timestamp
    
    def is_valid(self) -> bool:
        """Check if pattern is fully valid."""
        return (
            self.validate_confidence() and
            self.validate_occurrence_count() and
            self.validate_timestamps()
        )


@dataclass
class PatternStatistics:
    """
    Statistics about pattern occurrences.
    
    Attributes:
        total_patterns: Total number of unique patterns detected
        total_occurrences: Total occurrences across all patterns
        patterns_by_type: Count of patterns by type
        patterns_by_severity: Count of patterns by severity
        most_common_patterns: List of most frequently occurring patterns
        recent_patterns: Recently detected patterns
    """
    total_patterns: int = 0
    total_occurrences: int = 0
    patterns_by_type: Dict[PatternType, int] = field(default_factory=dict)
    patterns_by_severity: Dict[PatternSeverity, int] = field(default_factory=dict)
    most_common_patterns: List[ErrorPattern] = field(default_factory=list)
    recent_patterns: List[ErrorPattern] = field(default_factory=list)
    
    def validate_counts(self) -> bool:
        """Validate all counts are non-negative."""
        if self.total_patterns < 0 or self.total_occurrences < 0:
            return False
        
        for count in self.patterns_by_type.values():
            if count < 0:
                return False
        
        for count in self.patterns_by_severity.values():
            if count < 0:
                return False
        
        return True
    
    def is_valid(self) -> bool:
        """Check if statistics are valid."""
        return self.validate_counts()


@dataclass
class FailureRecord:
    """
    Record of a single failure event.
    
    Attributes:
        task_id: ID of the task that failed
        task_type: Type of task
        agent_config: Configuration of the agent that failed
        failure_reason: Reason for failure
        timestamp: When the failure occurred
        metadata: Additional metadata about the failure
    """
    task_id: str
    task_type: str
    agent_config: Dict[str, str]
    failure_reason: str
    timestamp: float
    metadata: Dict[str, str] = field(default_factory=dict)


class PatternRecognizer:
    """
    Recognizes and learns from error patterns in agent executions.
    
    This class analyzes failure records to identify recurring patterns,
    enabling the system to learn which agent configurations tend to fail
    for which types of tasks.
    """
    
    def __init__(self, min_occurrences: int = 3, confidence_threshold: float = 0.7):
        """
        Initialize pattern recognizer.
        
        Args:
            min_occurrences: Minimum occurrences before pattern is considered significant
            confidence_threshold: Minimum confidence for pattern detection
        """
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold
        self.patterns: Dict[str, ErrorPattern] = {}
        self.failure_records: List[FailureRecord] = []
        self.pattern_index: Dict[PatternType, Set[str]] = defaultdict(set)
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.min_occurrences < 1:
            return (False, "min_occurrences must be at least 1")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return (False, "confidence_threshold must be between 0.0 and 1.0")
        
        return (True, "")
    
    def record_failure(self, failure: FailureRecord) -> Tuple[bool, str]:
        """
        Record a failure event for pattern analysis.
        
        Args:
            failure: The failure record to add
        
        Returns:
            Tuple of (success, message)
        """
        if not failure.task_id:
            return (False, "task_id cannot be empty")
        
        if not failure.task_type:
            return (False, "task_type cannot be empty")
        
        if not failure.failure_reason:
            return (False, "failure_reason cannot be empty")
        
        if failure.timestamp <= 0:
            return (False, "timestamp must be positive")
        
        self.failure_records.append(failure)
        return (True, "Failure recorded successfully")
    
    def analyze_patterns(self) -> Tuple[bool, str]:
        """
        Analyze failure records to detect patterns.
        
        Returns:
            Tuple of (success, message)
        """
        if len(self.failure_records) < self.min_occurrences:
            return (False, f"Need at least {self.min_occurrences} failures to analyze")
        
        # Detect agent configuration failure patterns
        self._detect_agent_config_patterns()
        
        # Detect task type mismatch patterns
        self._detect_task_type_patterns()
        
        # Detect timeout patterns
        self._detect_timeout_patterns()
        
        # Detect quality degradation patterns
        self._detect_quality_patterns()
        
        patterns_found = len(self.patterns)
        return (True, f"Analysis complete. Found {patterns_found} patterns.")
    
    def _detect_agent_config_patterns(self) -> None:
        """Detect patterns related to agent configuration failures."""
        # Group failures by agent configuration
        config_failures: Dict[str, List[FailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            # Create a signature from agent config
            config_sig = self._create_config_signature(record.agent_config)
            config_failures[config_sig].append(record)
        
        # Identify patterns where same config fails repeatedly
        for config_sig, failures in config_failures.items():
            if len(failures) >= self.min_occurrences:
                confidence = min(len(failures) / (self.min_occurrences * 2), 1.0)
                
                if confidence >= self.confidence_threshold:
                    pattern_id = f"agent_config_{config_sig}"
                    
                    if pattern_id not in self.patterns:
                        pattern = ErrorPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.AGENT_CONFIG_FAILURE,
                            description=f"Agent configuration {config_sig} frequently fails",
                            severity=self._calculate_severity(len(failures)),
                            characteristics=failures[0].agent_config,
                            occurrence_count=len(failures),
                            confidence=confidence,
                            first_seen_timestamp=failures[0].timestamp,
                            last_seen_timestamp=failures[-1].timestamp
                        )
                        
                        self.patterns[pattern_id] = pattern
                        self.pattern_index[PatternType.AGENT_CONFIG_FAILURE].add(pattern_id)
    
    def _detect_task_type_patterns(self) -> None:
        """Detect patterns related to task type mismatches."""
        # Group failures by task type and agent config
        type_config_failures: Dict[Tuple[str, str], List[FailureRecord]] = defaultdict(list)
        
        for record in self.failure_records:
            config_sig = self._create_config_signature(record.agent_config)
            key = (record.task_type, config_sig)
            type_config_failures[key].append(record)
        
        # Identify patterns where specific task types fail with specific configs
        for (task_type, config_sig), failures in type_config_failures.items():
            if len(failures) >= self.min_occurrences:
                confidence = min(len(failures) / (self.min_occurrences * 2), 1.0)
                
                if confidence >= self.confidence_threshold:
                    pattern_id = f"task_type_{task_type}_{config_sig}"
                    
                    if pattern_id not in self.patterns:
                        pattern = ErrorPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.TASK_TYPE_MISMATCH,
                            description=f"Task type {task_type} fails with config {config_sig}",
                            severity=self._calculate_severity(len(failures)),
                            characteristics={
                                "task_type": task_type,
                                **failures[0].agent_config
                            },
                            occurrence_count=len(failures),
                            confidence=confidence,
                            first_seen_timestamp=failures[0].timestamp,
                            last_seen_timestamp=failures[-1].timestamp
                        )
                        
                        self.patterns[pattern_id] = pattern
                        self.pattern_index[PatternType.TASK_TYPE_MISMATCH].add(pattern_id)
    
    def _detect_timeout_patterns(self) -> None:
        """Detect patterns related to timeouts."""
        timeout_failures = [
            r for r in self.failure_records
            if "timeout" in r.failure_reason.lower()
        ]
        
        if len(timeout_failures) >= self.min_occurrences:
            # Group by task type
            by_task_type: Dict[str, List[FailureRecord]] = defaultdict(list)
            for record in timeout_failures:
                by_task_type[record.task_type].append(record)
            
            for task_type, failures in by_task_type.items():
                if len(failures) >= self.min_occurrences:
                    confidence = min(len(failures) / (self.min_occurrences * 2), 1.0)
                    
                    if confidence >= self.confidence_threshold:
                        pattern_id = f"timeout_{task_type}"
                        
                        if pattern_id not in self.patterns:
                            pattern = ErrorPattern(
                                pattern_id=pattern_id,
                                pattern_type=PatternType.TIMEOUT_PATTERN,
                                description=f"Task type {task_type} frequently times out",
                                severity=self._calculate_severity(len(failures)),
                                characteristics={"task_type": task_type},
                                occurrence_count=len(failures),
                                confidence=confidence,
                                first_seen_timestamp=failures[0].timestamp,
                                last_seen_timestamp=failures[-1].timestamp
                            )
                            
                            self.patterns[pattern_id] = pattern
                            self.pattern_index[PatternType.TIMEOUT_PATTERN].add(pattern_id)
    
    def _detect_quality_patterns(self) -> None:
        """Detect patterns related to quality degradation."""
        quality_failures = [
            r for r in self.failure_records
            if "quality" in r.failure_reason.lower() or "score" in r.failure_reason.lower()
        ]
        
        if len(quality_failures) >= self.min_occurrences:
            confidence = min(len(quality_failures) / (self.min_occurrences * 2), 1.0)
            
            if confidence >= self.confidence_threshold:
                pattern_id = "quality_degradation_general"
                
                if pattern_id not in self.patterns:
                    pattern = ErrorPattern(
                        pattern_id=pattern_id,
                        pattern_type=PatternType.QUALITY_DEGRADATION,
                        description="Frequent quality degradation detected",
                        severity=self._calculate_severity(len(quality_failures)),
                        characteristics={"type": "general_quality_issue"},
                        occurrence_count=len(quality_failures),
                        confidence=confidence,
                        first_seen_timestamp=quality_failures[0].timestamp,
                        last_seen_timestamp=quality_failures[-1].timestamp
                    )
                    
                    self.patterns[pattern_id] = pattern
                    self.pattern_index[PatternType.QUALITY_DEGRADATION].add(pattern_id)
    
    def _create_config_signature(self, config: Dict[str, str]) -> str:
        """
        Create a unique signature from agent configuration.
        
        Args:
            config: Agent configuration dictionary
        
        Returns:
            String signature representing the configuration
        """
        # Sort keys for consistent signature
        sorted_items = sorted(config.items())
        signature_parts = [f"{k}={v}" for k, v in sorted_items]
        return "_".join(signature_parts)
    
    def _calculate_severity(self, occurrence_count: int) -> PatternSeverity:
        """
        Calculate severity based on occurrence count.
        
        Args:
            occurrence_count: Number of times pattern occurred
        
        Returns:
            Severity level
        """
        if occurrence_count >= self.min_occurrences * 4:
            return PatternSeverity.CRITICAL
        if occurrence_count >= self.min_occurrences * 2:
            return PatternSeverity.HIGH
        if occurrence_count >= self.min_occurrences * 1.5:
            return PatternSeverity.MEDIUM
        return PatternSeverity.LOW
    
    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """
        Get a specific pattern by ID.
        
        Args:
            pattern_id: ID of the pattern to retrieve
        
        Returns:
            ErrorPattern if found, None otherwise
        """
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[ErrorPattern]:
        """
        Get all patterns of a specific type.
        
        Args:
            pattern_type: Type of patterns to retrieve
        
        Returns:
            List of patterns of the specified type
        """
        pattern_ids = self.pattern_index.get(pattern_type, set())
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def get_statistics(self) -> PatternStatistics:
        """
        Get statistics about detected patterns.
        
        Returns:
            PatternStatistics object with current statistics
        """
        patterns_by_type: Dict[PatternType, int] = defaultdict(int)
        patterns_by_severity: Dict[PatternSeverity, int] = defaultdict(int)
        
        total_occurrences = 0
        
        for pattern in self.patterns.values():
            patterns_by_type[pattern.pattern_type] += 1
            patterns_by_severity[pattern.severity] += 1
            total_occurrences += pattern.occurrence_count
        
        # Get most common patterns (top 10 by occurrence count)
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.occurrence_count,
            reverse=True
        )
        most_common = sorted_patterns[:10]
        
        # Get recent patterns (top 10 by last seen timestamp)
        recent_sorted = sorted(
            self.patterns.values(),
            key=lambda p: p.last_seen_timestamp,
            reverse=True
        )
        recent = recent_sorted[:10]
        
        return PatternStatistics(
            total_patterns=len(self.patterns),
            total_occurrences=total_occurrences,
            patterns_by_type=dict(patterns_by_type),
            patterns_by_severity=dict(patterns_by_severity),
            most_common_patterns=most_common,
            recent_patterns=recent
        )
    
    def clear_patterns(self) -> None:
        """Clear all stored patterns and failure records."""
        self.patterns.clear()
        self.failure_records.clear()
        self.pattern_index.clear()
    
    def should_avoid_config(
        self,
        agent_config: Dict[str, str],
        task_type: str
    ) -> Tuple[bool, str]:
        """
        Check if an agent configuration should be avoided for a task type.
        
        Args:
            agent_config: Agent configuration to check
            task_type: Type of task
        
        Returns:
            Tuple of (should_avoid, reason)
        """
        config_sig = self._create_config_signature(agent_config)
        
        # Check for agent config failures
        agent_pattern_id = f"agent_config_{config_sig}"
        if agent_pattern_id in self.patterns:
            pattern = self.patterns[agent_pattern_id]
            if pattern.severity in {PatternSeverity.HIGH, PatternSeverity.CRITICAL}:
                return (True, f"Configuration has {pattern.occurrence_count} failures")
        
        # Check for task type mismatches
        task_pattern_id = f"task_type_{task_type}_{config_sig}"
        if task_pattern_id in self.patterns:
            pattern = self.patterns[task_pattern_id]
            if pattern.severity in {PatternSeverity.HIGH, PatternSeverity.CRITICAL}:
                return (True, f"Configuration fails for {task_type} tasks")
        
        return (False, "")
