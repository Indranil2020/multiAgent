"""
Red Flag Detection Patterns

Defines the patterns used to detect suspicious or problematic agent outputs.
These patterns are based on the MAKER paper's red-flagging approach.
"""

from dataclasses import dataclass, field
from typing import List, Set, Pattern, Dict
import re


@dataclass
class DetectionPattern:
    """A pattern for detecting red flags in agent responses"""
    name: str
    patterns: List[str]
    description: str
    severity: str = "high"  # high, medium, low
    case_sensitive: bool = False
    
    def matches(self, text: str) -> bool:
        """Check if any pattern matches the text"""
        search_text = text if self.case_sensitive else text.lower()
        patterns_to_check = self.patterns if self.case_sensitive else [p.lower() for p in self.patterns]
        
        for pattern in patterns_to_check:
            if pattern in search_text:
                return True
        return False
    
    def find_matches(self, text: str) -> List[str]:
        """Find all matching patterns in text"""
        matches = []
        search_text = text if self.case_sensitive else text.lower()
        patterns_to_check = self.patterns if self.case_sensitive else [p.lower() for p in self.patterns]
        
        for pattern in patterns_to_check:
            if pattern in search_text:
                matches.append(pattern)
        return matches


class PatternRegistry:
    """Registry of all detection patterns"""
    
    def __init__(self):
        self.patterns: List[DetectionPattern] = []
        self._initialize_default_patterns()
    
    def _initialize_default_patterns(self) -> None:
        """Initialize the default set of detection patterns"""
        
        # Uncertainty markers
        self.patterns.append(DetectionPattern(
            name="uncertainty_markers",
            patterns=[
                "i'm not sure",
                "i think",
                "probably",
                "maybe",
                "might",
                "could be",
                "not certain",
                "unclear",
                "unsure",
                "possibly",
                "perhaps",
                "seems like",
                "appears to",
                "i believe",
                "i guess",
                "not confident",
                "uncertain"
            ],
            description="Markers indicating agent uncertainty",
            severity="high"
        ))
        
        # Error and problem markers
        self.patterns.append(DetectionPattern(
            name="error_markers",
            patterns=[
                "todo",
                "fixme",
                "hack",
                "workaround",
                "temporary",
                "broken",
                "doesn't work",
                "bug",
                "issue",
                "problem",
                "error",
                "failed",
                "incomplete",
                "not implemented",
                "placeholder",
                "stub"
            ],
            description="Markers indicating errors or incomplete work",
            severity="high"
        ))
        
        # Code quality issues
        self.patterns.append(DetectionPattern(
            name="quality_issues",
            patterns=[
                "quick and dirty",
                "dirty hack",
                "bad code",
                "needs refactoring",
                "technical debt",
                "code smell",
                "anti-pattern",
                "not optimal",
                "inefficient",
                "messy"
            ],
            description="Markers indicating code quality issues",
            severity="medium"
        ))
        
        # Security concerns
        self.patterns.append(DetectionPattern(
            name="security_concerns",
            patterns=[
                "security risk",
                "vulnerability",
                "unsafe",
                "insecure",
                "exploit",
                "injection",
                "xss",
                "csrf",
                "sql injection",
                "buffer overflow",
                "race condition",
                "hardcoded password",
                "hardcoded secret"
            ],
            description="Markers indicating security concerns",
            severity="high"
        ))
        
        # Incomplete implementation
        self.patterns.append(DetectionPattern(
            name="incomplete_implementation",
            patterns=[
                "not finished",
                "work in progress",
                "wip",
                "coming soon",
                "to be implemented",
                "tbd",
                "to be determined",
                "needs implementation",
                "missing",
                "not done",
                "partial implementation"
            ],
            description="Markers indicating incomplete implementation",
            severity="high"
        ))
        
        # Complexity warnings
        self.patterns.append(DetectionPattern(
            name="complexity_warnings",
            patterns=[
                "too complex",
                "overly complicated",
                "hard to understand",
                "confusing",
                "convoluted",
                "spaghetti code",
                "needs simplification"
            ],
            description="Markers indicating excessive complexity",
            severity="medium"
        ))
        
        # Dependency issues
        self.patterns.append(DetectionPattern(
            name="dependency_issues",
            patterns=[
                "missing dependency",
                "dependency conflict",
                "version mismatch",
                "compatibility issue",
                "deprecated",
                "obsolete",
                "legacy code"
            ],
            description="Markers indicating dependency problems",
            severity="medium"
        ))
        
        # Testing gaps
        self.patterns.append(DetectionPattern(
            name="testing_gaps",
            patterns=[
                "not tested",
                "untested",
                "needs tests",
                "missing tests",
                "no test coverage",
                "skip test",
                "test disabled"
            ],
            description="Markers indicating testing gaps",
            severity="medium"
        ))
    
    def add_pattern(self, pattern: DetectionPattern) -> None:
        """Add a custom detection pattern"""
        self.patterns.append(pattern)
    
    def get_pattern(self, name: str) -> DetectionPattern:
        """Get a pattern by name"""
        for pattern in self.patterns:
            if pattern.name == name:
                return pattern
        raise ValueError(f"Pattern not found: {name}")
    
    def get_patterns_by_severity(self, severity: str) -> List[DetectionPattern]:
        """Get all patterns of a specific severity"""
        return [p for p in self.patterns if p.severity == severity]
    
    def check_all_patterns(self, text: str) -> Dict[str, List[str]]:
        """Check text against all patterns and return matches"""
        matches = {}
        for pattern in self.patterns:
            pattern_matches = pattern.find_matches(text)
            if pattern_matches:
                matches[pattern.name] = pattern_matches
        return matches


class FormatValidator:
    """Validates format correctness of responses"""
    
    @staticmethod
    def check_balanced_delimiters(text: str) -> Dict[str, bool]:
        """Check if delimiters are balanced"""
        return {
            "parentheses": text.count('(') == text.count(')'),
            "brackets": text.count('[') == text.count(']'),
            "braces": text.count('{') == text.count('}'),
            "quotes": text.count('"') % 2 == 0,
            "single_quotes": text.count("'") % 2 == 0
        }
    
    @staticmethod
    def has_format_errors(text: str) -> bool:
        """Check if text has any format errors"""
        balanced = FormatValidator.check_balanced_delimiters(text)
        return not all(balanced.values())
    
    @staticmethod
    def get_format_error_details(text: str) -> List[str]:
        """Get detailed format error information"""
        balanced = FormatValidator.check_balanced_delimiters(text)
        errors = []
        
        if not balanced["parentheses"]:
            errors.append("Unbalanced parentheses")
        if not balanced["brackets"]:
            errors.append("Unbalanced brackets")
        if not balanced["braces"]:
            errors.append("Unbalanced braces")
        if not balanced["quotes"]:
            errors.append("Unbalanced double quotes")
        if not balanced["single_quotes"]:
            errors.append("Unbalanced single quotes")
        
        return errors


class LengthValidator:
    """Validates response length constraints"""
    
    @staticmethod
    def is_too_long(text: str, max_length: int) -> bool:
        """Check if text exceeds maximum length"""
        return len(text) > max_length
    
    @staticmethod
    def is_too_short(text: str, min_length: int) -> bool:
        """Check if text is below minimum length"""
        return len(text.strip()) < min_length
    
    @staticmethod
    def get_line_count(text: str) -> int:
        """Get number of non-empty lines"""
        return len([line for line in text.split('\n') if line.strip()])
    
    @staticmethod
    def exceeds_line_limit(text: str, max_lines: int) -> bool:
        """Check if text exceeds line limit"""
        return LengthValidator.get_line_count(text) > max_lines
    
    @staticmethod
    def get_length_stats(text: str) -> Dict[str, int]:
        """Get comprehensive length statistics"""
        return {
            "total_chars": len(text),
            "non_whitespace_chars": len(text.replace(" ", "").replace("\n", "").replace("\t", "")),
            "total_lines": len(text.split('\n')),
            "non_empty_lines": LengthValidator.get_line_count(text),
            "words": len(text.split())
        }
