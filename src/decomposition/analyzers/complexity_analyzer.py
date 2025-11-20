"""
Complexity analysis for task decomposition.

This module analyzes task complexity to determine if tasks need further
decomposition or are atomic enough for direct implementation.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import sys
sys.path.append('/home/niel/git/multiAgent')
from src.core.task_spec.types import (
    TypedParameter,
    Predicate,
    TestCase,
    Property,
    PerformanceRequirement
)


@dataclass
class ComplexityMetrics:
    """
    Metrics for task complexity analysis.
    
    Attributes:
        estimated_lines: Estimated lines of code needed
        estimated_complexity: Estimated cyclomatic complexity
        nesting_depth: Estimated nesting depth
        parameter_count: Number of input/output parameters
        predicate_count: Number of preconditions/postconditions
        test_case_count: Number of test cases
        is_atomic: Whether task is atomic (cannot be decomposed further)
    """
    estimated_lines: int
    estimated_complexity: int
    nesting_depth: int
    parameter_count: int
    predicate_count: int
    test_case_count: int
    is_atomic: bool
    
    def validate_metrics(self) -> bool:
        """Validate all metrics are non-negative."""
        return all([
            self.estimated_lines >= 0,
            self.estimated_complexity >= 0,
            self.nesting_depth >= 0,
            self.parameter_count >= 0,
            self.predicate_count >= 0,
            self.test_case_count >= 0
        ])
    
    def is_valid(self) -> bool:
        """Check if metrics are valid."""
        return self.validate_metrics()


class ComplexityAnalyzer:
    """
    Analyzes task complexity to guide decomposition decisions.
    
    This analyzer estimates various complexity metrics to determine if a task
    should be decomposed further or is atomic enough for implementation.
    """
    
    # Thresholds from architecture
    MAX_ATOMIC_LINES = 20
    MAX_ATOMIC_COMPLEXITY = 5
    MAX_ATOMIC_NESTING = 3
    
    def __init__(
        self,
        max_lines: int = MAX_ATOMIC_LINES,
        max_complexity: int = MAX_ATOMIC_COMPLEXITY,
        max_nesting: int = MAX_ATOMIC_NESTING
    ):
        """
        Initialize complexity analyzer.
        
        Args:
            max_lines: Maximum lines for atomic task
            max_complexity: Maximum cyclomatic complexity for atomic task
            max_nesting: Maximum nesting depth for atomic task
        """
        self.max_lines = max_lines
        self.max_complexity = max_complexity
        self.max_nesting = max_nesting
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.max_lines < 1:
            return (False, "max_lines must be at least 1")
        
        if self.max_complexity < 1:
            return (False, "max_complexity must be at least 1")
        
        if self.max_nesting < 1:
            return (False, "max_nesting must be at least 1")
        
        return (True, "")
    
    def analyze(
        self,
        description: str,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        test_cases: List[TestCase],
        properties: List[Property]
    ) -> Tuple[bool, Optional[ComplexityMetrics], str]:
        """
        Analyze task complexity.
        
        Args:
            description: Task description
            inputs: Input parameters
            outputs: Output parameters
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
            test_cases: Test cases
            properties: Properties for property-based testing
        
        Returns:
            Tuple of (success, metrics or None, message)
        """
        # Validate inputs
        if not description:
            return (False, None, "description cannot be empty")
        
        # Estimate lines of code
        estimated_lines = self._estimate_lines(
            inputs, outputs, preconditions, postconditions
        )
        
        # Estimate cyclomatic complexity
        estimated_complexity = self._estimate_complexity(
            preconditions, postconditions, description
        )
        
        # Estimate nesting depth
        nesting_depth = self._estimate_nesting(
            preconditions, postconditions, description
        )
        
        # Count parameters and predicates
        parameter_count = len(inputs) + len(outputs)
        predicate_count = len(preconditions) + len(postconditions)
        test_case_count = len(test_cases) + len(properties)
        
        # Determine if atomic
        is_atomic = self._is_atomic(
            estimated_lines,
            estimated_complexity,
            nesting_depth
        )
        
        metrics = ComplexityMetrics(
            estimated_lines=estimated_lines,
            estimated_complexity=estimated_complexity,
            nesting_depth=nesting_depth,
            parameter_count=parameter_count,
            predicate_count=predicate_count,
            test_case_count=test_case_count,
            is_atomic=is_atomic
        )
        
        if not metrics.is_valid():
            return (False, None, "Invalid metrics calculated")
        
        return (True, metrics, "Analysis complete")
    
    def _estimate_lines(
        self,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: List[Predicate],
        postconditions: List[Predicate]
    ) -> int:
        """
        Estimate lines of code needed.
        
        Args:
            inputs: Input parameters
            outputs: Output parameters
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
        
        Returns:
            Estimated line count
        """
        # Base lines for function signature and return
        base_lines = 5
        
        # Lines for input validation (1 line per precondition)
        validation_lines = len(preconditions)
        
        # Lines for core logic (estimate based on parameters)
        # More parameters typically means more complex logic
        logic_lines = max(3, len(inputs) * 2)
        
        # Lines for output construction
        output_lines = len(outputs)
        
        # Lines for postcondition checks
        postcondition_lines = len(postconditions)
        
        total = (
            base_lines +
            validation_lines +
            logic_lines +
            output_lines +
            postcondition_lines
        )
        
        return total
    
    def _estimate_complexity(
        self,
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        description: str
    ) -> int:
        """
        Estimate cyclomatic complexity.
        
        Args:
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
            description: Task description
        
        Returns:
            Estimated complexity
        """
        # Base complexity
        complexity = 1
        
        # Add complexity for each precondition (validation branches)
        complexity += len(preconditions)
        
        # Add complexity for conditional postconditions
        for postcond in postconditions:
            if 'if' in postcond.expression.lower():
                complexity += 1
        
        # Analyze description for complexity indicators
        desc_lower = description.lower()
        
        # Keywords that indicate branching
        branch_keywords = ['if', 'when', 'case', 'either', 'or', 'depending']
        for keyword in branch_keywords:
            if keyword in desc_lower:
                complexity += 1
        
        # Keywords that indicate loops
        loop_keywords = ['for each', 'for all', 'iterate', 'loop', 'while']
        for keyword in loop_keywords:
            if keyword in desc_lower:
                complexity += 2  # Loops add more complexity
        
        return complexity
    
    def _estimate_nesting(
        self,
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        description: str
    ) -> int:
        """
        Estimate nesting depth.
        
        Args:
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
            description: Task description
        
        Returns:
            Estimated nesting depth
        """
        # Base nesting (function level)
        nesting = 1
        
        # Check for nested conditions in predicates
        all_predicates = preconditions + postconditions
        for pred in all_predicates:
            expr = pred.expression.lower()
            if 'and' in expr or 'or' in expr:
                nesting = max(nesting, 2)
        
        # Analyze description for nesting indicators
        desc_lower = description.lower()
        
        # Nested conditionals
        if 'if' in desc_lower and 'then' in desc_lower:
            nesting = max(nesting, 2)
        
        # Nested loops
        if 'for each' in desc_lower:
            count = desc_lower.count('for each')
            nesting = max(nesting, count + 1)
        
        return nesting
    
    def _is_atomic(
        self,
        estimated_lines: int,
        estimated_complexity: int,
        nesting_depth: int
    ) -> bool:
        """
        Determine if task is atomic.
        
        Args:
            estimated_lines: Estimated line count
            estimated_complexity: Estimated complexity
            nesting_depth: Estimated nesting depth
        
        Returns:
            True if task is atomic, False otherwise
        """
        return (
            estimated_lines <= self.max_lines and
            estimated_complexity <= self.max_complexity and
            nesting_depth <= self.max_nesting
        )
    
    def get_decomposition_recommendation(
        self,
        metrics: ComplexityMetrics
    ) -> Tuple[bool, str]:
        """
        Get recommendation on whether to decompose.
        
        Args:
            metrics: Complexity metrics
        
        Returns:
            Tuple of (should_decompose, reason)
        """
        if not metrics.is_valid():
            return (False, "Invalid metrics")
        
        if metrics.is_atomic:
            return (False, "Task is atomic - no decomposition needed")
        
        reasons = []
        
        if metrics.estimated_lines > self.max_lines:
            reasons.append(
                f"Estimated {metrics.estimated_lines} lines exceeds limit of {self.max_lines}"
            )
        
        if metrics.estimated_complexity > self.max_complexity:
            reasons.append(
                f"Estimated complexity {metrics.estimated_complexity} exceeds limit of {self.max_complexity}"
            )
        
        if metrics.nesting_depth > self.max_nesting:
            reasons.append(
                f"Nesting depth {metrics.nesting_depth} exceeds limit of {self.max_nesting}"
            )
        
        reason = "; ".join(reasons)
        return (True, reason)
    
    def suggest_decomposition_count(
        self,
        metrics: ComplexityMetrics
    ) -> int:
        """
        Suggest number of subtasks to decompose into.
        
        Args:
            metrics: Complexity metrics
        
        Returns:
            Suggested number of subtasks
        """
        if metrics.is_atomic:
            return 1
        
        # Calculate how many times over the limits we are
        line_factor = metrics.estimated_lines / self.max_lines
        complexity_factor = metrics.estimated_complexity / self.max_complexity
        
        # Use the maximum factor
        max_factor = max(line_factor, complexity_factor)
        
        # Suggest decomposition count (minimum 2, maximum 10)
        suggested_count = int(max_factor) + 1
        suggested_count = max(2, min(suggested_count, 10))
        
        return suggested_count
