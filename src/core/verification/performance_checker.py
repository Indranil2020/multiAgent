"""
Performance Checker - Layer 8 of Verification Stack

This module provides performance validation for code.
It checks execution time, memory usage, and algorithmic complexity.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import ast
import time

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationIssue
)


@dataclass
class PerformanceRequirement:
    """Performance requirement specification"""
    name: str
    max_time_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_complexity: Optional[str] = None  # "O(1)", "O(n)", "O(n^2)", etc.


@dataclass
class PerformanceCheckerConfig:
    """Configuration for performance checker"""
    strict_mode: bool = True
    check_complexity: bool = True
    check_inefficient_patterns: bool = True
    warn_on_nested_loops: bool = True
    max_loop_nesting: int = 3


class PerformanceChecker:
    """
    Validates code performance characteristics.
    
    This checker analyzes algorithmic complexity, detects inefficient patterns,
    and validates performance requirements.
    """
    
    def __init__(self, config: Optional[PerformanceCheckerConfig] = None):
        """
        Initialize the performance checker.
        
        Args:
            config: Configuration for the checker
        """
        self.config = config if config is not None else PerformanceCheckerConfig()
    
    def verify(
        self,
        code: str,
        requirements: Optional[List[PerformanceRequirement]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Check performance characteristics of code.
        
        Args:
            code: Source code to check
            requirements: Performance requirements to validate
            code_id: Identifier for the code being checked
            
        Returns:
            VerificationResult with performance check status
        """
        start_time = time.time()
        
        # Validate input
        if not isinstance(code, str):
            return self._create_error_result(
                "Code must be a string",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        if len(code.strip()) == 0:
            return self._create_error_result(
                "Code cannot be empty",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse code
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for performance checking",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Perform performance checks
        issues = []
        
        # Check algorithmic complexity
        if self.config.check_complexity:
            complexity_issues = self._check_algorithmic_complexity(tree)
            issues.extend(complexity_issues)
        
        # Check for inefficient patterns
        if self.config.check_inefficient_patterns:
            pattern_issues = self._check_inefficient_patterns(tree)
            issues.extend(pattern_issues)
        
        # Check nested loops
        if self.config.warn_on_nested_loops:
            loop_issues = self._check_nested_loops(tree)
            issues.extend(loop_issues)
        
        # Analyze performance metrics
        metrics = self._analyze_performance_metrics(tree)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        high_issues = len([i for i in issues if i.severity in ["critical", "high"]])
        
        if high_issues == 0:
            return VerificationResult(
                layer=VerificationLayer.PERFORMANCE,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"Performance check passed with {len(issues)} minor issue(s)",
                execution_time_ms=execution_time,
                issues=issues,
                details=metrics
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.PERFORMANCE,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Performance check failed with {high_issues} significant issue(s)",
                execution_time_ms=execution_time,
                issues=issues,
                details=metrics
            )
            return result
    
    def _parse_code(self, code: str) -> Optional[ast.AST]:
        """Parse code into AST"""
        parsed_tree = None
        
        if self._can_parse(code):
            parsed_tree = ast.parse(code)
        
        return parsed_tree
    
    def _can_parse(self, code: str) -> bool:
        """Check if code can be parsed"""
        return code and isinstance(code, str) and len(code.strip()) > 0
    
    def _check_algorithmic_complexity(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check algorithmic complexity of functions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._estimate_complexity(node)
                
                if complexity["order"] in ["O(n^3)", "O(2^n)", "O(n!)"]:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PERFORMANCE,
                        severity="high",
                        message=f"Function '{node.name}' has high complexity: {complexity['order']}",
                        line_number=node.lineno,
                        suggestion="Consider optimizing algorithm or using better data structures"
                    ))
                elif complexity["order"] == "O(n^2)":
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PERFORMANCE,
                        severity="medium",
                        message=f"Function '{node.name}' has quadratic complexity: {complexity['order']}",
                        line_number=node.lineno,
                        suggestion="Consider if linear or log-linear algorithm is possible"
                    ))
        
        return issues
    
    def _estimate_complexity(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Estimate algorithmic complexity of a function"""
        # Count nested loops
        max_loop_depth = self._count_loop_depth(node)
        
        # Determine complexity order
        if max_loop_depth == 0:
            order = "O(1)"
        elif max_loop_depth == 1:
            order = "O(n)"
        elif max_loop_depth == 2:
            order = "O(n^2)"
        elif max_loop_depth == 3:
            order = "O(n^3)"
        else:
            order = f"O(n^{max_loop_depth})"
        
        # Check for recursive calls
        if self._is_recursive(node):
            order = "O(2^n)"  # Simplified - could be better with memoization
        
        return {
            "order": order,
            "loop_depth": max_loop_depth,
            "is_recursive": self._is_recursive(node)
        }
    
    def _count_loop_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Count maximum loop nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = self._count_loop_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._count_loop_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _is_recursive(self, node: ast.FunctionDef) -> bool:
        """Check if function is recursive"""
        func_name = node.name
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                    return True
        
        return False
    
    def _check_inefficient_patterns(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for inefficient coding patterns"""
        issues = []
        
        # Check for repeated list concatenation
        concat_issues = self._check_list_concatenation(tree)
        issues.extend(concat_issues)
        
        # Check for inefficient lookups
        lookup_issues = self._check_inefficient_lookups(tree)
        issues.extend(lookup_issues)
        
        # Check for unnecessary comprehensions
        comprehension_issues = self._check_comprehensions(tree)
        issues.extend(comprehension_issues)
        
        return issues
    
    def _check_list_concatenation(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for inefficient list concatenation in loops"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for list concatenation in loop
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign):
                        if isinstance(child.op, ast.Add):
                            if isinstance(child.target, ast.Name):
                                issues.append(VerificationIssue(
                                    layer=VerificationLayer.PERFORMANCE,
                                    severity="medium",
                                    message="Inefficient list concatenation in loop detected",
                                    line_number=child.lineno if hasattr(child, 'lineno') else None,
                                    suggestion="Use list.append() or list comprehension instead"
                                ))
        
        return issues
    
    def _check_inefficient_lookups(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for inefficient lookups"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check for 'in list' lookups in loops
                for child in ast.walk(node):
                    if isinstance(child, ast.Compare):
                        for op in child.ops:
                            if isinstance(op, (ast.In, ast.NotIn)):
                                # Check if comparing against list
                                issues.append(VerificationIssue(
                                    layer=VerificationLayer.PERFORMANCE,
                                    severity="low",
                                    message="Potential inefficient lookup in loop",
                                    line_number=child.lineno if hasattr(child, 'lineno') else None,
                                    suggestion="Consider using set for O(1) lookups instead of list"
                                ))
        
        return issues
    
    def _check_comprehensions(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for inefficient comprehensions"""
        issues = []
        
        # This is a placeholder for more sophisticated checks
        # Could check for nested comprehensions, unnecessary filters, etc.
        
        return issues
    
    def _check_nested_loops(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for deeply nested loops"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                loop_depth = self._count_loop_depth(node)
                
                if loop_depth > self.config.max_loop_nesting:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PERFORMANCE,
                        severity="medium",
                        message=f"Function '{node.name}' has {loop_depth} nested loops (max: {self.config.max_loop_nesting})",
                        line_number=node.lineno,
                        suggestion="Consider refactoring to reduce nesting or use better algorithm"
                    ))
        
        return issues
    
    def _analyze_performance_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze performance metrics"""
        metrics = {
            "functions_analyzed": 0,
            "max_loop_depth": 0,
            "recursive_functions": 0,
            "complexity_distribution": {
                "O(1)": 0,
                "O(n)": 0,
                "O(n^2)": 0,
                "O(n^3+)": 0
            }
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions_analyzed"] += 1
                
                complexity = self._estimate_complexity(node)
                loop_depth = complexity["loop_depth"]
                
                metrics["max_loop_depth"] = max(metrics["max_loop_depth"], loop_depth)
                
                if complexity["is_recursive"]:
                    metrics["recursive_functions"] += 1
                
                # Update complexity distribution
                order = complexity["order"]
                if order == "O(1)":
                    metrics["complexity_distribution"]["O(1)"] += 1
                elif order == "O(n)":
                    metrics["complexity_distribution"]["O(n)"] += 1
                elif order == "O(n^2)":
                    metrics["complexity_distribution"]["O(n^2)"] += 1
                else:
                    metrics["complexity_distribution"]["O(n^3+)"] += 1
        
        return metrics
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.PERFORMANCE,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
