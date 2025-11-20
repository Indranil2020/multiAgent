"""
Static Analyzer - Layer 6 of Verification Stack

This module provides static code analysis for quality and maintainability.
It checks code complexity, style, and potential issues without executing code.
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
class StaticAnalyzerConfig:
    """Configuration for static analyzer"""
    max_complexity: int = 10
    max_function_length: int = 50
    max_class_length: int = 300
    max_nesting_depth: int = 4
    max_parameters: int = 5
    check_naming: bool = True
    check_docstrings: bool = True
    check_code_smells: bool = True


class StaticAnalyzer:
    """
    Performs static analysis on code for quality and maintainability.
    
    This analyzer checks code complexity, style conventions, and potential
    code smells without executing the code.
    """
    
    def __init__(self, config: Optional[StaticAnalyzerConfig] = None):
        """
        Initialize the static analyzer.
        
        Args:
            config: Configuration for the analyzer
        """
        self.config = config if config is not None else StaticAnalyzerConfig()
    
    def verify(self, code: str, code_id: str = "unknown") -> VerificationResult:
        """
        Perform static analysis on the provided code.
        
        Args:
            code: Source code to analyze
            code_id: Identifier for the code being analyzed
            
        Returns:
            VerificationResult with static analysis status
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
                "Failed to parse code for static analysis",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Perform static analysis checks
        issues = []
        
        # Check complexity
        complexity_issues = self._check_complexity(tree)
        issues.extend(complexity_issues)
        
        # Check code length
        length_issues = self._check_length(tree, code)
        issues.extend(length_issues)
        
        # Check nesting depth
        nesting_issues = self._check_nesting_depth(tree)
        issues.extend(nesting_issues)
        
        # Check naming conventions
        if self.config.check_naming:
            naming_issues = self._check_naming_conventions(tree)
            issues.extend(naming_issues)
        
        # Check docstrings
        if self.config.check_docstrings:
            docstring_issues = self._check_docstrings(tree)
            issues.extend(docstring_issues)
        
        # Check code smells
        if self.config.check_code_smells:
            smell_issues = self._check_code_smells(tree)
            issues.extend(smell_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate metrics
        metrics = self._calculate_metrics(tree, code)
        
        # Create result
        critical_issues = len([i for i in issues if i.severity == "critical"])
        high_issues = len([i for i in issues if i.severity == "high"])
        
        if critical_issues == 0 and high_issues == 0:
            return VerificationResult(
                layer=VerificationLayer.STATIC_ANALYSIS,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"Static analysis passed with {len(issues)} minor issue(s)",
                execution_time_ms=execution_time,
                issues=issues,
                details=metrics
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.STATIC_ANALYSIS,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Static analysis failed with {len(issues)} issue(s)",
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
    
    def _check_complexity(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check cyclomatic complexity"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                
                if complexity > self.config.max_complexity:
                    severity = "high" if complexity > self.config.max_complexity * 1.5 else "medium"
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity=severity,
                        message=f"Function '{node.name}' has complexity {complexity} (max: {self.config.max_complexity})",
                        line_number=node.lineno,
                        suggestion="Consider refactoring to reduce complexity"
                    ))
        
        return issues
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _check_length(self, tree: ast.AST, code: str) -> List[VerificationIssue]:
        """Check code length"""
        issues = []
        
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate function length
                func_lines = self._get_node_lines(node, lines)
                
                if len(func_lines) > self.config.max_function_length:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="medium",
                        message=f"Function '{node.name}' is {len(func_lines)} lines (max: {self.config.max_function_length})",
                        line_number=node.lineno,
                        suggestion="Consider breaking function into smaller functions"
                    ))
                
                # Check parameter count
                param_count = len(node.args.args)
                if param_count > self.config.max_parameters:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="low",
                        message=f"Function '{node.name}' has {param_count} parameters (max: {self.config.max_parameters})",
                        line_number=node.lineno,
                        suggestion="Consider using a configuration object or reducing parameters"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Calculate class length
                class_lines = self._get_node_lines(node, lines)
                
                if len(class_lines) > self.config.max_class_length:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="medium",
                        message=f"Class '{node.name}' is {len(class_lines)} lines (max: {self.config.max_class_length})",
                        line_number=node.lineno,
                        suggestion="Consider breaking class into smaller classes"
                    ))
        
        return issues
    
    def _get_node_lines(self, node: ast.AST, lines: List[str]) -> List[str]:
        """Get lines of code for a node"""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return []
        
        start = node.lineno - 1
        end = node.end_lineno if node.end_lineno else node.lineno
        
        return lines[start:end]
    
    def _check_nesting_depth(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check nesting depth"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                max_depth = self._calculate_max_depth(node)
                
                if max_depth > self.config.max_nesting_depth:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="medium",
                        message=f"Function '{node.name}' has nesting depth {max_depth} (max: {self.config.max_nesting_depth})",
                        line_number=node.lineno,
                        suggestion="Consider extracting nested logic into separate functions"
                    ))
        
        return issues
    
    def _calculate_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                child_depth = self._calculate_max_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _check_naming_conventions(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check naming conventions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function naming (should be snake_case)
                if not self._is_snake_case(node.name) and not node.name.startswith('__'):
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="low",
                        message=f"Function '{node.name}' should use snake_case naming",
                        line_number=node.lineno,
                        suggestion=f"Rename to '{self._to_snake_case(node.name)}'"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Check class naming (should be PascalCase)
                if not self._is_pascal_case(node.name):
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="low",
                        message=f"Class '{node.name}' should use PascalCase naming",
                        line_number=node.lineno
                    ))
            
            elif isinstance(node, ast.Name):
                # Check variable naming
                if node.id.isupper() and len(node.id) > 1:
                    # Constants are OK
                    pass
                elif not self._is_snake_case(node.id) and not node.id.startswith('_'):
                    # Only report if not a built-in or common pattern
                    if node.id not in ['i', 'j', 'k', 'x', 'y', 'z', 'n', 'm']:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.STATIC_ANALYSIS,
                            severity="low",
                            message=f"Variable '{node.id}' should use snake_case naming",
                            line_number=node.lineno if hasattr(node, 'lineno') else None
                        ))
        
        return issues
    
    def _is_snake_case(self, name: str) -> bool:
        """Check if name is in snake_case"""
        return name.islower() or '_' in name and name.replace('_', '').islower()
    
    def _is_pascal_case(self, name: str) -> bool:
        """Check if name is in PascalCase"""
        return name[0].isupper() and '_' not in name
    
    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case"""
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())
        return ''.join(result)
    
    def _check_docstrings(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for docstrings"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Skip private methods
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                
                docstring = ast.get_docstring(node)
                
                if not docstring:
                    node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="low",
                        message=f"{node_type} '{node.name}' lacks docstring",
                        line_number=node.lineno,
                        suggestion="Add docstring describing purpose and parameters"
                    ))
        
        return issues
    
    def _check_code_smells(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for code smells"""
        issues = []
        
        # Check for duplicate code
        duplicate_issues = self._check_duplicates(tree)
        issues.extend(duplicate_issues)
        
        # Check for magic numbers
        magic_number_issues = self._check_magic_numbers(tree)
        issues.extend(magic_number_issues)
        
        # Check for long parameter lists
        # Already checked in _check_length
        
        # Check for dead code
        dead_code_issues = self._check_dead_code(tree)
        issues.extend(dead_code_issues)
        
        return issues
    
    def _check_duplicates(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for duplicate code blocks"""
        issues = []
        
        # This is simplified - real implementation would use more sophisticated duplicate detection
        # For now, we just check for identical function bodies
        
        function_bodies = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                body_str = ast.unparse(node.body) if hasattr(ast, 'unparse') else str(node.body)
                
                if body_str in function_bodies:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.STATIC_ANALYSIS,
                        severity="medium",
                        message=f"Function '{node.name}' has duplicate code with '{function_bodies[body_str]}'",
                        line_number=node.lineno,
                        suggestion="Consider extracting common code into a shared function"
                    ))
                else:
                    function_bodies[body_str] = node.name
        
        return issues
    
    def _check_magic_numbers(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for magic numbers"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                # Check if it's a number (not string or bool)
                if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    # Ignore common values
                    if node.value not in [0, 1, -1, 2]:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.STATIC_ANALYSIS,
                            severity="low",
                            message=f"Magic number {node.value} found",
                            line_number=node.lineno if hasattr(node, 'lineno') else None,
                            suggestion="Consider defining as a named constant"
                        ))
        
        return issues
    
    def _check_dead_code(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check for dead code"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for unreachable code after return
                has_return = False
                
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return):
                        has_return = True
                        
                        # Check if there's code after return
                        if i < len(node.body) - 1:
                            issues.append(VerificationIssue(
                                layer=VerificationLayer.STATIC_ANALYSIS,
                                severity="medium",
                                message=f"Unreachable code after return in function '{node.name}'",
                                line_number=stmt.lineno if hasattr(stmt, 'lineno') else None,
                                suggestion="Remove unreachable code"
                            ))
        
        return issues
    
    def _calculate_metrics(self, tree: ast.AST, code: str) -> Dict[str, Any]:
        """Calculate code metrics"""
        metrics = {
            "total_lines": len(code.split('\n')),
            "code_lines": len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in code.split('\n') if line.strip().startswith('#')]),
            "blank_lines": len([line for line in code.split('\n') if not line.strip()]),
            "functions": 0,
            "classes": 0,
            "avg_complexity": 0.0,
            "max_complexity": 0
        }
        
        complexities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
                complexity = self._calculate_complexity(node)
                complexities.append(complexity)
                metrics["max_complexity"] = max(metrics["max_complexity"], complexity)
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
        
        if len(complexities) > 0:
            metrics["avg_complexity"] = sum(complexities) / len(complexities)
        
        return metrics
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.STATIC_ANALYSIS,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
