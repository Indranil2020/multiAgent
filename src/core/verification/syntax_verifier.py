"""
Syntax Verifier - Layer 1 of Verification Stack

This module provides syntax verification for code using AST parsing.
It validates that code is syntactically correct before proceeding to other verification layers.
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
class SyntaxVerifierConfig:
    """Configuration for syntax verifier"""
    strict_mode: bool = True
    check_encoding: bool = True
    allowed_syntax_features: Optional[List[str]] = None
    max_nesting_depth: int = 10


class SyntaxVerifier:
    """
    Verifies syntax correctness of Python code using AST parsing.
    
    This verifier ensures code can be parsed into a valid Abstract Syntax Tree
    and optionally checks for syntax feature restrictions and nesting depth.
    """
    
    def __init__(self, config: Optional[SyntaxVerifierConfig] = None):
        """
        Initialize the syntax verifier.
        
        Args:
            config: Configuration for the verifier
        """
        self.config = config if config is not None else SyntaxVerifierConfig()
    
    def verify(self, code: str, code_id: str = "unknown") -> VerificationResult:
        """
        Verify syntax of the provided code.
        
        Args:
            code: Source code to verify
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with syntax verification status
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
        
        # Parse code into AST
        parse_result = self._parse_code(code)
        
        if not parse_result["success"]:
            return self._create_failed_result(
                parse_result["error"],
                parse_result.get("line_number"),
                parse_result.get("column_number"),
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        tree = parse_result["tree"]
        
        # Additional syntax checks
        issues = []
        
        # Check nesting depth
        if self.config.strict_mode:
            depth_issues = self._check_nesting_depth(tree)
            issues.extend(depth_issues)
        
        # Check for restricted syntax features
        if self.config.allowed_syntax_features is not None:
            feature_issues = self._check_syntax_features(tree)
            issues.extend(feature_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.SYNTAX,
                status=VerificationStatus.PASSED,
                passed=True,
                message="Syntax verification passed",
                execution_time_ms=execution_time,
                details={"ast_node_count": self._count_nodes(tree)}
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.SYNTAX,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Syntax verification failed with {len(issues)} issue(s)",
                execution_time_ms=execution_time,
                issues=issues
            )
            return result
    
    def _parse_code(self, code: str) -> Dict[str, Any]:
        """
        Parse code into AST.
        
        Args:
            code: Source code to parse
            
        Returns:
            Dictionary with parse result
        """
        # Attempt to parse the code
        parse_error = None
        tree = None
        
        # Manual parsing without try-except
        compile_result = self._safe_compile(code)
        
        if compile_result["success"]:
            tree = compile_result["tree"]
            return {
                "success": True,
                "tree": tree,
                "error": None
            }
        else:
            return {
                "success": False,
                "tree": None,
                "error": compile_result["error"],
                "line_number": compile_result.get("line_number"),
                "column_number": compile_result.get("column_number")
            }
    
    def _safe_compile(self, code: str) -> Dict[str, Any]:
        """
        Safely compile code to AST.
        
        Args:
            code: Source code
            
        Returns:
            Dictionary with compilation result
        """
        # Use compile with error handling through return values
        tree = None
        error_msg = None
        line_num = None
        col_num = None
        
        # Attempt compilation
        compiled = compile(code, "<string>", "exec", ast.PyCF_ONLY_AST, dont_inherit=True) if self._is_valid_python_code(code) else None
        
        if compiled is not None:
            return {
                "success": True,
                "tree": compiled,
                "error": None
            }
        else:
            # Extract error information
            error_info = self._extract_syntax_error(code)
            return {
                "success": False,
                "tree": None,
                "error": error_info["message"],
                "line_number": error_info.get("line_number"),
                "column_number": error_info.get("column_number")
            }
    
    def _is_valid_python_code(self, code: str) -> bool:
        """
        Check if code is valid Python without using exceptions.
        
        Args:
            code: Source code
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation checks
        if not code or not isinstance(code, str):
            return False
        
        # Check for balanced brackets
        if not self._check_balanced_brackets(code):
            return False
        
        # Check for valid indentation
        if not self._check_indentation(code):
            return False
        
        return True
    
    def _check_balanced_brackets(self, code: str) -> bool:
        """Check if brackets are balanced"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in code:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if len(stack) == 0:
                    return False
                if pairs[stack[-1]] != char:
                    return False
                stack.pop()
        
        return len(stack) == 0
    
    def _check_indentation(self, code: str) -> bool:
        """Check for valid indentation"""
        lines = code.split('\n')
        indent_stack = [0]
        
        for line in lines:
            stripped = line.lstrip()
            if len(stripped) == 0 or stripped.startswith('#'):
                continue
            
            indent = len(line) - len(stripped)
            
            # Indentation must be multiple of 4 or consistent
            if indent % 4 != 0 and indent % 2 != 0:
                return False
        
        return True
    
    def _extract_syntax_error(self, code: str) -> Dict[str, Any]:
        """
        Extract syntax error information from code.
        
        Args:
            code: Source code
            
        Returns:
            Dictionary with error information
        """
        # Perform basic syntax error detection
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Check for common syntax errors
            if line.strip().endswith(':') and i + 1 < len(lines):
                next_line = lines[i + 1]
                if len(next_line.strip()) > 0 and len(next_line) - len(next_line.lstrip()) <= len(line) - len(line.lstrip()):
                    return {
                        "message": f"Expected indented block after ':' on line {i + 1}",
                        "line_number": i + 2,
                        "column_number": 0
                    }
        
        return {
            "message": "Syntax error detected",
            "line_number": None,
            "column_number": None
        }
    
    def _check_nesting_depth(self, tree: ast.AST) -> List[VerificationIssue]:
        """
        Check nesting depth of code.
        
        Args:
            tree: AST tree
            
        Returns:
            List of issues found
        """
        issues = []
        max_depth = self._calculate_max_depth(tree)
        
        if max_depth > self.config.max_nesting_depth:
            issues.append(VerificationIssue(
                layer=VerificationLayer.SYNTAX,
                severity="medium",
                message=f"Nesting depth {max_depth} exceeds maximum {self.config.max_nesting_depth}",
                suggestion="Consider refactoring to reduce nesting depth"
            ))
        
        return issues
    
    def _calculate_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth"""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.FunctionDef, ast.ClassDef)):
                child_depth = self._calculate_max_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _check_syntax_features(self, tree: ast.AST) -> List[VerificationIssue]:
        """
        Check for restricted syntax features.
        
        Args:
            tree: AST tree
            
        Returns:
            List of issues found
        """
        issues = []
        
        # This would check for specific syntax features if restrictions are configured
        # For now, return empty list
        
        return issues
    
    def _count_nodes(self, tree: ast.AST) -> int:
        """Count total nodes in AST"""
        count = 1
        for child in ast.iter_child_nodes(tree):
            count += self._count_nodes(child)
        return count
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.SYNTAX,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
    
    def _create_failed_result(
        self,
        error_message: str,
        line_number: Optional[int] = None,
        column_number: Optional[int] = None,
        execution_time_ms: float = 0.0
    ) -> VerificationResult:
        """Create a failed result"""
        issue = VerificationIssue(
            layer=VerificationLayer.SYNTAX,
            severity="critical",
            message=error_message,
            line_number=line_number,
            column_number=column_number
        )
        
        return VerificationResult(
            layer=VerificationLayer.SYNTAX,
            status=VerificationStatus.FAILED,
            passed=False,
            message=f"Syntax error: {error_message}",
            issues=[issue],
            execution_time_ms=execution_time_ms
        )
