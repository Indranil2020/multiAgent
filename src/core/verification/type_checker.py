"""
Type Checker - Layer 2 of Verification Stack

This module provides type checking verification for Python code.
It validates type annotations and ensures type safety throughout the code.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set
import ast
import time

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationIssue
)


@dataclass
class TypeCheckerConfig:
    """Configuration for type checker"""
    strict_mode: bool = True
    require_annotations: bool = True
    check_return_types: bool = True
    check_parameter_types: bool = True
    allow_any_type: bool = False
    check_attribute_access: bool = True


class TypeCheckerInfo:
    """Information about types in code"""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.variables: Dict[str, str] = {}
        self.imports: Set[str] = set()


class TypeChecker:
    """
    Verifies type correctness of Python code.
    
    This verifier checks type annotations, infers types where possible,
    and validates type consistency throughout the code.
    """
    
    def __init__(self, config: Optional[TypeCheckerConfig] = None):
        """
        Initialize the type checker.
        
        Args:
            config: Configuration for the checker
        """
        self.config = config if config is not None else TypeCheckerConfig()
        self.type_info = TypeCheckerInfo()
    
    def verify(self, code: str, code_id: str = "unknown") -> VerificationResult:
        """
        Verify type correctness of the provided code.
        
        Args:
            code: Source code to verify
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with type checking status
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
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for type checking",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Collect type information
        self._collect_type_info(tree)
        
        # Perform type checks
        issues = []
        
        # Check function annotations
        if self.config.require_annotations:
            annotation_issues = self._check_function_annotations(tree)
            issues.extend(annotation_issues)
        
        # Check type consistency
        consistency_issues = self._check_type_consistency(tree)
        issues.extend(consistency_issues)
        
        # Check return types
        if self.config.check_return_types:
            return_issues = self._check_return_types(tree)
            issues.extend(return_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.TYPES,
                status=VerificationStatus.PASSED,
                passed=True,
                message="Type checking passed",
                execution_time_ms=execution_time,
                details={
                    "functions_checked": len(self.type_info.functions),
                    "classes_checked": len(self.type_info.classes)
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.TYPES,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Type checking failed with {len(issues)} issue(s)",
                execution_time_ms=execution_time,
                issues=issues
            )
            return result
    
    def _parse_code(self, code: str) -> Optional[ast.AST]:
        """Parse code into AST"""
        parsed_tree = None
        
        # Attempt to parse
        if self._can_parse(code):
            parsed_tree = ast.parse(code)
        
        return parsed_tree
    
    def _can_parse(self, code: str) -> bool:
        """Check if code can be parsed"""
        if not code or not isinstance(code, str):
            return False
        
        # Basic validation
        return len(code.strip()) > 0
    
    def _collect_type_info(self, tree: ast.AST) -> None:
        """Collect type information from AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._collect_function_info(node)
            elif isinstance(node, ast.ClassDef):
                self._collect_class_info(node)
            elif isinstance(node, ast.AnnAssign):
                self._collect_variable_info(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._collect_import_info(node)
    
    def _collect_function_info(self, node: ast.FunctionDef) -> None:
        """Collect information about a function"""
        func_info = {
            "name": node.name,
            "args": [],
            "return_type": None,
            "has_annotations": False,
            "line_number": node.lineno
        }
        
        # Collect argument types
        for arg in node.args.args:
            arg_info = {
                "name": arg.arg,
                "annotation": ast.unparse(arg.annotation) if arg.annotation else None
            }
            func_info["args"].append(arg_info)
            if arg.annotation:
                func_info["has_annotations"] = True
        
        # Collect return type
        if node.returns:
            func_info["return_type"] = ast.unparse(node.returns)
            func_info["has_annotations"] = True
        
        self.type_info.functions[node.name] = func_info
    
    def _collect_class_info(self, node: ast.ClassDef) -> None:
        """Collect information about a class"""
        class_info = {
            "name": node.name,
            "methods": [],
            "attributes": [],
            "line_number": node.lineno
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info["methods"].append(item.name)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    class_info["attributes"].append(item.target.id)
        
        self.type_info.classes[node.name] = class_info
    
    def _collect_variable_info(self, node: ast.AnnAssign) -> None:
        """Collect information about annotated variables"""
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            var_type = ast.unparse(node.annotation) if node.annotation else "Any"
            self.type_info.variables[var_name] = var_type
    
    def _collect_import_info(self, node: ast.AST) -> None:
        """Collect import information"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.type_info.imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self.type_info.imports.add(node.module)
    
    def _check_function_annotations(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check that functions have proper type annotations"""
        issues = []
        
        for func_name, func_info in self.type_info.functions.items():
            # Skip special methods
            if func_name.startswith('__') and func_name.endswith('__'):
                continue
            
            # Check if function has annotations
            if not func_info["has_annotations"]:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.TYPES,
                    severity="medium",
                    message=f"Function '{func_name}' lacks type annotations",
                    line_number=func_info["line_number"],
                    suggestion="Add type annotations to function parameters and return type"
                ))
            else:
                # Check if all parameters are annotated
                for arg in func_info["args"]:
                    if arg["annotation"] is None and arg["name"] != "self" and arg["name"] != "cls":
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.TYPES,
                            severity="low",
                            message=f"Parameter '{arg['name']}' in function '{func_name}' lacks type annotation",
                            line_number=func_info["line_number"],
                            suggestion=f"Add type annotation to parameter '{arg['name']}'"
                        ))
                
                # Check if return type is annotated
                if func_info["return_type"] is None:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.TYPES,
                        severity="low",
                        message=f"Function '{func_name}' lacks return type annotation",
                        line_number=func_info["line_number"],
                        suggestion="Add return type annotation to function"
                    ))
        
        return issues
    
    def _check_type_consistency(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check type consistency throughout code"""
        issues = []
        
        # This would perform more sophisticated type inference and checking
        # For now, we perform basic checks
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check function calls
                call_issues = self._check_function_call(node)
                issues.extend(call_issues)
        
        return issues
    
    def _check_function_call(self, node: ast.Call) -> List[VerificationIssue]:
        """Check a function call for type consistency"""
        issues = []
        
        # Extract function name
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Check if function is known
        if func_name and func_name in self.type_info.functions:
            func_info = self.type_info.functions[func_name]
            
            # Check argument count
            expected_args = len(func_info["args"])
            provided_args = len(node.args)
            
            if provided_args != expected_args:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.TYPES,
                    severity="high",
                    message=f"Function '{func_name}' expects {expected_args} arguments but {provided_args} provided",
                    line_number=node.lineno if hasattr(node, 'lineno') else None
                ))
        
        return issues
    
    def _check_return_types(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check that return statements match declared return types"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return_issues = self._check_function_returns(node)
                issues.extend(return_issues)
        
        return issues
    
    def _check_function_returns(self, node: ast.FunctionDef) -> List[VerificationIssue]:
        """Check return statements in a function"""
        issues = []
        
        func_name = node.name
        declared_return = ast.unparse(node.returns) if node.returns else None
        
        # Find all return statements
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        
        # Check if function has return type but no return statements
        if declared_return and declared_return != "None" and len(returns) == 0:
            issues.append(VerificationIssue(
                layer=VerificationLayer.TYPES,
                severity="medium",
                message=f"Function '{func_name}' declares return type '{declared_return}' but has no return statement",
                line_number=node.lineno
            ))
        
        # Check if function has return statements but no declared return type
        if not declared_return and len(returns) > 0:
            has_value_return = any(r.value is not None for r in returns)
            if has_value_return:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.TYPES,
                    severity="low",
                    message=f"Function '{func_name}' has return statements but no declared return type",
                    line_number=node.lineno,
                    suggestion="Add return type annotation to function"
                ))
        
        return issues
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.TYPES,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
