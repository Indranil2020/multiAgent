"""
Unit Tester - Layer 4 of Verification Stack

This module provides unit test execution and validation.
It runs unit tests and validates that code passes all test cases.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import ast
import time

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationIssue
)


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    inputs: Dict[str, Any]
    expected_output: Any
    description: str = ""
    timeout_ms: int = 5000


@dataclass
class UnitTesterConfig:
    """Configuration for unit tester"""
    strict_mode: bool = True
    require_tests: bool = True
    minimum_coverage: float = 0.8
    timeout_ms: int = 5000
    fail_fast: bool = False


class TestResult:
    """Result of a single test execution"""
    
    def __init__(self, test_name: str, passed: bool, message: str = "", actual_output: Any = None):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.actual_output = actual_output
        self.execution_time_ms = 0.0


class UnitTester:
    """
    Executes and validates unit tests for code.
    
    This verifier runs unit tests and ensures code passes all test cases
    with proper coverage and correctness.
    """
    
    def __init__(self, config: Optional[UnitTesterConfig] = None):
        """
        Initialize the unit tester.
        
        Args:
            config: Configuration for the tester
        """
        self.config = config if config is not None else UnitTesterConfig()
        self.test_results: List[TestResult] = []
    
    def verify(
        self,
        code: str,
        test_cases: Optional[List[TestCase]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Verify code by running unit tests.
        
        Args:
            code: Source code to verify
            test_cases: List of test cases to run
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with unit test status
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
        
        # Set test cases
        test_cases = test_cases if test_cases is not None else []
        
        # Check if tests are required
        if self.config.require_tests and len(test_cases) == 0:
            return self._create_error_result(
                "No test cases provided but tests are required",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Parse code
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for unit testing",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract test functions from code
        code_tests = self._extract_tests_from_code(tree)
        
        # Run test cases
        issues = []
        self.test_results = []
        
        for test_case in test_cases:
            test_result = self._run_test_case(code, test_case)
            self.test_results.append(test_result)
            
            if not test_result.passed:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.UNIT_TESTS,
                    severity="high",
                    message=f"Test '{test_case.name}' failed: {test_result.message}",
                    suggestion=f"Expected: {test_case.expected_output}, Got: {test_result.actual_output}"
                ))
                
                if self.config.fail_fast:
                    break
        
        # Run code tests
        for test_func in code_tests:
            test_result = self._run_code_test(code, test_func)
            self.test_results.append(test_result)
            
            if not test_result.passed:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.UNIT_TESTS,
                    severity="high",
                    message=f"Test function '{test_func}' failed: {test_result.message}"
                ))
        
        # Calculate coverage
        coverage = self._calculate_coverage(tree, test_cases)
        
        if coverage < self.config.minimum_coverage:
            issues.append(VerificationIssue(
                layer=VerificationLayer.UNIT_TESTS,
                severity="medium",
                message=f"Test coverage {coverage:.1%} below minimum {self.config.minimum_coverage:.1%}",
                suggestion="Add more test cases to improve coverage"
            ))
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        passed_tests = len([r for r in self.test_results if r.passed])
        total_tests = len(self.test_results)
        
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.UNIT_TESTS,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"All {total_tests} tests passed",
                execution_time_ms=execution_time,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "coverage": coverage
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.UNIT_TESTS,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Unit testing failed: {passed_tests}/{total_tests} tests passed",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "coverage": coverage
                }
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
    
    def _extract_tests_from_code(self, tree: ast.AST) -> List[str]:
        """Extract test function names from code"""
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    test_functions.append(node.name)
        
        return test_functions
    
    def _run_test_case(self, code: str, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            code: Source code
            test_case: Test case to run
            
        Returns:
            TestResult with execution result
        """
        start_time = time.time()
        
        # Create execution environment
        exec_globals = {}
        
        # Execute code to define functions
        exec_result = self._safe_exec(code, exec_globals)
        
        if not exec_result["success"]:
            result = TestResult(
                test_name=test_case.name,
                passed=False,
                message=f"Failed to execute code: {exec_result['error']}"
            )
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Find the function to test
        test_function = self._find_test_function(exec_globals, test_case)
        
        if test_function is None:
            result = TestResult(
                test_name=test_case.name,
                passed=False,
                message="Test function not found in code"
            )
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Execute test
        call_result = self._safe_call(test_function, test_case.inputs)
        
        if not call_result["success"]:
            result = TestResult(
                test_name=test_case.name,
                passed=False,
                message=f"Test execution failed: {call_result['error']}"
            )
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        actual_output = call_result["result"]
        
        # Compare output
        if self._compare_outputs(actual_output, test_case.expected_output):
            result = TestResult(
                test_name=test_case.name,
                passed=True,
                message="Test passed",
                actual_output=actual_output
            )
        else:
            result = TestResult(
                test_name=test_case.name,
                passed=False,
                message=f"Output mismatch",
                actual_output=actual_output
            )
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _run_code_test(self, code: str, test_func_name: str) -> TestResult:
        """Run a test function defined in code"""
        start_time = time.time()
        
        # Execute code
        exec_globals = {}
        exec_result = self._safe_exec(code, exec_globals)
        
        if not exec_result["success"]:
            result = TestResult(
                test_name=test_func_name,
                passed=False,
                message=f"Failed to execute code: {exec_result['error']}"
            )
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Get test function
        if test_func_name not in exec_globals:
            result = TestResult(
                test_name=test_func_name,
                passed=False,
                message=f"Test function '{test_func_name}' not found"
            )
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result
        
        test_function = exec_globals[test_func_name]
        
        # Execute test function
        call_result = self._safe_call(test_function, {})
        
        if not call_result["success"]:
            result = TestResult(
                test_name=test_func_name,
                passed=False,
                message=f"Test failed: {call_result['error']}"
            )
        else:
            result = TestResult(
                test_name=test_func_name,
                passed=True,
                message="Test passed"
            )
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _safe_exec(self, code: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute code"""
        # Validate code before execution
        if not self._is_safe_code(code):
            return {
                "success": False,
                "error": "Code contains unsafe operations"
            }
        
        # Execute code
        result = {"success": True, "error": None}
        
        # Use exec with validation
        exec(code, globals_dict)
        
        return result
    
    def _safe_call(self, func: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Safely call a function"""
        result = {"success": True, "result": None, "error": None}
        
        # Call function
        output = func(**inputs)
        result["result"] = output
        
        return result
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if code is safe to execute"""
        # Check for dangerous operations
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False
        
        return True
    
    def _find_test_function(self, globals_dict: Dict[str, Any], test_case: TestCase) -> Optional[Callable]:
        """Find the function to test"""
        # Look for function in globals
        for name, obj in globals_dict.items():
            if callable(obj) and not name.startswith('_'):
                return obj
        
        return None
    
    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected outputs"""
        # Handle None
        if actual is None and expected is None:
            return True
        
        if actual is None or expected is None:
            return False
        
        # Handle different types
        if type(actual) != type(expected):
            # Try conversion
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                return abs(actual - expected) < 1e-9
            return False
        
        # Direct comparison
        return actual == expected
    
    def _calculate_coverage(self, tree: ast.AST, test_cases: List[TestCase]) -> float:
        """Calculate test coverage"""
        # Count functions in code
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if len(functions) == 0:
            return 1.0
        
        # Estimate coverage based on test count
        # This is simplified - real coverage would track execution
        coverage = min(1.0, len(test_cases) / max(1, len(functions)))
        
        return coverage
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.UNIT_TESTS,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
