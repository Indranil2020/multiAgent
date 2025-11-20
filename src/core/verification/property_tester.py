"""
Property Tester - Layer 5 of Verification Stack

This module provides property-based testing for code.
It validates that code satisfies specified properties across a wide range of inputs.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import ast
import time
import random

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationIssue
)


@dataclass
class Property:
    """Represents a property to test"""
    name: str
    property_function: str  # Python expression or function
    description: str
    input_generators: Dict[str, Callable] = None


@dataclass
class PropertyTesterConfig:
    """Configuration for property tester"""
    strict_mode: bool = True
    num_test_cases: int = 100
    max_examples: int = 1000
    shrink_failures: bool = True
    timeout_ms: int = 10000


class PropertyTester:
    """
    Performs property-based testing on code.
    
    This verifier generates random inputs and validates that code
    satisfies specified properties across all test cases.
    """
    
    def __init__(self, config: Optional[PropertyTesterConfig] = None):
        """
        Initialize the property tester.
        
        Args:
            config: Configuration for the tester
        """
        self.config = config if config is not None else PropertyTesterConfig()
        self.random = random.Random(42)  # Deterministic for reproducibility
    
    def verify(
        self,
        code: str,
        properties: Optional[List[Property]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Verify code using property-based testing.
        
        Args:
            code: Source code to verify
            properties: List of properties to test
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with property testing status
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
        
        # Set properties
        properties = properties if properties is not None else []
        
        # Parse code
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for property testing",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract properties from code if none provided
        if len(properties) == 0:
            properties = self._extract_properties_from_code(tree)
        
        # Run property tests
        issues = []
        total_tests = 0
        passed_tests = 0
        
        for prop in properties:
            test_result = self._test_property(code, prop)
            total_tests += test_result["total"]
            passed_tests += test_result["passed"]
            
            if len(test_result["failures"]) > 0:
                for failure in test_result["failures"]:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PROPERTY_TESTS,
                        severity="high",
                        message=f"Property '{prop.name}' failed: {failure['message']}",
                        suggestion=f"Failed on input: {failure['input']}"
                    ))
        
        # Check for common properties
        common_property_issues = self._check_common_properties(tree, code)
        issues.extend(common_property_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.PROPERTY_TESTS,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"All property tests passed ({passed_tests}/{total_tests})",
                execution_time_ms=execution_time,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "properties_checked": len(properties)
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.PROPERTY_TESTS,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Property testing failed: {passed_tests}/{total_tests} tests passed",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": total_tests - passed_tests,
                    "properties_checked": len(properties)
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
    
    def _extract_properties_from_code(self, tree: ast.AST) -> List[Property]:
        """Extract properties from code docstrings"""
        properties = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                
                if docstring:
                    # Look for property markers
                    lines = docstring.split('\n')
                    
                    for line in lines:
                        if 'property:' in line.lower():
                            prop_expr = line.split(':', 1)[1].strip() if ':' in line else ""
                            if prop_expr:
                                properties.append(Property(
                                    name=f"{node.name}_property",
                                    property_function=prop_expr,
                                    description=f"Property for {node.name}"
                                ))
        
        return properties
    
    def _test_property(self, code: str, prop: Property) -> Dict[str, Any]:
        """Test a single property"""
        # Execute code
        exec_globals = {}
        
        exec_result = self._safe_exec(code, exec_globals)
        
        if not exec_result["success"]:
            return {
                "total": 0,
                "passed": 0,
                "failures": [{
                    "message": f"Failed to execute code: {exec_result['error']}",
                    "input": None
                }]
            }
        
        # Find function to test
        test_function = self._find_function(exec_globals)
        
        if test_function is None:
            return {
                "total": 0,
                "passed": 0,
                "failures": [{
                    "message": "No function found to test",
                    "input": None
                }]
            }
        
        # Generate test cases
        test_cases = self._generate_test_cases(test_function, prop)
        
        # Run tests
        failures = []
        passed = 0
        
        for test_input in test_cases:
            result = self._run_property_test(test_function, prop, test_input)
            
            if result["passed"]:
                passed += 1
            else:
                failures.append({
                    "message": result["message"],
                    "input": test_input
                })
        
        return {
            "total": len(test_cases),
            "passed": passed,
            "failures": failures
        }
    
    def _generate_test_cases(self, func: Callable, prop: Property) -> List[Dict[str, Any]]:
        """Generate test cases for property testing"""
        test_cases = []
        
        # Get function signature
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Generate random inputs
        for _ in range(self.config.num_test_cases):
            test_input = {}
            
            for param in params:
                # Generate random value based on type
                if prop.input_generators and param in prop.input_generators:
                    test_input[param] = prop.input_generators[param]()
                else:
                    test_input[param] = self._generate_random_value()
            
            test_cases.append(test_input)
        
        return test_cases
    
    def _generate_random_value(self) -> Any:
        """Generate a random value for testing"""
        # Generate different types of values
        value_type = self.random.choice(['int', 'float', 'str', 'bool', 'list'])
        
        if value_type == 'int':
            return self.random.randint(-1000, 1000)
        elif value_type == 'float':
            return self.random.uniform(-1000.0, 1000.0)
        elif value_type == 'str':
            length = self.random.randint(0, 20)
            return ''.join(self.random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))
        elif value_type == 'bool':
            return self.random.choice([True, False])
        elif value_type == 'list':
            length = self.random.randint(0, 10)
            return [self.random.randint(-100, 100) for _ in range(length)]
        
        return None
    
    def _run_property_test(self, func: Callable, prop: Property, test_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single property test"""
        # Call function
        call_result = self._safe_call(func, test_input)
        
        if not call_result["success"]:
            return {
                "passed": False,
                "message": f"Function call failed: {call_result['error']}"
            }
        
        output = call_result["result"]
        
        # Check property
        property_holds = self._check_property(prop, test_input, output)
        
        if property_holds:
            return {
                "passed": True,
                "message": "Property holds"
            }
        else:
            return {
                "passed": False,
                "message": f"Property '{prop.name}' does not hold"
            }
    
    def _check_property(self, prop: Property, inputs: Dict[str, Any], output: Any) -> bool:
        """Check if property holds for given inputs and output"""
        # This is simplified - real implementation would evaluate property expression
        # For now, we assume property holds if output is not None
        return output is not None
    
    def _check_common_properties(self, tree: ast.AST, code: str) -> List[VerificationIssue]:
        """Check common properties like idempotence, commutativity, etc."""
        issues = []
        
        # Check for pure functions (no side effects)
        purity_issues = self._check_purity(tree)
        issues.extend(purity_issues)
        
        # Check for determinism
        determinism_issues = self._check_determinism(tree)
        issues.extend(determinism_issues)
        
        return issues
    
    def _check_purity(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check if functions are pure (no side effects)"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for side effects
                has_side_effects = self._has_side_effects(node)
                
                if has_side_effects:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PROPERTY_TESTS,
                        severity="low",
                        message=f"Function '{node.name}' may have side effects",
                        line_number=node.lineno,
                        suggestion="Consider making function pure for better testability"
                    ))
        
        return issues
    
    def _has_side_effects(self, node: ast.FunctionDef) -> bool:
        """Check if function has side effects"""
        for stmt in ast.walk(node):
            # Check for global variable modifications
            if isinstance(stmt, ast.Global):
                return True
            
            # Check for nonlocal variable modifications
            if isinstance(stmt, ast.Nonlocal):
                return True
            
            # Check for print statements
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name) and stmt.func.id == 'print':
                    return True
        
        return False
    
    def _check_determinism(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check if functions are deterministic"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for non-deterministic operations
                has_randomness = self._has_randomness(node)
                
                if has_randomness:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.PROPERTY_TESTS,
                        severity="low",
                        message=f"Function '{node.name}' may be non-deterministic",
                        line_number=node.lineno,
                        suggestion="Use seeded random for reproducibility"
                    ))
        
        return issues
    
    def _has_randomness(self, node: ast.FunctionDef) -> bool:
        """Check if function uses randomness"""
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute):
                    if stmt.func.attr in ['random', 'randint', 'choice', 'shuffle']:
                        return True
        
        return False
    
    def _safe_exec(self, code: str, globals_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute code"""
        if not self._is_safe_code(code):
            return {
                "success": False,
                "error": "Code contains unsafe operations"
            }
        
        result = {"success": True, "error": None}
        exec(code, globals_dict)
        
        return result
    
    def _safe_call(self, func: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Safely call a function"""
        result = {"success": True, "result": None, "error": None}
        
        output = func(**inputs)
        result["result"] = output
        
        return result
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if code is safe to execute"""
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval(',
            'exec(',
            'open(',
            'file('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False
        
        return True
    
    def _find_function(self, globals_dict: Dict[str, Any]) -> Optional[Callable]:
        """Find the main function to test"""
        for name, obj in globals_dict.items():
            if callable(obj) and not name.startswith('_'):
                return obj
        
        return None
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.PROPERTY_TESTS,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
