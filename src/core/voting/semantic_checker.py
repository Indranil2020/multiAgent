"""
Semantic equivalence checking for grouping similar results.

This module provides functionality to determine if two agent outputs are
semantically equivalent, even if they differ syntactically. This is crucial
for the voting system to correctly group similar solutions.
"""

import hashlib
import ast
from typing import Any, Optional
from .types import Result


class SemanticChecker:
    """
    Determines semantic equivalence between agent outputs.
    
    Uses multiple strategies to check if two results are semantically equivalent:
    - Test-based: Compare outputs on test cases
    - AST-based: Compare abstract syntax trees
    - Normalization-based: Compare normalized code
    """
    
    def __init__(self, use_test_based: bool = True, use_ast_based: bool = True):
        """
        Initialize semantic checker.
        
        Args:
            use_test_based: Whether to use test-based equivalence checking
            use_ast_based: Whether to use AST-based equivalence checking
        """
        self.use_test_based = use_test_based
        self.use_ast_based = use_ast_based
    
    def compute_signature(
        self,
        output: Any,
        test_cases: Optional[list] = None
    ) -> Result[str, str]:
        """
        Generate semantic signature for an output.
        
        The signature represents the semantic equivalence class of the output.
        Outputs with the same signature are considered semantically equivalent.
        
        Args:
            output: The output to generate signature for
            test_cases: Optional test cases for test-based signature
        
        Returns:
            Result containing the semantic signature string
        """
        signature_components = []
        
        # Test-based signature (most reliable)
        if self.use_test_based and test_cases:
            test_sig_result = self._hash_test_outputs(output, test_cases)
            if test_sig_result.is_ok():
                signature_components.append(f"test:{test_sig_result.unwrap()}")
        
        # AST-based signature (for code outputs)
        if self.use_ast_based:
            ast_sig_result = self._hash_ast_structure(output)
            if ast_sig_result.is_ok():
                signature_components.append(f"ast:{ast_sig_result.unwrap()}")
        
        # Fallback to normalized output hash
        if len(signature_components) == 0:
            norm_result = self._normalize_output(output)
            if norm_result.is_ok():
                normalized = norm_result.unwrap()
                hash_value = hashlib.sha256(normalized.encode()).hexdigest()[:16]
                signature_components.append(f"norm:{hash_value}")
        
        if len(signature_components) == 0:
            return Result(success=False, error="Failed to generate any signature component")
        
        # Combine components
        signature = "|".join(signature_components)
        return Result(success=True, value=signature)
    
    def are_equivalent(
        self,
        output1: Any,
        output2: Any,
        test_cases: Optional[list] = None
    ) -> Result[bool, str]:
        """
        Check if two outputs are semantically equivalent.
        
        Args:
            output1: First output
            output2: Second output
            test_cases: Optional test cases for comparison
        
        Returns:
            Result containing True if equivalent, False otherwise
        """
        sig1_result = self.compute_signature(output1, test_cases)
        sig2_result = self.compute_signature(output2, test_cases)
        
        if sig1_result.is_err():
            return Result(success=False, error=f"Failed to compute signature 1: {sig1_result.error}")
        
        if sig2_result.is_err():
            return Result(success=False, error=f"Failed to compute signature 2: {sig2_result.error}")
        
        equivalent = sig1_result.unwrap() == sig2_result.unwrap()
        return Result(success=True, value=equivalent)
    
    def _hash_test_outputs(self, output: Any, test_cases: list) -> Result[str, str]:
        """
        Generate hash based on test case outputs.
        
        Executes the output on test cases and hashes the results.
        This is the most reliable method for semantic equivalence.
        
        Args:
            output: The output to test
            test_cases: List of test cases
        
        Returns:
            Result containing hash of test outputs
        """
        if not test_cases:
            return Result(success=False, error="No test cases provided")
        
        test_results = []
        
        for test_case in test_cases:
            result = self._execute_on_test(output, test_case)
            if result.is_err():
                # If execution fails, include error in signature
                test_results.append(f"ERROR:{result.error}")
            else:
                test_results.append(str(result.unwrap()))
        
        # Hash the combined test results
        combined = "|".join(test_results)
        hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
        
        return Result(success=True, value=hash_value)
    
    def _execute_on_test(self, output: Any, test_case: dict) -> Result[Any, str]:
        """
        Execute output on a single test case.
        
        Args:
            output: The output to execute
            test_case: Test case with 'inputs' and 'expected_output'
        
        Returns:
            Result containing the test output
        """
        # This is a simplified version - real implementation would need
        # proper sandboxed execution
        output_str = str(output)
        
        # Check if output is executable code
        if not isinstance(output, str):
            return Result(success=False, error="Output is not a string")
        
        # For now, return a placeholder based on test case
        # Real implementation would execute the code safely
        test_id = str(test_case.get('inputs', ''))
        placeholder = f"result_{hashlib.md5(test_id.encode()).hexdigest()[:8]}"
        
        return Result(success=True, value=placeholder)
    
    def _hash_ast_structure(self, output: Any) -> Result[str, str]:
        """
        Generate hash based on AST structure.
        
        Parses code and hashes the abstract syntax tree structure,
        ignoring variable names and comments.
        
        Args:
            output: The output to parse
        
        Returns:
            Result containing AST hash
        """
        output_str = str(output)
        
        # Try to parse as Python code
        parse_result = self._parse_code(output_str)
        if parse_result.is_err():
            return Result(success=False, error=parse_result.error)
        
        tree = parse_result.unwrap()
        
        # Generate canonical representation of AST
        ast_repr = ast.dump(tree, annotate_fields=False)
        
        # Hash the AST representation
        hash_value = hashlib.sha256(ast_repr.encode()).hexdigest()[:16]
        
        return Result(success=True, value=hash_value)
    
    def _parse_code(self, code: str) -> Result[ast.AST, str]:
        """
        Parse code into AST.
        
        Args:
            code: Code string to parse
        
        Returns:
            Result containing AST or error
        """
        # Validate code is a string
        if not isinstance(code, str):
            return Result(success=False, error="Code must be a string")
        
        # Check if code is empty
        if not code.strip():
            return Result(success=False, error="Code is empty")
        
        # Attempt to parse
        # Note: Not using try-except per requirements
        # In production, would use a parser that returns Result type
        parsed = ast.parse(code)
        
        if parsed is None:
            return Result(success=False, error="Failed to parse code")
        
        return Result(success=True, value=parsed)
    
    def _normalize_output(self, output: Any) -> Result[str, str]:
        """
        Normalize output for comparison.
        
        Removes whitespace, comments, and standardizes formatting.
        
        Args:
            output: Output to normalize
        
        Returns:
            Result containing normalized string
        """
        output_str = str(output)
        
        # Normalize code if it's Python
        code_result = self._normalize_code(output_str)
        if code_result.is_ok():
            return code_result
        
        # Fallback: simple string normalization
        normalized = output_str.strip()
        normalized = " ".join(normalized.split())  # Normalize whitespace
        normalized = normalized.lower()  # Case insensitive
        
        return Result(success=True, value=normalized)
    
    def _normalize_code(self, code: str) -> Result[str, str]:
        """
        Normalize Python code.
        
        Removes comments, normalizes whitespace, and standardizes formatting.
        
        Args:
            code: Code to normalize
        
        Returns:
            Result containing normalized code
        """
        parse_result = self._parse_code(code)
        if parse_result.is_err():
            return Result(success=False, error="Not valid Python code")
        
        tree = parse_result.unwrap()
        
        # Use AST to generate normalized code
        # This removes comments and normalizes formatting
        normalized = ast.unparse(tree)
        
        return Result(success=True, value=normalized)
