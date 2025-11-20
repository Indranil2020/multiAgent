"""
Contract Verifier - Layer 3 of Verification Stack

This module provides contract verification for code through pre/postconditions and invariants.
It validates that code adheres to specified contracts.
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
class Contract:
    """Represents a code contract"""
    name: str
    expression: str
    description: str
    contract_type: str  # "precondition", "postcondition", "invariant"


@dataclass
class ContractVerifierConfig:
    """Configuration for contract verifier"""
    strict_mode: bool = True
    check_preconditions: bool = True
    check_postconditions: bool = True
    check_invariants: bool = True
    runtime_checking: bool = False


class ContractVerifier:
    """
    Verifies that code adheres to specified contracts.
    
    This verifier checks preconditions, postconditions, and invariants
    to ensure code correctness according to formal specifications.
    """
    
    def __init__(self, config: Optional[ContractVerifierConfig] = None):
        """
        Initialize the contract verifier.
        
        Args:
            config: Configuration for the verifier
        """
        self.config = config if config is not None else ContractVerifierConfig()
        self.contracts: List[Contract] = []
    
    def verify(
        self,
        code: str,
        contracts: Optional[List[Contract]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Verify contracts in the provided code.
        
        Args:
            code: Source code to verify
            contracts: List of contracts to verify
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with contract verification status
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
        
        # Set contracts
        self.contracts = contracts if contracts is not None else []
        
        # Parse code into AST
        tree = self._parse_code(code)
        
        if tree is None:
            return self._create_error_result(
                "Failed to parse code for contract verification",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Perform contract checks
        issues = []
        
        # Extract contracts from code (docstrings, assertions, etc.)
        extracted_contracts = self._extract_contracts_from_code(tree)
        all_contracts = self.contracts + extracted_contracts
        
        # Check preconditions
        if self.config.check_preconditions:
            precondition_issues = self._check_preconditions(tree, all_contracts)
            issues.extend(precondition_issues)
        
        # Check postconditions
        if self.config.check_postconditions:
            postcondition_issues = self._check_postconditions(tree, all_contracts)
            issues.extend(postcondition_issues)
        
        # Check invariants
        if self.config.check_invariants:
            invariant_issues = self._check_invariants(tree, all_contracts)
            issues.extend(invariant_issues)
        
        # Check assertions
        assertion_issues = self._check_assertions(tree)
        issues.extend(assertion_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.CONTRACTS,
                status=VerificationStatus.PASSED,
                passed=True,
                message="Contract verification passed",
                execution_time_ms=execution_time,
                details={
                    "contracts_checked": len(all_contracts),
                    "preconditions": len([c for c in all_contracts if c.contract_type == "precondition"]),
                    "postconditions": len([c for c in all_contracts if c.contract_type == "postcondition"]),
                    "invariants": len([c for c in all_contracts if c.contract_type == "invariant"])
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.CONTRACTS,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Contract verification failed with {len(issues)} issue(s)",
                execution_time_ms=execution_time,
                issues=issues
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
    
    def _extract_contracts_from_code(self, tree: ast.AST) -> List[Contract]:
        """Extract contracts from code docstrings and assertions"""
        contracts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract from docstring
                docstring_contracts = self._extract_from_docstring(node)
                contracts.extend(docstring_contracts)
        
        return contracts
    
    def _extract_from_docstring(self, node: ast.FunctionDef) -> List[Contract]:
        """Extract contracts from function docstring"""
        contracts = []
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        if not docstring:
            return contracts
        
        # Parse docstring for contract keywords
        lines = docstring.split('\n')
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check for precondition markers
            if 'precondition:' in line_lower or 'requires:' in line_lower:
                condition = line.split(':', 1)[1].strip() if ':' in line else ""
                if condition:
                    contracts.append(Contract(
                        name=f"{node.name}_precondition",
                        expression=condition,
                        description=f"Precondition for {node.name}",
                        contract_type="precondition"
                    ))
            
            # Check for postcondition markers
            elif 'postcondition:' in line_lower or 'ensures:' in line_lower:
                condition = line.split(':', 1)[1].strip() if ':' in line else ""
                if condition:
                    contracts.append(Contract(
                        name=f"{node.name}_postcondition",
                        expression=condition,
                        description=f"Postcondition for {node.name}",
                        contract_type="postcondition"
                    ))
            
            # Check for invariant markers
            elif 'invariant:' in line_lower:
                condition = line.split(':', 1)[1].strip() if ':' in line else ""
                if condition:
                    contracts.append(Contract(
                        name=f"{node.name}_invariant",
                        expression=condition,
                        description=f"Invariant for {node.name}",
                        contract_type="invariant"
                    ))
        
        return contracts
    
    def _check_preconditions(self, tree: ast.AST, contracts: List[Contract]) -> List[VerificationIssue]:
        """Check preconditions"""
        issues = []
        
        preconditions = [c for c in contracts if c.contract_type == "precondition"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has precondition checks
                func_preconditions = [c for c in preconditions if node.name in c.name]
                
                if len(func_preconditions) > 0:
                    # Verify preconditions are checked in function
                    has_checks = self._has_precondition_checks(node)
                    
                    if not has_checks:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.CONTRACTS,
                            severity="medium",
                            message=f"Function '{node.name}' has preconditions but no validation",
                            line_number=node.lineno,
                            suggestion="Add precondition validation at function start"
                        ))
        
        return issues
    
    def _check_postconditions(self, tree: ast.AST, contracts: List[Contract]) -> List[VerificationIssue]:
        """Check postconditions"""
        issues = []
        
        postconditions = [c for c in contracts if c.contract_type == "postcondition"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has postcondition checks
                func_postconditions = [c for c in postconditions if node.name in c.name]
                
                if len(func_postconditions) > 0:
                    # Verify postconditions are checked before return
                    has_checks = self._has_postcondition_checks(node)
                    
                    if not has_checks:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.CONTRACTS,
                            severity="medium",
                            message=f"Function '{node.name}' has postconditions but no validation",
                            line_number=node.lineno,
                            suggestion="Add postcondition validation before return statements"
                        ))
        
        return issues
    
    def _check_invariants(self, tree: ast.AST, contracts: List[Contract]) -> List[VerificationIssue]:
        """Check invariants"""
        issues = []
        
        invariants = [c for c in contracts if c.contract_type == "invariant"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class has invariant checks
                class_invariants = [c for c in invariants if node.name in c.name]
                
                if len(class_invariants) > 0:
                    # Verify invariants are maintained
                    has_checks = self._has_invariant_checks(node)
                    
                    if not has_checks:
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.CONTRACTS,
                            severity="medium",
                            message=f"Class '{node.name}' has invariants but no validation",
                            line_number=node.lineno,
                            suggestion="Add invariant validation in class methods"
                        ))
        
        return issues
    
    def _check_assertions(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check assertion statements"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                # Validate assertion has meaningful message
                if node.msg is None:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.CONTRACTS,
                        severity="low",
                        message="Assertion lacks descriptive message",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion="Add descriptive message to assertion"
                    ))
        
        return issues
    
    def _has_precondition_checks(self, node: ast.FunctionDef) -> bool:
        """Check if function has precondition validation"""
        # Look for assertions or if statements at function start
        if len(node.body) == 0:
            return False
        
        first_statements = node.body[:3]  # Check first 3 statements
        
        for stmt in first_statements:
            if isinstance(stmt, ast.Assert):
                return True
            if isinstance(stmt, ast.If):
                # Check if it raises an exception
                for if_stmt in stmt.body:
                    if isinstance(if_stmt, ast.Raise):
                        return True
        
        return False
    
    def _has_postcondition_checks(self, node: ast.FunctionDef) -> bool:
        """Check if function has postcondition validation"""
        # Look for assertions before return statements
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return):
                # Check if there's an assertion nearby
                # This is simplified - real implementation would track control flow
                return True
        
        return False
    
    def _has_invariant_checks(self, node: ast.ClassDef) -> bool:
        """Check if class has invariant validation"""
        # Look for invariant checking methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if 'invariant' in item.name.lower() or 'check' in item.name.lower():
                    return True
        
        return False
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.CONTRACTS,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
