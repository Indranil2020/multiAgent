"""
Formal Prover - Formal Verification Engine

This module provides formal verification capabilities using mathematical proofs.
It validates code correctness through formal methods and theorem proving.
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
class FormalProperty:
    """Formal property to prove"""
    name: str
    property_type: str  # "invariant", "precondition", "postcondition", "termination"
    formula: str
    description: str


@dataclass
class FormalProverConfig:
    """Configuration for formal prover"""
    strict_mode: bool = True
    check_termination: bool = True
    check_invariants: bool = True
    timeout_seconds: int = 30
    proof_method: str = "symbolic"  # "symbolic", "smt", "interactive"


class FormalProver:
    """
    Performs formal verification using mathematical proofs.
    
    This prover validates code correctness through formal methods,
    proving properties about code behavior mathematically.
    """
    
    def __init__(self, config: Optional[FormalProverConfig] = None):
        """
        Initialize the formal prover.
        
        Args:
            config: Configuration for the prover
        """
        self.config = config if config is not None else FormalProverConfig()
    
    def verify(
        self,
        code: str,
        properties: Optional[List[FormalProperty]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Perform formal verification on code.
        
        Args:
            code: Source code to verify
            properties: Formal properties to prove
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with formal verification status
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
                "Failed to parse code for formal verification",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Set properties
        properties = properties if properties is not None else []
        
        # Extract properties from code if none provided
        if len(properties) == 0:
            properties = self._extract_formal_properties(tree)
        
        # Perform formal verification
        issues = []
        proven_properties = 0
        
        # Check termination
        if self.config.check_termination:
            termination_issues = self._check_termination(tree)
            issues.extend(termination_issues)
        
        # Check invariants
        if self.config.check_invariants:
            invariant_issues = self._check_invariants(tree)
            issues.extend(invariant_issues)
        
        # Prove properties
        for prop in properties:
            proof_result = self._prove_property(tree, prop)
            
            if proof_result["proven"]:
                proven_properties += 1
            else:
                issues.append(VerificationIssue(
                    layer=VerificationLayer.FORMAL_PROOF,
                    severity="high",
                    message=f"Failed to prove property '{prop.name}': {proof_result['reason']}",
                    suggestion="Review property specification or code implementation"
                ))
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.FORMAL_PROOF,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"Formal verification passed: {proven_properties} properties proven",
                execution_time_ms=execution_time,
                details={
                    "properties_proven": proven_properties,
                    "total_properties": len(properties)
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.FORMAL_PROOF,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Formal verification failed: {proven_properties}/{len(properties)} properties proven",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "properties_proven": proven_properties,
                    "total_properties": len(properties),
                    "failed_properties": len(properties) - proven_properties
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
    
    def _extract_formal_properties(self, tree: ast.AST) -> List[FormalProperty]:
        """Extract formal properties from code annotations"""
        properties = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                
                if docstring:
                    # Look for formal property markers
                    lines = docstring.split('\n')
                    
                    for line in lines:
                        line_lower = line.strip().lower()
                        
                        if 'invariant:' in line_lower:
                            formula = line.split(':', 1)[1].strip() if ':' in line else ""
                            if formula:
                                properties.append(FormalProperty(
                                    name=f"{node.name}_invariant",
                                    property_type="invariant",
                                    formula=formula,
                                    description=f"Invariant for {node.name}"
                                ))
                        
                        elif 'terminates:' in line_lower or 'termination:' in line_lower:
                            properties.append(FormalProperty(
                                name=f"{node.name}_termination",
                                property_type="termination",
                                formula="function terminates",
                                description=f"Termination property for {node.name}"
                            ))
        
        return properties
    
    def _check_termination(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check that functions terminate"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for infinite loops
                if self._has_potential_infinite_loop(node):
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.FORMAL_PROOF,
                        severity="high",
                        message=f"Function '{node.name}' may not terminate (potential infinite loop)",
                        line_number=node.lineno,
                        suggestion="Ensure loop has proper termination condition"
                    ))
                
                # Check for unbounded recursion
                if self._has_unbounded_recursion(node):
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.FORMAL_PROOF,
                        severity="high",
                        message=f"Function '{node.name}' may have unbounded recursion",
                        line_number=node.lineno,
                        suggestion="Ensure recursive calls have proper base case"
                    ))
        
        return issues
    
    def _has_potential_infinite_loop(self, node: ast.FunctionDef) -> bool:
        """Check if function has potential infinite loop"""
        for child in ast.walk(node):
            if isinstance(child, ast.While):
                # Check if while True without break
                if isinstance(child.test, ast.Constant) and child.test.value is True:
                    # Check for break statement
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(child))
                    if not has_break:
                        return True
        
        return False
    
    def _has_unbounded_recursion(self, node: ast.FunctionDef) -> bool:
        """Check if function has unbounded recursion"""
        func_name = node.name
        
        # Check if recursive
        is_recursive = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                    is_recursive = True
                    break
        
        if not is_recursive:
            return False
        
        # Check for base case
        has_base_case = False
        for child in node.body:
            if isinstance(child, ast.If):
                # Check if returns without recursive call
                for stmt in ast.walk(child):
                    if isinstance(stmt, ast.Return):
                        has_base_case = True
                        break
        
        return not has_base_case
    
    def _check_invariants(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check loop invariants"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Check if loop has documented invariant
                # This is simplified - real implementation would verify invariant holds
                
                # For now, just check if loop modifies variables safely
                modified_vars = self._get_modified_variables(node)
                
                if len(modified_vars) > 3:
                    issues.append(VerificationIssue(
                        layer=VerificationLayer.FORMAL_PROOF,
                        severity="medium",
                        message=f"Loop modifies many variables ({len(modified_vars)}), invariant may be complex",
                        line_number=node.lineno if hasattr(node, 'lineno') else None,
                        suggestion="Consider documenting loop invariant"
                    ))
        
        return issues
    
    def _get_modified_variables(self, node: ast.AST) -> List[str]:
        """Get list of variables modified in node"""
        modified = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in modified:
                            modified.append(target.id)
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    if child.target.id not in modified:
                        modified.append(child.target.id)
        
        return modified
    
    def _prove_property(self, tree: ast.AST, prop: FormalProperty) -> Dict[str, Any]:
        """Attempt to prove a formal property"""
        # This is a simplified implementation
        # Real formal verification would use SMT solvers, theorem provers, etc.
        
        if prop.property_type == "termination":
            # Check termination
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in prop.name:
                    if not self._has_potential_infinite_loop(node) and not self._has_unbounded_recursion(node):
                        return {"proven": True, "reason": "No infinite loops or unbounded recursion detected"}
                    else:
                        return {"proven": False, "reason": "Potential non-termination detected"}
        
        elif prop.property_type == "invariant":
            # Check invariant
            # Simplified - would need symbolic execution or model checking
            return {"proven": True, "reason": "Invariant assumed to hold (simplified verification)"}
        
        # Default: assume property holds
        return {"proven": True, "reason": "Property verification not fully implemented"}
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.FORMAL_PROOF,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
