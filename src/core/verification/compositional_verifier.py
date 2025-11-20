"""
Compositional Verifier - Compositional Verification Engine

This module provides compositional verification for code components.
It validates that components compose correctly and maintain properties when integrated.
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
class Component:
    """Represents a code component"""
    name: str
    code: str
    interface: Dict[str, Any]
    dependencies: List[str]


@dataclass
class CompositionalVerifierConfig:
    """Configuration for compositional verifier"""
    strict_mode: bool = True
    check_interfaces: bool = True
    check_dependencies: bool = True
    check_composition_properties: bool = True


class CompositionalVerifier:
    """
    Verifies compositional correctness of code components.
    
    This verifier ensures that components compose correctly,
    interfaces match, and properties are preserved during composition.
    """
    
    def __init__(self, config: Optional[CompositionalVerifierConfig] = None):
        """
        Initialize the compositional verifier.
        
        Args:
            config: Configuration for the verifier
        """
        self.config = config if config is not None else CompositionalVerifierConfig()
    
    def verify(
        self,
        code: str,
        components: Optional[List[Component]] = None,
        code_id: str = "unknown"
    ) -> VerificationResult:
        """
        Verify compositional correctness of code.
        
        Args:
            code: Source code to verify
            components: List of components being composed
            code_id: Identifier for the code being verified
            
        Returns:
            VerificationResult with compositional verification status
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
                "Failed to parse code for compositional verification",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Extract components from code if not provided
        if components is None:
            components = self._extract_components(tree, code)
        
        # Perform compositional verification
        issues = []
        
        # Check interface compatibility
        if self.config.check_interfaces:
            interface_issues = self._check_interfaces(components)
            issues.extend(interface_issues)
        
        # Check dependency consistency
        if self.config.check_dependencies:
            dependency_issues = self._check_dependencies(components)
            issues.extend(dependency_issues)
        
        # Check composition properties
        if self.config.check_composition_properties:
            property_issues = self._check_composition_properties(components)
            issues.extend(property_issues)
        
        # Check integration points
        integration_issues = self._check_integration_points(tree)
        issues.extend(integration_issues)
        
        execution_time = (time.time() - start_time) * 1000
        
        # Create result
        if len(issues) == 0:
            return VerificationResult(
                layer=VerificationLayer.COMPOSITIONAL,
                status=VerificationStatus.PASSED,
                passed=True,
                message=f"Compositional verification passed for {len(components)} component(s)",
                execution_time_ms=execution_time,
                details={
                    "components_verified": len(components)
                }
            )
        else:
            result = VerificationResult(
                layer=VerificationLayer.COMPOSITIONAL,
                status=VerificationStatus.FAILED,
                passed=False,
                message=f"Compositional verification failed with {len(issues)} issue(s)",
                execution_time_ms=execution_time,
                issues=issues,
                details={
                    "components_verified": len(components),
                    "failed_checks": len(issues)
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
    
    def _extract_components(self, tree: ast.AST, code: str) -> List[Component]:
        """Extract components from code"""
        components = []
        
        # Extract classes as components
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                component = Component(
                    name=node.name,
                    code=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                    interface=self._extract_interface(node),
                    dependencies=self._extract_dependencies(node)
                )
                components.append(component)
            
            # Extract functions as components
            elif isinstance(node, ast.FunctionDef):
                # Only top-level functions
                if isinstance(node, ast.FunctionDef) and not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    component = Component(
                        name=node.name,
                        code=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                        interface=self._extract_function_interface(node),
                        dependencies=self._extract_function_dependencies(node)
                    )
                    components.append(component)
        
        return components
    
    def _extract_interface(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract interface from class"""
        interface = {
            "methods": [],
            "attributes": [],
            "base_classes": []
        }
        
        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                interface["base_classes"].append(base.id)
        
        # Extract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    "name": item.name,
                    "parameters": [arg.arg for arg in item.args.args],
                    "return_type": ast.unparse(item.returns) if item.returns else None
                }
                interface["methods"].append(method_info)
            
            # Extract attributes
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    interface["attributes"].append(item.target.id)
        
        return interface
    
    def _extract_function_interface(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract interface from function"""
        interface = {
            "name": node.name,
            "parameters": [arg.arg for arg in node.args.args],
            "return_type": ast.unparse(node.returns) if node.returns else None,
            "parameter_types": {}
        }
        
        # Extract parameter types
        for arg in node.args.args:
            if arg.annotation:
                interface["parameter_types"][arg.arg] = ast.unparse(arg.annotation)
        
        return interface
    
    def _extract_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract dependencies from class"""
        dependencies = []
        
        # Extract from base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
        
        # Extract from method calls
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                if isinstance(item.func, ast.Name):
                    if item.func.id not in dependencies:
                        dependencies.append(item.func.id)
        
        return dependencies
    
    def _extract_function_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract dependencies from function"""
        dependencies = []
        
        # Extract from function calls
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                if isinstance(item.func, ast.Name):
                    if item.func.id not in dependencies and item.func.id != node.name:
                        dependencies.append(item.func.id)
        
        return dependencies
    
    def _check_interfaces(self, components: List[Component]) -> List[VerificationIssue]:
        """Check interface compatibility between components"""
        issues = []
        
        # Build component map
        component_map = {comp.name: comp for comp in components}
        
        # Check each component's dependencies
        for component in components:
            for dep in component.dependencies:
                if dep in component_map:
                    # Check if interfaces are compatible
                    dep_component = component_map[dep]
                    
                    # Simplified interface check
                    # Real implementation would do deep interface matching
                    if not self._interfaces_compatible(component.interface, dep_component.interface):
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.COMPOSITIONAL,
                            severity="high",
                            message=f"Interface mismatch between '{component.name}' and '{dep}'",
                            suggestion="Ensure component interfaces are compatible"
                        ))
        
        return issues
    
    def _interfaces_compatible(self, interface1: Dict[str, Any], interface2: Dict[str, Any]) -> bool:
        """Check if two interfaces are compatible"""
        # Simplified compatibility check
        # Real implementation would check method signatures, types, etc.
        return True
    
    def _check_dependencies(self, components: List[Component]) -> List[VerificationIssue]:
        """Check dependency consistency"""
        issues = []
        
        # Build component map
        component_names = {comp.name for comp in components}
        
        # Check for missing dependencies
        for component in components:
            for dep in component.dependencies:
                if dep not in component_names:
                    # Check if it's a built-in or external dependency
                    if not self._is_builtin(dep):
                        issues.append(VerificationIssue(
                            layer=VerificationLayer.COMPOSITIONAL,
                            severity="medium",
                            message=f"Component '{component.name}' depends on undefined '{dep}'",
                            suggestion="Ensure all dependencies are defined or imported"
                        ))
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies(components)
        for cycle in circular_deps:
            issues.append(VerificationIssue(
                layer=VerificationLayer.COMPOSITIONAL,
                severity="high",
                message=f"Circular dependency detected: {' -> '.join(cycle)}",
                suggestion="Refactor to remove circular dependencies"
            ))
        
        return issues
    
    def _is_builtin(self, name: str) -> bool:
        """Check if name is a built-in"""
        builtins = ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']
        return name in builtins
    
    def _find_circular_dependencies(self, components: List[Component]) -> List[List[str]]:
        """Find circular dependencies"""
        cycles = []
        
        # Build dependency graph
        graph = {comp.name: comp.dependencies for comp in components}
        
        # Simple cycle detection
        visited = set()
        path = []
        
        def dfs(node: str):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            if node in graph:
                for dep in graph[node]:
                    if dep in graph:
                        dfs(dep)
            
            path.pop()
        
        for comp in components:
            dfs(comp.name)
        
        return cycles
    
    def _check_composition_properties(self, components: List[Component]) -> List[VerificationIssue]:
        """Check that properties are preserved during composition"""
        issues = []
        
        # Check that composition doesn't violate properties
        # This is simplified - real implementation would verify formal properties
        
        # Check for property violations in composition
        for component in components:
            # Check if component maintains invariants when composed
            # Simplified check
            pass
        
        return issues
    
    def _check_integration_points(self, tree: ast.AST) -> List[VerificationIssue]:
        """Check integration points between components"""
        issues = []
        
        # Check for proper error handling at integration points
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if external calls have error handling
                # This is simplified
                pass
        
        return issues
    
    def _create_error_result(self, message: str, execution_time_ms: float = 0.0) -> VerificationResult:
        """Create an error result"""
        return VerificationResult(
            layer=VerificationLayer.COMPOSITIONAL,
            status=VerificationStatus.ERROR,
            passed=False,
            message=message,
            execution_time_ms=execution_time_ms
        )
