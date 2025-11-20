"""
Verification Stack - Main Orchestrator for 8-Layer Verification System

This module orchestrates all verification layers and provides a unified interface
for comprehensive code verification following the zero-error architecture.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time

from . import (
    VerificationLayer,
    VerificationStatus,
    VerificationResult,
    VerificationReport
)

from .syntax_verifier import SyntaxVerifier, SyntaxVerifierConfig
from .type_checker import TypeChecker, TypeCheckerConfig
from .contract_verifier import ContractVerifier, ContractVerifierConfig, Contract
from .unit_tester import UnitTester, UnitTesterConfig, TestCase
from .property_tester import PropertyTester, PropertyTesterConfig, Property
from .static_analyzer import StaticAnalyzer, StaticAnalyzerConfig
from .security_scanner import SecurityScanner, SecurityScannerConfig
from .performance_checker import PerformanceChecker, PerformanceCheckerConfig, PerformanceRequirement
from .formal_prover import FormalProver, FormalProverConfig, FormalProperty
from .compositional_verifier import CompositionalVerifier, CompositionalVerifierConfig, Component


@dataclass
class VerificationStackConfig:
    """Configuration for the verification stack"""
    # Layer enable/disable flags
    enable_syntax: bool = True
    enable_types: bool = True
    enable_contracts: bool = True
    enable_unit_tests: bool = True
    enable_property_tests: bool = True
    enable_static_analysis: bool = True
    enable_security: bool = True
    enable_performance: bool = True
    enable_formal_proof: bool = False  # Optional, computationally expensive
    enable_compositional: bool = False  # Optional, for multi-component systems
    
    # Fail-fast configuration
    fail_fast: bool = False
    stop_on_critical: bool = True
    
    # Individual layer configs
    syntax_config: Optional[SyntaxVerifierConfig] = None
    type_config: Optional[TypeCheckerConfig] = None
    contract_config: Optional[ContractVerifierConfig] = None
    unit_test_config: Optional[UnitTesterConfig] = None
    property_test_config: Optional[PropertyTesterConfig] = None
    static_analysis_config: Optional[StaticAnalyzerConfig] = None
    security_config: Optional[SecurityScannerConfig] = None
    performance_config: Optional[PerformanceCheckerConfig] = None
    formal_proof_config: Optional[FormalProverConfig] = None
    compositional_config: Optional[CompositionalVerifierConfig] = None


@dataclass
class VerificationInput:
    """Input for verification"""
    code: str
    code_id: str = "unknown"
    
    # Optional inputs for specific layers
    contracts: Optional[List[Contract]] = None
    test_cases: Optional[List[TestCase]] = None
    properties: Optional[List[Property]] = None
    performance_requirements: Optional[List[PerformanceRequirement]] = None
    formal_properties: Optional[List[FormalProperty]] = None
    components: Optional[List[Component]] = None


class VerificationStack:
    """
    Main orchestrator for the 8-layer verification system.
    
    This class coordinates all verification layers and provides a unified
    interface for comprehensive code verification.
    
    Layers (in order):
        1. Syntax Verification
        2. Type Checking
        3. Contract Verification
        4. Unit Testing
        5. Property-Based Testing
        6. Static Analysis
        7. Security Scanning
        8. Performance Checking
        
    Optional Layers:
        - Formal Proof
        - Compositional Verification
    """
    
    def __init__(self, config: Optional[VerificationStackConfig] = None):
        """
        Initialize the verification stack.
        
        Args:
            config: Configuration for the stack
        """
        self.config = config if config is not None else VerificationStackConfig()
        
        # Initialize verifiers
        self.syntax_verifier = SyntaxVerifier(
            self.config.syntax_config if self.config.syntax_config else SyntaxVerifierConfig()
        )
        
        self.type_checker = TypeChecker(
            self.config.type_config if self.config.type_config else TypeCheckerConfig()
        )
        
        self.contract_verifier = ContractVerifier(
            self.config.contract_config if self.config.contract_config else ContractVerifierConfig()
        )
        
        self.unit_tester = UnitTester(
            self.config.unit_test_config if self.config.unit_test_config else UnitTesterConfig()
        )
        
        self.property_tester = PropertyTester(
            self.config.property_test_config if self.config.property_test_config else PropertyTesterConfig()
        )
        
        self.static_analyzer = StaticAnalyzer(
            self.config.static_analysis_config if self.config.static_analysis_config else StaticAnalyzerConfig()
        )
        
        self.security_scanner = SecurityScanner(
            self.config.security_config if self.config.security_config else SecurityScannerConfig()
        )
        
        self.performance_checker = PerformanceChecker(
            self.config.performance_config if self.config.performance_config else PerformanceCheckerConfig()
        )
        
        self.formal_prover = FormalProver(
            self.config.formal_proof_config if self.config.formal_proof_config else FormalProverConfig()
        )
        
        self.compositional_verifier = CompositionalVerifier(
            self.config.compositional_config if self.config.compositional_config else CompositionalVerifierConfig()
        )
    
    def verify(self, verification_input: VerificationInput) -> VerificationReport:
        """
        Run complete verification stack on code.
        
        Args:
            verification_input: Input containing code and verification parameters
            
        Returns:
            VerificationReport with results from all layers
        """
        start_time = time.time()
        
        # Create report
        report = VerificationReport(
            code_id=verification_input.code_id,
            overall_passed=True
        )
        
        # Layer 1: Syntax Verification
        if self.config.enable_syntax:
            syntax_result = self._run_layer(
                "Syntax Verification",
                lambda: self.syntax_verifier.verify(
                    verification_input.code,
                    verification_input.code_id
                )
            )
            report.add_result(syntax_result)
            
            if self._should_stop(syntax_result):
                report.metadata["stopped_at"] = "syntax"
                return report
        
        # Layer 2: Type Checking
        if self.config.enable_types:
            type_result = self._run_layer(
                "Type Checking",
                lambda: self.type_checker.verify(
                    verification_input.code,
                    verification_input.code_id
                )
            )
            report.add_result(type_result)
            
            if self._should_stop(type_result):
                report.metadata["stopped_at"] = "types"
                return report
        
        # Layer 3: Contract Verification
        if self.config.enable_contracts:
            contract_result = self._run_layer(
                "Contract Verification",
                lambda: self.contract_verifier.verify(
                    verification_input.code,
                    verification_input.contracts,
                    verification_input.code_id
                )
            )
            report.add_result(contract_result)
            
            if self._should_stop(contract_result):
                report.metadata["stopped_at"] = "contracts"
                return report
        
        # Layer 4: Unit Testing
        if self.config.enable_unit_tests:
            unit_test_result = self._run_layer(
                "Unit Testing",
                lambda: self.unit_tester.verify(
                    verification_input.code,
                    verification_input.test_cases,
                    verification_input.code_id
                )
            )
            report.add_result(unit_test_result)
            
            if self._should_stop(unit_test_result):
                report.metadata["stopped_at"] = "unit_tests"
                return report
        
        # Layer 5: Property-Based Testing
        if self.config.enable_property_tests:
            property_result = self._run_layer(
                "Property-Based Testing",
                lambda: self.property_tester.verify(
                    verification_input.code,
                    verification_input.properties,
                    verification_input.code_id
                )
            )
            report.add_result(property_result)
            
            if self._should_stop(property_result):
                report.metadata["stopped_at"] = "property_tests"
                return report
        
        # Layer 6: Static Analysis
        if self.config.enable_static_analysis:
            static_result = self._run_layer(
                "Static Analysis",
                lambda: self.static_analyzer.verify(
                    verification_input.code,
                    verification_input.code_id
                )
            )
            report.add_result(static_result)
            
            if self._should_stop(static_result):
                report.metadata["stopped_at"] = "static_analysis"
                return report
        
        # Layer 7: Security Scanning
        if self.config.enable_security:
            security_result = self._run_layer(
                "Security Scanning",
                lambda: self.security_scanner.verify(
                    verification_input.code,
                    verification_input.code_id
                )
            )
            report.add_result(security_result)
            
            if self._should_stop(security_result):
                report.metadata["stopped_at"] = "security"
                return report
        
        # Layer 8: Performance Checking
        if self.config.enable_performance:
            performance_result = self._run_layer(
                "Performance Checking",
                lambda: self.performance_checker.verify(
                    verification_input.code,
                    verification_input.performance_requirements,
                    verification_input.code_id
                )
            )
            report.add_result(performance_result)
            
            if self._should_stop(performance_result):
                report.metadata["stopped_at"] = "performance"
                return report
        
        # Optional: Formal Proof
        if self.config.enable_formal_proof:
            formal_result = self._run_layer(
                "Formal Verification",
                lambda: self.formal_prover.verify(
                    verification_input.code,
                    verification_input.formal_properties,
                    verification_input.code_id
                )
            )
            report.add_result(formal_result)
            
            if self._should_stop(formal_result):
                report.metadata["stopped_at"] = "formal_proof"
                return report
        
        # Optional: Compositional Verification
        if self.config.enable_compositional:
            compositional_result = self._run_layer(
                "Compositional Verification",
                lambda: self.compositional_verifier.verify(
                    verification_input.code,
                    verification_input.components,
                    verification_input.code_id
                )
            )
            report.add_result(compositional_result)
            
            if self._should_stop(compositional_result):
                report.metadata["stopped_at"] = "compositional"
                return report
        
        # Finalize report
        total_time = (time.time() - start_time) * 1000
        report.metadata["total_verification_time_ms"] = total_time
        report.metadata["layers_executed"] = len(report.results)
        
        return report
    
    def verify_quick(self, code: str, code_id: str = "unknown") -> VerificationReport:
        """
        Run quick verification (syntax, types, basic checks only).
        
        Args:
            code: Source code to verify
            code_id: Identifier for the code
            
        Returns:
            VerificationReport with quick verification results
        """
        # Create minimal config
        quick_config = VerificationStackConfig(
            enable_syntax=True,
            enable_types=True,
            enable_contracts=False,
            enable_unit_tests=False,
            enable_property_tests=False,
            enable_static_analysis=True,
            enable_security=True,
            enable_performance=False,
            enable_formal_proof=False,
            enable_compositional=False
        )
        
        # Create temporary stack with quick config
        quick_stack = VerificationStack(quick_config)
        
        # Run verification
        verification_input = VerificationInput(code=code, code_id=code_id)
        return quick_stack.verify(verification_input)
    
    def verify_comprehensive(
        self,
        code: str,
        code_id: str = "unknown",
        test_cases: Optional[List[TestCase]] = None,
        contracts: Optional[List[Contract]] = None
    ) -> VerificationReport:
        """
        Run comprehensive verification (all layers).
        
        Args:
            code: Source code to verify
            code_id: Identifier for the code
            test_cases: Unit test cases
            contracts: Code contracts
            
        Returns:
            VerificationReport with comprehensive verification results
        """
        # Create comprehensive config
        comprehensive_config = VerificationStackConfig(
            enable_syntax=True,
            enable_types=True,
            enable_contracts=True,
            enable_unit_tests=True,
            enable_property_tests=True,
            enable_static_analysis=True,
            enable_security=True,
            enable_performance=True,
            enable_formal_proof=True,
            enable_compositional=False
        )
        
        # Create temporary stack with comprehensive config
        comprehensive_stack = VerificationStack(comprehensive_config)
        
        # Run verification
        verification_input = VerificationInput(
            code=code,
            code_id=code_id,
            test_cases=test_cases,
            contracts=contracts
        )
        return comprehensive_stack.verify(verification_input)
    
    def _run_layer(self, layer_name: str, verifier_func) -> VerificationResult:
        """
        Run a single verification layer.
        
        Args:
            layer_name: Name of the layer
            verifier_func: Function to execute verification
            
        Returns:
            VerificationResult from the layer
        """
        # Execute verifier
        result = verifier_func()
        
        # Add layer name to metadata
        result.metadata["layer_name"] = layer_name
        
        return result
    
    def _should_stop(self, result: VerificationResult) -> bool:
        """
        Determine if verification should stop based on result.
        
        Args:
            result: Verification result to check
            
        Returns:
            True if verification should stop, False otherwise
        """
        # Stop if fail-fast is enabled and result failed
        if self.config.fail_fast and not result.passed:
            return True
        
        # Stop if critical issues found and stop_on_critical is enabled
        if self.config.stop_on_critical:
            critical_issues = [i for i in result.issues if i.severity == "critical"]
            if len(critical_issues) > 0:
                return True
        
        return False
    
    def get_layer_status(self) -> Dict[str, bool]:
        """
        Get status of which layers are enabled.
        
        Returns:
            Dictionary mapping layer names to enabled status
        """
        return {
            "syntax": self.config.enable_syntax,
            "types": self.config.enable_types,
            "contracts": self.config.enable_contracts,
            "unit_tests": self.config.enable_unit_tests,
            "property_tests": self.config.enable_property_tests,
            "static_analysis": self.config.enable_static_analysis,
            "security": self.config.enable_security,
            "performance": self.config.enable_performance,
            "formal_proof": self.config.enable_formal_proof,
            "compositional": self.config.enable_compositional
        }
    
    def configure_layer(self, layer: str, enabled: bool) -> None:
        """
        Enable or disable a specific verification layer.
        
        Args:
            layer: Layer name
            enabled: Whether to enable the layer
        """
        layer_map = {
            "syntax": "enable_syntax",
            "types": "enable_types",
            "contracts": "enable_contracts",
            "unit_tests": "enable_unit_tests",
            "property_tests": "enable_property_tests",
            "static_analysis": "enable_static_analysis",
            "security": "enable_security",
            "performance": "enable_performance",
            "formal_proof": "enable_formal_proof",
            "compositional": "enable_compositional"
        }
        
        if layer in layer_map:
            setattr(self.config, layer_map[layer], enabled)
