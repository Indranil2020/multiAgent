"""
Verification API Routes.

This module provides REST endpoints for code verification and quality
checking in the zero-error system.

Endpoints:
- POST   /verification/verify     - Verify code
- GET    /verification/results/{id} - Get verification results
- GET    /verification/stats      - Get verification statistics
- POST   /verification/layers     - Run specific verification layers
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VerificationRequest:
    """
    Request to verify code.

    Attributes:
        code: Source code to verify
        language: Programming language
        layers: Verification layers to run
        max_complexity: Maximum allowed complexity
        max_lines: Maximum allowed lines
    """
    code: str
    language: str = "python"
    layers: List[str] = field(default_factory=list)
    max_complexity: int = 10
    max_lines: int = 20

    def validate(self) -> bool:
        """Validate request."""
        if not self.code or len(self.code.strip()) == 0:
            return False

        valid_languages = ['python', 'javascript', 'typescript', 'go', 'rust']
        if self.language not in valid_languages:
            return False

        valid_layers = [
            'syntax', 'type_checking', 'contracts',
            'unit_tests', 'property_tests', 'static_analysis',
            'security', 'performance'
        ]

        # If no layers specified, use all
        if not self.layers:
            self.layers = valid_layers
        else:
            # Validate layer names
            for layer in self.layers:
                if layer not in valid_layers:
                    return False

        if not (1 <= self.max_complexity <= 50):
            return False

        if not (1 <= self.max_lines <= 1000):
            return False

        return True


@dataclass
class LayerResult:
    """
    Result from a single verification layer.

    Attributes:
        layer_name: Name of verification layer
        passed: Whether layer passed
        errors: List of errors found
        warnings: List of warnings
        execution_time_ms: Layer execution time
    """
    layer_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


@dataclass
class VerificationResponse:
    """
    Response from code verification.

    Attributes:
        verification_id: Unique verification identifier
        overall_passed: Whether all layers passed
        layers_passed: Number of layers passed
        layers_failed: Number of layers failed
        layer_results: Results from each layer
        quality_score: Overall quality score (0.0-1.0)
        total_execution_time_ms: Total verification time
    """
    verification_id: str
    overall_passed: bool
    layers_passed: int
    layers_failed: int
    layer_results: List[LayerResult] = field(default_factory=list)
    quality_score: float = 0.0
    total_execution_time_ms: int = 0


@dataclass
class VerificationStatsResponse:
    """
    Response containing verification statistics.

    Attributes:
        total_verifications: Total verifications run
        total_passed: Total that passed all layers
        total_failed: Total that failed at least one layer
        avg_quality_score: Average quality score
        avg_execution_time_ms: Average verification time
        layer_stats: Statistics by layer
    """
    total_verifications: int
    total_passed: int
    total_failed: int
    avg_quality_score: float
    avg_execution_time_ms: float
    layer_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class VerificationRouteHandler:
    """
    Handler for verification-related API routes.

    This class implements the business logic for code verification
    endpoints, integrating with the verification stack.
    """

    def __init__(self, verification_stack=None):
        """
        Initialize verification route handler.

        Args:
            verification_stack: Verification stack
        """
        self.verification_stack = verification_stack
        self.verification_results: Dict[str, VerificationResponse] = {}
        self.stats = {
            'total_verifications': 0,
            'total_passed': 0,
            'total_failed': 0,
            'total_quality_score': 0.0,
            'total_execution_time_ms': 0
        }

    def verify_code(self, request: VerificationRequest) -> Optional[VerificationResponse]:
        """
        Verify code against specified layers.

        Args:
            request: Verification request

        Returns:
            Verification response or None if verification failed to start
        """
        if not request.validate():
            return None

        # Generate verification ID
        import hashlib
        verification_id = hashlib.sha256(
            f"{request.language}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        # Run verification layers
        layer_results = []
        layers_passed = 0
        layers_failed = 0
        total_time = 0

        for layer_name in request.layers:
            # Simulate layer execution
            # In production, would call actual verification stack
            result = self._run_verification_layer(
                layer_name,
                request.code,
                request.language,
                request.max_complexity,
                request.max_lines
            )

            layer_results.append(result)
            total_time += result.execution_time_ms

            if result.passed:
                layers_passed += 1
            else:
                layers_failed += 1

        # Calculate overall result
        overall_passed = layers_failed == 0

        # Calculate quality score
        quality_score = 0.0
        if len(layer_results) > 0:
            quality_score = layers_passed / len(layer_results)

        # Create response
        response = VerificationResponse(
            verification_id=verification_id,
            overall_passed=overall_passed,
            layers_passed=layers_passed,
            layers_failed=layers_failed,
            layer_results=layer_results,
            quality_score=quality_score,
            total_execution_time_ms=total_time
        )

        # Store result
        self.verification_results[verification_id] = response

        # Update stats
        self.stats['total_verifications'] += 1
        if overall_passed:
            self.stats['total_passed'] += 1
        else:
            self.stats['total_failed'] += 1
        self.stats['total_quality_score'] += quality_score
        self.stats['total_execution_time_ms'] += total_time

        return response

    def get_verification_result(
        self,
        verification_id: str
    ) -> Optional[VerificationResponse]:
        """
        Get verification result by ID.

        Args:
            verification_id: Verification identifier

        Returns:
            Verification response or None if not found
        """
        if not verification_id or verification_id not in self.verification_results:
            return None

        return self.verification_results[verification_id]

    def get_verification_stats(self) -> VerificationStatsResponse:
        """
        Get verification statistics.

        Returns:
            Verification statistics response
        """
        avg_quality_score = 0.0
        if self.stats['total_verifications'] > 0:
            avg_quality_score = (
                self.stats['total_quality_score'] /
                self.stats['total_verifications']
            )

        avg_execution_time = 0.0
        if self.stats['total_verifications'] > 0:
            avg_execution_time = (
                self.stats['total_execution_time_ms'] /
                self.stats['total_verifications']
            )

        # Calculate layer statistics
        layer_stats = self._calculate_layer_stats()

        return VerificationStatsResponse(
            total_verifications=self.stats['total_verifications'],
            total_passed=self.stats['total_passed'],
            total_failed=self.stats['total_failed'],
            avg_quality_score=avg_quality_score,
            avg_execution_time_ms=avg_execution_time,
            layer_stats=layer_stats
        )

    def _run_verification_layer(
        self,
        layer_name: str,
        code: str,
        language: str,
        max_complexity: int,
        max_lines: int
    ) -> LayerResult:
        """
        Run a single verification layer.

        Args:
            layer_name: Layer to run
            code: Source code
            language: Programming language
            max_complexity: Maximum complexity
            max_lines: Maximum lines

        Returns:
            Layer result
        """
        # Simplified simulation
        # In production, would call actual verification layer

        errors = []
        warnings = []
        passed = True

        # Basic checks as simulation
        lines = code.split('\n')
        actual_lines = len([line for line in lines if line.strip()])

        if layer_name == 'syntax':
            # Simulate syntax check
            if not code.strip():
                errors.append("Empty code")
                passed = False

        elif layer_name == 'contracts':
            # Simulate contract check
            if actual_lines > max_lines:
                errors.append(f"Code exceeds max lines: {actual_lines} > {max_lines}")
                passed = False

        elif layer_name == 'static_analysis':
            # Simulate static analysis
            if len(code) < 10:
                warnings.append("Code is very short")

        # Simulate execution time
        execution_time_ms = len(code) // 10 + 10

        return LayerResult(
            layer_name=layer_name,
            passed=passed,
            errors=errors,
            warnings=warnings,
            execution_time_ms=execution_time_ms
        )

    def _calculate_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics by layer.

        Returns:
            Layer statistics dictionary
        """
        layer_counts: Dict[str, Dict[str, int]] = {}

        for result in self.verification_results.values():
            for layer_result in result.layer_results:
                layer_name = layer_result.layer_name

                if layer_name not in layer_counts:
                    layer_counts[layer_name] = {
                        'total': 0,
                        'passed': 0,
                        'failed': 0
                    }

                layer_counts[layer_name]['total'] += 1
                if layer_result.passed:
                    layer_counts[layer_name]['passed'] += 1
                else:
                    layer_counts[layer_name]['failed'] += 1

        # Convert to percentage
        layer_stats: Dict[str, Dict[str, Any]] = {}
        for layer_name, counts in layer_counts.items():
            pass_rate = 0.0
            if counts['total'] > 0:
                pass_rate = counts['passed'] / counts['total']

            layer_stats[layer_name] = {
                'total_runs': counts['total'],
                'pass_rate': pass_rate,
                'passed': counts['passed'],
                'failed': counts['failed']
            }

        return layer_stats
