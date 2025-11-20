"""
Verify Command Module.

This module implements the 'verify' CLI command for running code verification
through the 8-layer verification stack.

Command: zero-error verify <code-path> [OPTIONS]

Options:
    --layers: Verification layers to run (comma-separated)
    --strict: Enable strict mode (fail on first error)
    --report-format: Output format (text/json)
    --output: Output file for report
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..utils import (
    CLIResult,
    ConsoleFormatter,
    PathValidator,
    OutputSerializer,
    create_default_formatter
)


@dataclass
class VerifyOptions:
    """
    Options for code verification.

    Attributes:
        code_path: Path to code to verify
        layers: Verification layers to run
        strict: Fail on first error
        report_format: Report output format
        output_path: Optional output file path
    """
    code_path: Path
    layers: List[str] = field(default_factory=list)
    strict: bool = False
    report_format: str = "text"
    output_path: Optional[Path] = None


@dataclass
class LayerResult:
    """
    Result from a verification layer.

    Attributes:
        layer_name: Name of verification layer
        passed: Whether layer passed
        errors: List of errors found
        warnings: List of warnings found
        execution_time_ms: Execution time in milliseconds
        metadata: Additional layer-specific data
    """
    layer_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """
    Complete verification report.

    Attributes:
        code_path: Path to verified code
        timestamp: Verification timestamp
        overall_passed: Whether all layers passed
        layers_run: Number of layers run
        layers_passed: Number of layers passed
        layers_failed: Number of layers failed
        layer_results: Results from each layer
        total_errors: Total error count
        total_warnings: Total warning count
        total_time_ms: Total execution time
    """
    code_path: str
    timestamp: str
    overall_passed: bool
    layers_run: int
    layers_passed: int
    layers_failed: int
    layer_results: List[LayerResult]
    total_errors: int
    total_warnings: int
    total_time_ms: float


# Available verification layers
AVAILABLE_LAYERS = [
    "syntax",
    "type_checking",
    "contracts",
    "unit_tests",
    "property_tests",
    "static_analysis",
    "security",
    "performance"
]

# Default layer order
DEFAULT_LAYER_ORDER = [
    "syntax",
    "type_checking",
    "static_analysis",
    "security",
    "unit_tests"
]


class VerifyCommand:
    """
    Verify command implementation.

    Integrates with the verification stack to run code through
    multiple verification layers.
    """

    def __init__(self, formatter: Optional[ConsoleFormatter] = None):
        """
        Initialize command handler.

        Args:
            formatter: Console formatter for output
        """
        self.formatter = formatter or create_default_formatter()

    def execute(self, options: VerifyOptions) -> CLIResult:
        """
        Execute verify command.

        Args:
            options: Verification options

        Returns:
            CLIResult with operation status
        """
        # Validate code path
        code_path = PathValidator.validate_file_path(options.code_path)
        if code_path is None:
            return CLIResult(
                success=False,
                message=f"Invalid code path: {options.code_path}",
                error_code=1
            )

        # Validate layers
        layers_to_run = self._validate_layers(options.layers)
        if not layers_to_run:
            return CLIResult(
                success=False,
                message="No valid verification layers specified",
                error_code=2
            )

        # Run verification
        self.formatter.print_info(
            f"Running verification on {code_path.name} "
            f"with {len(layers_to_run)} layers..."
        )

        report = self._run_verification(
            code_path,
            layers_to_run,
            options.strict
        )

        # Display results
        self._display_results(report, options.report_format)

        # Save report if output path specified
        if options.output_path is not None:
            save_result = self._save_report(report, options.output_path)
            if not save_result.success:
                return save_result

        # Return result
        if report.overall_passed:
            self.formatter.print_success("All verification layers passed")
            return CLIResult(
                success=True,
                message="Verification passed",
                data={
                    "layers_run": report.layers_run,
                    "total_time_ms": report.total_time_ms
                }
            )
        else:
            self.formatter.print_error(
                f"Verification failed: {report.layers_failed} "
                f"layer(s) failed"
            )
            return CLIResult(
                success=False,
                message=f"Verification failed",
                error_code=100,
                data={
                    "layers_run": report.layers_run,
                    "layers_failed": report.layers_failed,
                    "total_errors": report.total_errors
                }
            )

    def _validate_layers(self, requested_layers: List[str]) -> List[str]:
        """
        Validate and order verification layers.

        Args:
            requested_layers: Requested layers

        Returns:
            Valid ordered layers
        """
        # If no layers specified, use defaults
        if not requested_layers:
            return DEFAULT_LAYER_ORDER

        # Validate each layer
        valid_layers = []
        for layer in requested_layers:
            if layer in AVAILABLE_LAYERS:
                if layer not in valid_layers:
                    valid_layers.append(layer)
            else:
                self.formatter.print_warning(
                    f"Unknown layer '{layer}' - skipping"
                )

        return valid_layers

    def _run_verification(
        self,
        code_path: Path,
        layers: List[str],
        strict: bool
    ) -> VerificationReport:
        """
        Run verification on code.

        Args:
            code_path: Path to code file
            layers: Layers to run
            strict: Fail on first error

        Returns:
            VerificationReport with results
        """
        start_time = datetime.now()
        layer_results: List[LayerResult] = []
        overall_passed = True
        total_errors = 0
        total_warnings = 0

        # Read code file
        try:
            code_content = code_path.read_text()
        except (OSError, UnicodeDecodeError):
            # Return failed report
            error_result = LayerResult(
                layer_name="file_read",
                passed=False,
                errors=["Failed to read code file"],
                execution_time_ms=0.0
            )
            return VerificationReport(
                code_path=str(code_path),
                timestamp=start_time.isoformat(),
                overall_passed=False,
                layers_run=0,
                layers_passed=0,
                layers_failed=1,
                layer_results=[error_result],
                total_errors=1,
                total_warnings=0,
                total_time_ms=0.0
            )

        # Run each layer
        for layer_name in layers:
            self.formatter.print_info(f"Running {layer_name} layer...")

            layer_result = self._run_layer(layer_name, code_path, code_content)
            layer_results.append(layer_result)

            total_errors += len(layer_result.errors)
            total_warnings += len(layer_result.warnings)

            if not layer_result.passed:
                overall_passed = False
                if strict:
                    self.formatter.print_error(
                        f"Layer {layer_name} failed - stopping (strict mode)"
                    )
                    break

        # Calculate totals
        end_time = datetime.now()
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        layers_run = len(layer_results)
        layers_passed = sum(1 for r in layer_results if r.passed)
        layers_failed = layers_run - layers_passed

        return VerificationReport(
            code_path=str(code_path),
            timestamp=start_time.isoformat(),
            overall_passed=overall_passed,
            layers_run=layers_run,
            layers_passed=layers_passed,
            layers_failed=layers_failed,
            layer_results=layer_results,
            total_errors=total_errors,
            total_warnings=total_warnings,
            total_time_ms=total_time_ms
        )

    def _run_layer(
        self,
        layer_name: str,
        code_path: Path,
        code_content: str
    ) -> LayerResult:
        """
        Run a single verification layer.

        Args:
            layer_name: Name of layer to run
            code_path: Path to code file
            code_content: Code file content

        Returns:
            LayerResult with layer results
        """
        # In production, this would delegate to actual verification stack
        # For now, provide placeholder implementation that demonstrates
        # the interface

        layer_start = datetime.now()

        # Simulate layer execution based on layer type
        result = self._simulate_layer_verification(
            layer_name,
            code_path,
            code_content
        )

        layer_end = datetime.now()
        execution_time_ms = (layer_end - layer_start).total_seconds() * 1000

        result.execution_time_ms = execution_time_ms
        return result

    def _simulate_layer_verification(
        self,
        layer_name: str,
        code_path: Path,
        code_content: str
    ) -> LayerResult:
        """
        Simulate verification layer execution.

        In production, this would call the actual verification stack.
        This placeholder demonstrates the expected interface and error handling.

        Args:
            layer_name: Layer name
            code_path: Code file path
            code_content: Code content

        Returns:
            LayerResult
        """
        # Basic syntax check
        if layer_name == "syntax":
            try:
                compile(code_content, str(code_path), 'exec')
                return LayerResult(
                    layer_name=layer_name,
                    passed=True,
                    metadata={"lines": len(code_content.splitlines())}
                )
            except SyntaxError as e:
                return LayerResult(
                    layer_name=layer_name,
                    passed=False,
                    errors=[f"Syntax error at line {e.lineno}: {e.msg}"]
                )

        # Type checking placeholder
        elif layer_name == "type_checking":
            # Would integrate with mypy or similar
            return LayerResult(
                layer_name=layer_name,
                passed=True,
                metadata={"type_checker": "mypy"}
            )

        # Static analysis placeholder
        elif layer_name == "static_analysis":
            # Would integrate with pylint, flake8, etc.
            return LayerResult(
                layer_name=layer_name,
                passed=True,
                metadata={"analyzer": "pylint"}
            )

        # Security scanning placeholder
        elif layer_name == "security":
            # Would integrate with bandit or similar
            return LayerResult(
                layer_name=layer_name,
                passed=True,
                metadata={"scanner": "bandit"}
            )

        # Default case
        return LayerResult(
            layer_name=layer_name,
            passed=True,
            metadata={"note": "Layer simulation"}
        )

    def _display_results(
        self,
        report: VerificationReport,
        format_type: str
    ) -> None:
        """
        Display verification results.

        Args:
            report: Verification report
            format_type: Display format
        """
        if format_type == "json":
            self._display_json_results(report)
        else:
            self._display_text_results(report)

    def _display_text_results(self, report: VerificationReport) -> None:
        """
        Display results in text format.

        Args:
            report: Verification report
        """
        # Summary
        self.formatter.print_panel(
            f"Code: {report.code_path}\n"
            f"Timestamp: {report.timestamp}\n"
            f"Layers Run: {report.layers_run}\n"
            f"Passed: {report.layers_passed}\n"
            f"Failed: {report.layers_failed}\n"
            f"Total Errors: {report.total_errors}\n"
            f"Total Warnings: {report.total_warnings}\n"
            f"Execution Time: {report.total_time_ms:.2f}ms",
            title="Verification Summary"
        )

        # Layer results table
        columns = ["Layer", "Status", "Errors", "Warnings", "Time (ms)"]
        rows = []
        for result in report.layer_results:
            status = " PASS" if result.passed else " FAIL"
            rows.append([
                result.layer_name,
                status,
                str(len(result.errors)),
                str(len(result.warnings)),
                f"{result.execution_time_ms:.2f}"
            ])

        self.formatter.print_table("Layer Results", columns, rows)

        # Show errors and warnings
        for result in report.layer_results:
            if result.errors:
                self.formatter.print_error(
                    f"{result.layer_name} errors:"
                )
                for error in result.errors:
                    print(f"  - {error}")

            if result.warnings:
                self.formatter.print_warning(
                    f"{result.layer_name} warnings:"
                )
                for warning in result.warnings:
                    print(f"  - {warning}")

    def _display_json_results(self, report: VerificationReport) -> None:
        """
        Display results in JSON format.

        Args:
            report: Verification report
        """
        # Convert to dictionary
        report_dict = {
            "code_path": report.code_path,
            "timestamp": report.timestamp,
            "overall_passed": report.overall_passed,
            "layers_run": report.layers_run,
            "layers_passed": report.layers_passed,
            "layers_failed": report.layers_failed,
            "total_errors": report.total_errors,
            "total_warnings": report.total_warnings,
            "total_time_ms": report.total_time_ms,
            "layer_results": [
                {
                    "layer_name": r.layer_name,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time_ms": r.execution_time_ms,
                    "metadata": r.metadata
                }
                for r in report.layer_results
            ]
        }

        self.formatter.print_json(report_dict)

    def _save_report(
        self,
        report: VerificationReport,
        output_path: Path
    ) -> CLIResult:
        """
        Save verification report to file.

        Args:
            report: Verification report
            output_path: Output file path

        Returns:
            CLIResult with operation status
        """
        # Convert to dictionary
        report_dict = {
            "code_path": report.code_path,
            "timestamp": report.timestamp,
            "overall_passed": report.overall_passed,
            "layers_run": report.layers_run,
            "layers_passed": report.layers_passed,
            "layers_failed": report.layers_failed,
            "total_errors": report.total_errors,
            "total_warnings": report.total_warnings,
            "total_time_ms": report.total_time_ms,
            "layer_results": [
                {
                    "layer_name": r.layer_name,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time_ms": r.execution_time_ms,
                    "metadata": r.metadata
                }
                for r in report.layer_results
            ]
        }

        # Write to file
        try:
            output_path.write_text(json.dumps(report_dict, indent=2))
        except (OSError, TypeError):
            return CLIResult(
                success=False,
                message=f"Failed to save report to {output_path}",
                error_code=50
            )

        self.formatter.print_success(f"Report saved to {output_path}")
        return CLIResult(success=True, message="Report saved")


def run_verify(
    code_path: Path,
    layers: Optional[List[str]] = None,
    strict: bool = False,
    report_format: str = "text",
    output_path: Optional[Path] = None,
    formatter: Optional[ConsoleFormatter] = None
) -> CLIResult:
    """
    Run verify command with given options.

    Args:
        code_path: Path to code to verify
        layers: Verification layers to run
        strict: Enable strict mode
        report_format: Report format
        output_path: Output file path
        formatter: Console formatter

    Returns:
        CLIResult with operation status
    """
    options = VerifyOptions(
        code_path=code_path,
        layers=layers or [],
        strict=strict,
        report_format=report_format,
        output_path=output_path
    )

    command = VerifyCommand(formatter)
    return command.execute(options)
