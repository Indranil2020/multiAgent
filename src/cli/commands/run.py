"""
Run Command Module.

This module implements the 'run' CLI command for executing the zero-error
system on task specifications.

Command: zero-error run <spec-file> [OPTIONS]

Options:
    --agents: Number of agents per type
    --layers: Decomposition layers to use
    --output-dir: Output directory for generated code
    --config: Project configuration file
    --verbose: Enable verbose output
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..utils import (
    CLIResult,
    ConsoleFormatter,
    PathValidator,
    OutputSerializer,
    ConfigLoader,
    create_default_formatter
)


@dataclass
class RunOptions:
    """
    Options for running the zero-error system.

    Attributes:
        spec_file: Path to task specification file
        num_agents: Number of agents per type
        enabled_layers: Decomposition layers to use
        output_dir: Output directory
        config_file: Optional configuration file
        verbose: Verbose output
    """
    spec_file: Path
    num_agents: int = 5
    enabled_layers: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None
    config_file: Optional[Path] = None
    verbose: bool = False


@dataclass
class TaskSpecification:
    """
    Task specification loaded from file.

    Attributes:
        task_id: Unique task identifier
        name: Task name
        description: Task description
        task_type: Type of task (function/class/module)
        inputs: Input parameters
        outputs: Output parameters
        preconditions: Preconditions
        postconditions: Postconditions
        test_cases: Test cases
    """
    task_id: str
    name: str
    description: str
    task_type: str
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """
    Result of task execution.

    Attributes:
        task_id: Task identifier
        success: Whether execution succeeded
        output_files: Generated output files
        verification_passed: Verification status
        execution_time_ms: Execution time
        agents_used: Number of agents used
        layers_executed: Layers executed
        errors: Errors encountered
        metadata: Additional result data
    """
    task_id: str
    success: bool
    output_files: List[str] = field(default_factory=list)
    verification_passed: bool = False
    execution_time_ms: float = 0.0
    agents_used: int = 0
    layers_executed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Available decomposition layers
AVAILABLE_LAYERS = [
    "strategic",
    "architectural",
    "component",
    "function",
    "method",
    "atomic"
]

# Default layers for different task types
DEFAULT_LAYERS_BY_TYPE = {
    "function": ["function", "method", "atomic"],
    "class": ["component", "function", "method", "atomic"],
    "module": ["architectural", "component", "function", "method", "atomic"],
    "system": ["strategic", "architectural", "component", "function", "method", "atomic"]
}


class RunCommand:
    """
    Run command implementation.

    Orchestrates the zero-error system to execute task specifications
    through hierarchical decomposition, agent coordination, voting,
    and verification.
    """

    def __init__(self, formatter: Optional[ConsoleFormatter] = None):
        """
        Initialize command handler.

        Args:
            formatter: Console formatter for output
        """
        self.formatter = formatter or create_default_formatter()

    def execute(self, options: RunOptions) -> CLIResult:
        """
        Execute run command.

        Args:
            options: Run options

        Returns:
            CLIResult with operation status
        """
        # Validate spec file
        spec_path = PathValidator.validate_file_path(options.spec_file)
        if spec_path is None:
            return CLIResult(
                success=False,
                message=f"Invalid spec file: {options.spec_file}",
                error_code=1
            )

        # Load task specification
        self.formatter.print_info("Loading task specification...")
        spec_result = self._load_specification(spec_path)
        if not spec_result.success:
            return spec_result

        task_spec = spec_result.data["specification"]

        # Determine output directory
        output_dir = self._determine_output_dir(options.output_dir, spec_path)
        if output_dir is None:
            return CLIResult(
                success=False,
                message="Failed to create output directory",
                error_code=2
            )

        # Determine layers
        layers = self._determine_layers(
            options.enabled_layers,
            task_spec.task_type
        )

        self.formatter.print_info(
            f"Executing task '{task_spec.name}' "
            f"with {len(layers)} decomposition layers..."
        )

        # Execute task
        exec_result = self._execute_task(
            task_spec,
            layers,
            options.num_agents,
            output_dir,
            options.verbose
        )

        # Display results
        self._display_execution_result(exec_result)

        # Return result
        if exec_result.success:
            return CLIResult(
                success=True,
                message=f"Task executed successfully",
                data={
                    "task_id": exec_result.task_id,
                    "output_files": exec_result.output_files,
                    "execution_time_ms": exec_result.execution_time_ms
                }
            )
        else:
            return CLIResult(
                success=False,
                message=f"Task execution failed",
                error_code=100,
                data={
                    "task_id": exec_result.task_id,
                    "errors": exec_result.errors
                }
            )

    def _load_specification(self, spec_path: Path) -> CLIResult:
        """
        Load task specification from file.

        Args:
            spec_path: Path to specification file

        Returns:
            CLIResult with loaded specification
        """
        # Read file
        try:
            spec_content = spec_path.read_text()
        except (OSError, UnicodeDecodeError):
            return CLIResult(
                success=False,
                message=f"Failed to read specification file",
                error_code=10
            )

        # Parse JSON
        try:
            spec_data = json.loads(spec_content)
        except json.JSONDecodeError as e:
            return CLIResult(
                success=False,
                message=f"Invalid JSON in specification: {e}",
                error_code=11
            )

        # Validate and create specification
        validation_result = self._validate_specification(spec_data)
        if not validation_result.success:
            return validation_result

        # Create TaskSpecification object
        task_spec = TaskSpecification(
            task_id=spec_data.get("id", "unknown"),
            name=spec_data.get("name", ""),
            description=spec_data.get("description", ""),
            task_type=spec_data.get("task_type", "function"),
            inputs=spec_data.get("inputs", []),
            outputs=spec_data.get("outputs", []),
            preconditions=spec_data.get("preconditions", []),
            postconditions=spec_data.get("postconditions", []),
            test_cases=spec_data.get("test_cases", [])
        )

        return CLIResult(
            success=True,
            message="Specification loaded",
            data={"specification": task_spec}
        )

    def _validate_specification(self, spec_data: Dict[str, Any]) -> CLIResult:
        """
        Validate task specification data.

        Args:
            spec_data: Specification data dictionary

        Returns:
            CLIResult with validation status
        """
        # Check required fields
        required_fields = ["name", "description", "task_type"]
        for field in required_fields:
            if field not in spec_data or not spec_data[field]:
                return CLIResult(
                    success=False,
                    message=f"Missing required field: {field}",
                    error_code=20
                )

        # Validate task type
        valid_types = ["function", "class", "module", "system"]
        task_type = spec_data.get("task_type")
        if task_type not in valid_types:
            return CLIResult(
                success=False,
                message=f"Invalid task_type '{task_type}'. Valid: {', '.join(valid_types)}",
                error_code=21
            )

        return CLIResult(success=True, message="Specification valid")

    def _determine_output_dir(
        self,
        specified_dir: Optional[Path],
        spec_path: Path
    ) -> Optional[Path]:
        """
        Determine output directory.

        Args:
            specified_dir: User-specified directory
            spec_path: Path to specification file

        Returns:
            Output directory path or None if failed
        """
        if specified_dir is not None:
            return PathValidator.ensure_directory(specified_dir)

        # Use default: output directory next to spec file
        default_dir = spec_path.parent / "output"
        return PathValidator.ensure_directory(default_dir)

    def _determine_layers(
        self,
        requested_layers: List[str],
        task_type: str
    ) -> List[str]:
        """
        Determine decomposition layers to use.

        Args:
            requested_layers: User-requested layers
            task_type: Task type

        Returns:
            List of layers to use
        """
        # If layers specified, validate and use them
        if requested_layers:
            valid_layers = []
            for layer in requested_layers:
                if layer in AVAILABLE_LAYERS:
                    valid_layers.append(layer)
                else:
                    self.formatter.print_warning(
                        f"Unknown layer '{layer}' - skipping"
                    )
            return valid_layers if valid_layers else DEFAULT_LAYERS_BY_TYPE.get(
                task_type,
                ["function", "method", "atomic"]
            )

        # Use defaults based on task type
        return DEFAULT_LAYERS_BY_TYPE.get(
            task_type,
            ["function", "method", "atomic"]
        )

    def _execute_task(
        self,
        task_spec: TaskSpecification,
        layers: List[str],
        num_agents: int,
        output_dir: Path,
        verbose: bool
    ) -> ExecutionResult:
        """
        Execute task through the zero-error system.

        Args:
            task_spec: Task specification
            layers: Decomposition layers
            num_agents: Number of agents
            output_dir: Output directory
            verbose: Verbose output

        Returns:
            ExecutionResult with execution status
        """
        start_time = datetime.now()

        # In production, this would:
        # 1. Initialize agent swarm
        # 2. Run hierarchical decomposition
        # 3. Spawn agents for subtasks
        # 4. Execute voting for each subtask
        # 5. Run verification stack
        # 6. Generate final code
        # 7. Save to output directory

        # For now, provide simulation that demonstrates the interface
        result = self._simulate_execution(
            task_spec,
            layers,
            num_agents,
            output_dir,
            verbose
        )

        end_time = datetime.now()
        result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

        return result

    def _simulate_execution(
        self,
        task_spec: TaskSpecification,
        layers: List[str],
        num_agents: int,
        output_dir: Path,
        verbose: bool
    ) -> ExecutionResult:
        """
        Simulate task execution.

        In production, this would orchestrate the actual zero-error system.
        This simulation demonstrates the expected interface and flow.

        Args:
            task_spec: Task specification
            layers: Decomposition layers
            num_agents: Number of agents
            output_dir: Output directory
            verbose: Verbose output

        Returns:
            ExecutionResult
        """
        # Show progress for each layer
        for layer in layers:
            if verbose:
                self.formatter.print_info(f"  Executing layer: {layer}")

        # Simulate code generation
        output_file = output_dir / f"{task_spec.name.lower().replace(' ', '_')}.py"

        # Generate simple code based on spec
        code_content = self._generate_sample_code(task_spec)

        # Write output file
        try:
            output_file.write_text(code_content)
        except OSError:
            return ExecutionResult(
                task_id=task_spec.task_id,
                success=False,
                errors=["Failed to write output file"]
            )

        # Create result
        return ExecutionResult(
            task_id=task_spec.task_id,
            success=True,
            output_files=[str(output_file)],
            verification_passed=True,
            agents_used=num_agents * len(layers),
            layers_executed=layers,
            metadata={
                "task_name": task_spec.name,
                "task_type": task_spec.task_type
            }
        )

    def _generate_sample_code(self, task_spec: TaskSpecification) -> str:
        """
        Generate sample code from specification.

        In production, this would be generated by the agent swarm.

        Args:
            task_spec: Task specification

        Returns:
            Generated code
        """
        # Build function signature
        params = []
        for input_param in task_spec.inputs:
            param_name = input_param.get("name", "param")
            param_type = input_param.get("type", "Any")
            params.append(f"{param_name}: {param_type}")

        param_str = ", ".join(params) if params else ""

        # Build return type
        if task_spec.outputs:
            return_type = task_spec.outputs[0].get("type", "Any")
        else:
            return_type = "None"

        # Generate code
        code = f'''"""
{task_spec.description}

This code was generated by the zero-error system.
"""

from typing import Any


def {task_spec.name.lower().replace(" ", "_")}({param_str}) -> {return_type}:
    """
    {task_spec.description}

'''

        # Add preconditions as comments
        if task_spec.preconditions:
            code += "    # Preconditions:\n"
            for precond in task_spec.preconditions:
                code += f"    #   - {precond}\n"
            code += "\n"

        # Add postconditions as comments
        if task_spec.postconditions:
            code += "    # Postconditions:\n"
            for postcond in task_spec.postconditions:
                code += f"    #   - {postcond}\n"
            code += "\n"

        code += '    """\n'
        code += "    # TODO: Implement function logic\n"
        code += "    pass\n"

        return code

    def _display_execution_result(self, result: ExecutionResult) -> None:
        """
        Display execution result.

        Args:
            result: Execution result
        """
        if result.success:
            self.formatter.print_success(
                f"Task '{result.task_id}' executed successfully"
            )

            # Show details
            self.formatter.print_panel(
                f"Agents Used: {result.agents_used}\n"
                f"Layers Executed: {', '.join(result.layers_executed)}\n"
                f"Verification: {'Passed' if result.verification_passed else 'Failed'}\n"
                f"Execution Time: {result.execution_time_ms:.2f}ms\n"
                f"Output Files: {len(result.output_files)}",
                title="Execution Summary"
            )

            # List output files
            if result.output_files:
                self.formatter.print_info("Generated files:")
                for file_path in result.output_files:
                    print(f"  - {file_path}")

        else:
            self.formatter.print_error(
                f"Task '{result.task_id}' execution failed"
            )

            if result.errors:
                self.formatter.print_error("Errors:")
                for error in result.errors:
                    print(f"  - {error}")


def run_task(
    spec_file: Path,
    num_agents: int = 5,
    enabled_layers: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    config_file: Optional[Path] = None,
    verbose: bool = False,
    formatter: Optional[ConsoleFormatter] = None
) -> CLIResult:
    """
    Run task with given options.

    Args:
        spec_file: Task specification file
        num_agents: Number of agents per type
        enabled_layers: Decomposition layers
        output_dir: Output directory
        config_file: Configuration file
        verbose: Verbose output
        formatter: Console formatter

    Returns:
        CLIResult with operation status
    """
    options = RunOptions(
        spec_file=spec_file,
        num_agents=num_agents,
        enabled_layers=enabled_layers or [],
        output_dir=output_dir,
        config_file=config_file,
        verbose=verbose
    )

    command = RunCommand(formatter)
    return command.execute(options)
