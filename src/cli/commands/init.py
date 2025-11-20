"""
Init Command Module.

This module implements the 'init' CLI command for initializing new zero-error
projects. It creates the necessary directory structure, configuration files,
and example specifications.

Command: zero-error init <project-path> [OPTIONS]

Options:
    --template: Project template (web/api/cli/library)
    --with-examples: Include example task specifications
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

from ..utils import (
    CLIResult,
    ConsoleFormatter,
    PathValidator,
    create_default_formatter
)


@dataclass
class InitOptions:
    """
    Options for project initialization.

    Attributes:
        project_path: Path for new project
        template: Project template type
        with_examples: Whether to include examples
    """
    project_path: Path
    template: str = "api"
    with_examples: bool = True


@dataclass
class ProjectTemplate:
    """
    Project template definition.

    Attributes:
        name: Template name
        description: Template description
        directories: Directories to create
        config: Default configuration
    """
    name: str
    description: str
    directories: List[str]
    config: Dict[str, Any]


# Template definitions
TEMPLATES: Dict[str, ProjectTemplate] = {
    "api": ProjectTemplate(
        name="api",
        description="REST API project",
        directories=[
            "config",
            "specs",
            "specs/tasks",
            "specs/agents",
            "output",
            "output/code",
            "output/tests",
            "output/docs",
            "logs"
        ],
        config={
            "project_type": "api",
            "verification_layers": [
                "syntax",
                "type_checking",
                "contracts",
                "unit_tests",
                "static_analysis",
                "security"
            ],
            "agent_pool": {
                "coder": 5,
                "verifier": 3,
                "tester": 2
            },
            "voting": {
                "k_threshold": 2,
                "timeout_seconds": 30
            }
        }
    ),
    "web": ProjectTemplate(
        name="web",
        description="Web application project",
        directories=[
            "config",
            "specs",
            "specs/components",
            "specs/pages",
            "output",
            "output/components",
            "output/pages",
            "output/tests",
            "logs"
        ],
        config={
            "project_type": "web",
            "verification_layers": [
                "syntax",
                "type_checking",
                "unit_tests",
                "static_analysis",
                "security"
            ],
            "agent_pool": {
                "coder": 5,
                "verifier": 2,
                "tester": 2
            },
            "voting": {
                "k_threshold": 2,
                "timeout_seconds": 30
            }
        }
    ),
    "cli": ProjectTemplate(
        name="cli",
        description="Command-line application",
        directories=[
            "config",
            "specs",
            "specs/commands",
            "output",
            "output/commands",
            "output/tests",
            "logs"
        ],
        config={
            "project_type": "cli",
            "verification_layers": [
                "syntax",
                "type_checking",
                "contracts",
                "unit_tests",
                "static_analysis"
            ],
            "agent_pool": {
                "coder": 3,
                "verifier": 2,
                "tester": 1
            },
            "voting": {
                "k_threshold": 2,
                "timeout_seconds": 30
            }
        }
    ),
    "library": ProjectTemplate(
        name="library",
        description="Python library project",
        directories=[
            "config",
            "specs",
            "specs/modules",
            "output",
            "output/modules",
            "output/tests",
            "output/docs",
            "logs"
        ],
        config={
            "project_type": "library",
            "verification_layers": [
                "syntax",
                "type_checking",
                "contracts",
                "unit_tests",
                "property_tests",
                "static_analysis"
            ],
            "agent_pool": {
                "coder": 3,
                "verifier": 3,
                "tester": 2
            },
            "voting": {
                "k_threshold": 2,
                "timeout_seconds": 30
            }
        }
    )
}


# Example task specifications
EXAMPLE_TASK_SPEC = {
    "id": "example_001",
    "name": "Example Task",
    "description": "Example task specification demonstrating the zero-error system",
    "task_type": "function",
    "inputs": [
        {
            "name": "input_value",
            "type": "int",
            "description": "Input integer value"
        }
    ],
    "outputs": [
        {
            "name": "result",
            "type": "int",
            "description": "Processed result"
        }
    ],
    "preconditions": [
        "input_value >= 0"
    ],
    "postconditions": [
        "result >= 0",
        "result == input_value * 2"
    ],
    "test_cases": [
        {
            "input": {"input_value": 0},
            "expected_output": {"result": 0}
        },
        {
            "input": {"input_value": 5},
            "expected_output": {"result": 10}
        },
        {
            "input": {"input_value": 100},
            "expected_output": {"result": 200}
        }
    ]
}


class InitCommand:
    """
    Init command implementation.

    Handles project initialization including directory creation,
    configuration generation, and example file creation.
    """

    def __init__(self, formatter: Optional[ConsoleFormatter] = None):
        """
        Initialize command handler.

        Args:
            formatter: Console formatter for output
        """
        self.formatter = formatter or create_default_formatter()

    def execute(self, options: InitOptions) -> CLIResult:
        """
        Execute init command.

        Args:
            options: Initialization options

        Returns:
            CLIResult with operation status
        """
        # Validate project path
        if options.project_path.exists():
            return CLIResult(
                success=False,
                message=f"Path already exists: {options.project_path}",
                error_code=1
            )

        # Validate template
        if options.template not in TEMPLATES:
            valid_templates = ", ".join(TEMPLATES.keys())
            return CLIResult(
                success=False,
                message=f"Invalid template '{options.template}'. Valid: {valid_templates}",
                error_code=2
            )

        template = TEMPLATES[options.template]

        # Create project structure
        creation_result = self._create_project_structure(
            options.project_path,
            template
        )
        if not creation_result.success:
            return creation_result

        # Create configuration
        config_result = self._create_configuration(
            options.project_path,
            template
        )
        if not config_result.success:
            return config_result

        # Create examples if requested
        if options.with_examples:
            examples_result = self._create_examples(options.project_path)
            if not examples_result.success:
                return examples_result

        # Create README
        readme_result = self._create_readme(
            options.project_path,
            template
        )
        if not readme_result.success:
            return readme_result

        self.formatter.print_success(
            f"Initialized {template.name} project at {options.project_path}"
        )

        return CLIResult(
            success=True,
            message=f"Project initialized successfully",
            data={
                "project_path": str(options.project_path),
                "template": template.name
            }
        )

    def _create_project_structure(
        self,
        project_path: Path,
        template: ProjectTemplate
    ) -> CLIResult:
        """
        Create project directory structure.

        Args:
            project_path: Root project path
            template: Project template

        Returns:
            CLIResult with operation status
        """
        # Create root directory
        root_path = PathValidator.ensure_directory(project_path)
        if root_path is None:
            return CLIResult(
                success=False,
                message=f"Failed to create project directory: {project_path}",
                error_code=10
            )

        self.formatter.print_info(f"Creating project structure...")

        # Create subdirectories
        for directory in template.directories:
            dir_path = project_path / directory
            result_path = PathValidator.ensure_directory(dir_path)
            if result_path is None:
                return CLIResult(
                    success=False,
                    message=f"Failed to create directory: {directory}",
                    error_code=11
                )

        return CLIResult(success=True, message="Structure created")

    def _create_configuration(
        self,
        project_path: Path,
        template: ProjectTemplate
    ) -> CLIResult:
        """
        Create project configuration files.

        Args:
            project_path: Root project path
            template: Project template

        Returns:
            CLIResult with operation status
        """
        self.formatter.print_info("Creating configuration files...")

        # Create project config
        config_path = project_path / "config" / "project.json"
        config_data = {
            "name": project_path.name,
            "version": "0.1.0",
            "template": template.name,
            **template.config
        }

        try:
            config_path.write_text(json.dumps(config_data, indent=2))
        except (OSError, TypeError):
            return CLIResult(
                success=False,
                message="Failed to write project configuration",
                error_code=20
            )

        # Create verification config
        verification_path = project_path / "config" / "verification.json"
        verification_data = {
            "layers": template.config["verification_layers"],
            "strict_mode": True,
            "fail_fast": False
        }

        try:
            verification_path.write_text(json.dumps(verification_data, indent=2))
        except (OSError, TypeError):
            return CLIResult(
                success=False,
                message="Failed to write verification configuration",
                error_code=21
            )

        # Create agent config
        agent_path = project_path / "config" / "agents.json"
        agent_data = {
            "pool": template.config["agent_pool"],
            "max_agents": sum(template.config["agent_pool"].values()) * 2,
            "spawn_strategy": "on_demand"
        }

        try:
            agent_path.write_text(json.dumps(agent_data, indent=2))
        except (OSError, TypeError):
            return CLIResult(
                success=False,
                message="Failed to write agent configuration",
                error_code=22
            )

        return CLIResult(success=True, message="Configuration created")

    def _create_examples(self, project_path: Path) -> CLIResult:
        """
        Create example files.

        Args:
            project_path: Root project path

        Returns:
            CLIResult with operation status
        """
        self.formatter.print_info("Creating example specifications...")

        # Create example task spec
        example_path = project_path / "specs" / "tasks" / "example_task.json"

        # Ensure parent directory exists
        if not example_path.parent.exists():
            parent = PathValidator.ensure_directory(example_path.parent)
            if parent is None:
                return CLIResult(
                    success=False,
                    message="Failed to create specs/tasks directory",
                    error_code=30
                )

        try:
            example_path.write_text(json.dumps(EXAMPLE_TASK_SPEC, indent=2))
        except (OSError, TypeError):
            return CLIResult(
                success=False,
                message="Failed to write example specification",
                error_code=31
            )

        return CLIResult(success=True, message="Examples created")

    def _create_readme(
        self,
        project_path: Path,
        template: ProjectTemplate
    ) -> CLIResult:
        """
        Create project README.

        Args:
            project_path: Root project path
            template: Project template

        Returns:
            CLIResult with operation status
        """
        readme_path = project_path / "README.md"

        readme_content = f"""# {project_path.name}

{template.description} initialized with the zero-error system.

## Project Structure

- `config/` - Configuration files
- `specs/` - Task specifications
- `output/` - Generated code and artifacts
- `logs/` - System logs

## Getting Started

1. Define your task specifications in `specs/tasks/`
2. Run the system: `zero-error run specs/tasks/your_task.json`
3. View generated code in `output/code/`

## Configuration

Edit configuration files in `config/`:
- `project.json` - Project settings
- `verification.json` - Verification settings
- `agents.json` - Agent pool settings

## Documentation

For more information, see the zero-error system documentation.
"""

        try:
            readme_path.write_text(readme_content)
        except OSError:
            return CLIResult(
                success=False,
                message="Failed to write README",
                error_code=40
            )

        return CLIResult(success=True, message="README created")


def run_init(
    project_path: Path,
    template: str = "api",
    with_examples: bool = True,
    formatter: Optional[ConsoleFormatter] = None
) -> CLIResult:
    """
    Run init command with given options.

    Args:
        project_path: Path for new project
        template: Project template
        with_examples: Include examples
        formatter: Console formatter

    Returns:
        CLIResult with operation status
    """
    options = InitOptions(
        project_path=project_path,
        template=template,
        with_examples=with_examples
    )

    command = InitCommand(formatter)
    return command.execute(options)
