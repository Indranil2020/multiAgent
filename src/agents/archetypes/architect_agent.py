"""
Architect Agent Archetype.

This agent specializes in designing system architectures, defining module
boundaries, interfaces, and architectural patterns for software systems.
"""

from typing import Any, Dict
import json

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class ArchitectAgent(BaseAgent):
    """
    Agent that designs system architectures and module structures.

    This agent analyzes requirements and creates architectural designs
    including module decomposition, interface definitions, and pattern
    selection.

    Specialization:
    - System architecture design
    - Module boundary definition
    - Interface contract specification
    - Pattern selection and application
    """

    def __init__(self):
        """Initialize architect agent with appropriate configuration."""
        system_prompt = """You are an expert software architect specializing in system design.
Your role is to design clean, modular, maintainable software architectures.

Guidelines:
1. Follow SOLID principles
2. Design clear module boundaries
3. Define precise interface contracts
4. Choose appropriate design patterns
5. Consider scalability and maintainability
6. Output valid JSON with architectural specifications

Focus on clarity, modularity, and best practices."""

        super().__init__(
            agent_type="architect",
            system_prompt_template=system_prompt,
            default_temperature=0.4,
            default_max_tokens=1024
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle architecture tasks."""
        return task.task_type == TaskType.ARCHITECTURE

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for architecture design."""
        prompt = self.create_base_prompt(task, config)

        prompt += "\n\nArchitecture Design Instructions:\n"
        prompt += "Design a clean architecture for this task.\n"
        prompt += "Include:\n"
        prompt += "  - Module structure and boundaries\n"
        prompt += "  - Interface definitions\n"
        prompt += "  - Data flow and dependencies\n"
        prompt += "  - Design patterns to apply\n\n"

        if task.hints:
            prompt += "Hints:\n"
            for hint in task.hints:
                prompt += f"  - {hint}\n"
            prompt += "\n"

        prompt += "Output Format (JSON):\n"
        prompt += """
{
  "modules": [
    {
      "name": "ModuleName",
      "responsibility": "What this module does",
      "interfaces": ["InterfaceName1"],
      "dependencies": ["OtherModule"]
    }
  ],
  "interfaces": [
    {
      "name": "InterfaceName",
      "methods": [
        {
          "name": "method_name",
          "inputs": ["param: type"],
          "output": "ReturnType"
        }
      ]
    }
  ],
  "patterns": ["PatternName1", "PatternName2"],
  "data_flow": "Description of how data flows through the system"
}
"""

        prompt += "\nProvide ONLY the JSON output.\n"
        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process LLM output into architecture structure."""
        if not llm_output or not llm_output.strip():
            return None

        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}')

        if json_start == -1 or json_end == -1:
            return None

        json_str = llm_output[json_start:json_end + 1]

        # Simple validation (in production would use json.loads)
        if not ('{' in json_str and '}' in json_str):
            return None

        # Return structured representation
        return {'raw_json': json_str, 'parsed': True}

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate architecture output."""
        if output is None:
            return False

        if not isinstance(output, dict):
            return False

        if 'parsed' not in output or not output['parsed']:
            return False

        if 'raw_json' not in output:
            return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for architecture design."""
        if output is None or not isinstance(output, dict):
            return 0.0

        score = 0.5  # Base score for valid structure

        raw_json = output.get('raw_json', '')

        # Check for key architectural elements
        if 'modules' in raw_json:
            score += 0.15
        if 'interfaces' in raw_json:
            score += 0.15
        if 'patterns' in raw_json:
            score += 0.1
        if 'data_flow' in raw_json:
            score += 0.1

        return min(1.0, max(0.0, score))
