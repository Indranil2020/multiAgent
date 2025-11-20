"""
Coder Agent Archetype.

This agent specializes in generating production-ready code for atomic tasks,
following all constraints and quality requirements.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class CoderAgent(BaseAgent):
    """
    Agent that generates production-ready code for atomic tasks.

    This agent takes task specifications and generates clean, tested,
    documented code that satisfies all contracts and constraints.

    Specialization:
    - Code generation for atomic tasks
    - Contract satisfaction
    - Constraint adherence (lines, complexity)
    - Type-safe implementation
    """

    def __init__(self):
        """Initialize coder agent with appropriate configuration."""
        system_prompt = """You are an expert programmer who writes clean, production-ready code.
Your code must:
1. Satisfy all preconditions and postconditions
2. Stay within line and complexity limits
3. Include full type annotations
4. Follow Python best practices
5. Be immediately executable
6. Handle all edge cases explicitly (no try/except for control flow)

Write concise, clear, correct code."""

        super().__init__(
            agent_type="coder",
            system_prompt_template=system_prompt,
            default_temperature=0.7,
            default_max_tokens=512
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle code generation tasks."""
        return task.task_type == TaskType.CODE_GENERATION

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for code generation."""
        prompt = self.create_base_prompt(task, config)

        prompt += "\n\nCode Generation Instructions:\n"
        prompt += "Generate complete, production-ready Python code.\n\n"

        prompt += "Requirements:\n"
        prompt += f"  - Maximum {task.max_lines} lines\n"
        prompt += f"  - Maximum cyclomatic complexity: {task.max_complexity}\n"
        prompt += "  - Full type annotations\n"
        prompt += "  - No try/except for control flow\n"
        prompt += "  - Explicit edge case handling\n\n"

        if task.test_cases:
            prompt += "Test Cases (your code must pass these):\n"
            for tc in task.test_cases[:3]:  # Show first 3 test cases
                prompt += f"  - {tc.name}: {tc.description}\n"
            prompt += "\n"

        if task.hints:
            prompt += "Hints:\n"
            for hint in task.hints:
                prompt += f"  - {hint}\n"
            prompt += "\n"

        prompt += "Output ONLY the Python code, no markdown formatting, no explanations.\n"
        prompt += "Start directly with imports or function definition.\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process LLM output into clean code."""
        if not llm_output or not llm_output.strip():
            return None

        code = llm_output.strip()

        # Remove markdown code blocks if present
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]

        if code.endswith('```'):
            code = code[:-3]

        code = code.strip()

        # Basic validation
        if not code:
            return None

        # Check line count
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) > task.max_lines:
            return None

        return code

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate generated code."""
        if output is None:
            return False

        if not isinstance(output, str):
            return False

        code = output.strip()

        if not code:
            return False

        # Must contain actual code indicators
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'return ', '=']
        has_code = any(indicator in code for indicator in code_indicators)

        if not has_code:
            return False

        # Check line count
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) > task.max_lines:
            return False

        # Should not have obvious errors
        error_indicators = ['TODO', 'FIXME', 'raise NotImplementedError']
        has_errors = any(indicator in code for indicator in error_indicators)

        if has_errors:
            return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for generated code."""
        if output is None or not isinstance(output, str):
            return 0.0

        code = output.strip()
        score = 0.3  # Base score

        # Check for type hints
        if ':' in code and '->' in code:
            score += 0.2

        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.15

        # Check line count (reward conciseness)
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) <= task.max_lines:
            score += 0.15

        # Check for function definition
        if 'def ' in code:
            score += 0.1

        # Check for return statement
        if 'return ' in code:
            score += 0.1

        return min(1.0, max(0.0, score))
