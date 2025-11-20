"""
Documenter Agent Archetype.

This agent specializes in generating comprehensive documentation including
docstrings, API docs, usage examples, and design documentation.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class DocumenterAgent(BaseAgent):
    """
    Agent that generates comprehensive documentation.

    This agent creates clear, thorough documentation for code including
    docstrings, API documentation, usage examples, and design docs.

    Specialization:
    - Docstring generation (Google style)
    - API documentation
    - Usage examples
    - Design documentation
    """

    def __init__(self):
        """Initialize documenter agent with appropriate configuration."""
        system_prompt = """You are an expert technical writer specializing in code documentation.
Your role is to create clear, comprehensive documentation.

Guidelines:
1. Use Google-style docstrings
2. Explain what, why, and how
3. Include usage examples
4. Document parameters and return values
5. Note any edge cases or limitations
6. Write for developers who will use the code

Focus on clarity and completeness."""

        super().__init__(
            agent_type="documenter",
            system_prompt_template=system_prompt,
            default_temperature=0.5,
            default_max_tokens=768
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle documentation tasks."""
        return task.task_type == TaskType.DOCUMENTATION

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for documentation generation."""
        prompt = self.create_base_prompt(task, config)

        # Get code to document from context
        code_to_document = context_data.get('code', '')

        prompt += "\n\nCode to Document:\n"
        prompt += "```python\n"
        prompt += code_to_document
        prompt += "\n```\n\n"

        prompt += "Documentation Instructions:\n"
        prompt += "Generate comprehensive documentation including:\n"
        prompt += "1. Module-level docstring\n"
        prompt += "2. Function/class docstrings (Google style)\n"
        prompt += "3. Parameter descriptions\n"
        prompt += "4. Return value descriptions\n"
        prompt += "5. Usage examples\n"
        prompt += "6. Notes about edge cases or limitations\n\n"

        prompt += "Google-Style Docstring Format:\n"
        prompt += '"""\n'
        prompt += "Brief description.\n\n"
        prompt += "Longer description if needed.\n\n"
        prompt += "Args:\n"
        prompt += "    param_name: Description\n\n"
        prompt += "Returns:\n"
        prompt += "    Description of return value\n\n"
        prompt += "Example:\n"
        prompt += "    >>> code_example()\n"
        prompt += '"""\n\n'

        prompt += "Output the code WITH documentation added.\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process documentation output."""
        if not llm_output or not llm_output.strip():
            return None

        code = llm_output.strip()

        # Remove markdown if present
        if code.startswith('```python'):
            code = code[9:]
        elif code.startswith('```'):
            code = code[3:]

        if code.endswith('```'):
            code = code[:-3]

        code = code.strip()

        if not code:
            return None

        return code

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate documentation output."""
        if output is None:
            return False

        if not isinstance(output, str):
            return False

        code = output.strip()

        if not code:
            return False

        # Must contain docstrings
        has_docstrings = '"""' in code or "'''" in code

        if not has_docstrings:
            return False

        # Should contain documentation keywords
        doc_keywords = ['Args:', 'Returns:', 'Example:', 'Parameters:', 'Description']
        has_doc_structure = any(keyword in code for keyword in doc_keywords)

        return has_doc_structure

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for documentation."""
        if output is None or not isinstance(output, str):
            return 0.0

        code = output.strip()
        score = 0.3  # Base score

        # Check for docstrings
        docstring_count = code.count('"""') + code.count("'''")
        if docstring_count >= 2:  # At least one complete docstring
            score += 0.2

        # Check for structured documentation
        if 'Args:' in code:
            score += 0.1
        if 'Returns:' in code:
            score += 0.1
        if 'Example:' in code or '>>>' in code:
            score += 0.15

        # Check for parameter documentation
        if 'param' in code.lower() or ':param' in code:
            score += 0.1

        # Check for adequate length (good docs are thorough)
        if len(code) >= 200:
            score += 0.05

        return min(1.0, max(0.0, score))
