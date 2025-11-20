"""
Tester Agent Archetype.

This agent specializes in generating comprehensive test suites including
unit tests, edge cases, and property-based tests.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class TesterAgent(BaseAgent):
    """
    Agent that generates comprehensive test suites.

    This agent creates unit tests, edge case tests, and property-based
    tests to thoroughly validate code correctness.

    Specialization:
    - Unit test generation
    - Edge case identification
    - Property-based test creation
    - Test coverage optimization
    """

    def __init__(self):
        """Initialize tester agent with appropriate configuration."""
        system_prompt = """You are an expert test engineer specializing in comprehensive testing.
Your role is to generate thorough test suites that catch bugs.

Guidelines:
1. Cover happy path, edge cases, and error conditions
2. Test boundary values
3. Include property-based tests where applicable
4. Use descriptive test names
5. Ensure tests are independent
6. Generate pytest-compatible code

Write tests that maximize bug detection."""

        super().__init__(
            agent_type="tester",
            system_prompt_template=system_prompt,
            default_temperature=0.6,
            default_max_tokens=768
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle testing tasks."""
        return task.task_type == TaskType.TESTING

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for test generation."""
        prompt = self.create_base_prompt(task, config)

        # Get code to test from context
        code_to_test = context_data.get('code', '')

        prompt += "\n\nCode to Test:\n"
        prompt += "```python\n"
        prompt += code_to_test
        prompt += "\n```\n\n"

        prompt += "Test Generation Instructions:\n"
        prompt += "Generate comprehensive pytest tests including:\n"
        prompt += "1. Happy path tests\n"
        prompt += "2. Edge cases (boundary values, empty inputs, etc.)\n"
        prompt += "3. Error conditions\n"
        prompt += "4. Property-based tests if applicable\n\n"

        if task.test_cases:
            prompt += "Required Test Cases:\n"
            for tc in task.test_cases:
                prompt += f"  - {tc.name}: {tc.description}\n"
                prompt += f"    Input: {tc.inputs}\n"
                prompt += f"    Expected: {tc.expected_output}\n"
            prompt += "\n"

        prompt += "Output pytest-compatible Python test code.\n"
        prompt += "Use assert statements, descriptive names, and clear structure.\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process test generation output."""
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
        """Validate generated tests."""
        if output is None:
            return False

        if not isinstance(output, str):
            return False

        code = output.strip()

        if not code:
            return False

        # Must contain test functions
        if 'def test_' not in code:
            return False

        # Must contain assertions
        if 'assert' not in code:
            return False

        # Should import pytest or unittest
        has_test_framework = any(
            framework in code
            for framework in ['import pytest', 'import unittest', 'from pytest']
        )

        return has_test_framework or 'def test_' in code

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for generated tests."""
        if output is None or not isinstance(output, str):
            return 0.0

        code = output.strip()
        score = 0.3  # Base score

        # Count test functions
        test_count = code.count('def test_')
        if test_count >= 3:
            score += 0.2
        elif test_count >= 1:
            score += 0.1

        # Check for assert statements
        assert_count = code.count('assert ')
        if assert_count >= 5:
            score += 0.2
        elif assert_count >= 2:
            score += 0.1

        # Check for edge case tests
        edge_case_indicators = ['empty', 'none', 'zero', 'negative', 'boundary']
        edge_case_count = sum(
            1 for indicator in edge_case_indicators
            if indicator in code.lower()
        )
        if edge_case_count >= 2:
            score += 0.15

        # Check for docstrings
        if '"""' in code:
            score += 0.05

        # Check for pytest features
        if '@pytest' in code or 'parametrize' in code:
            score += 0.1

        return min(1.0, max(0.0, score))
