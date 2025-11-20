"""
Optimizer Agent Archetype.

This agent specializes in optimizing code for performance, reducing complexity,
and improving efficiency while maintaining correctness.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class OptimizerAgent(BaseAgent):
    """
    Agent that optimizes code for performance and efficiency.

    This agent analyzes code and applies optimizations to improve
    performance, reduce complexity, and enhance efficiency.

    Specialization:
    - Performance optimization
    - Complexity reduction
    - Algorithm improvement
    - Resource efficiency
    """

    def __init__(self):
        """Initialize optimizer agent with appropriate configuration."""
        system_prompt = """You are an expert performance engineer specializing in code optimization.
Your role is to optimize code while maintaining correctness.

Optimization strategies:
1. Reduce time complexity where possible
2. Optimize space usage
3. Eliminate redundant operations
4. Use efficient data structures
5. Apply algorithmic improvements
6. Maintain readability

Focus on measurable improvements."""

        super().__init__(
            agent_type="optimizer",
            system_prompt_template=system_prompt,
            default_temperature=0.5,
            default_max_tokens=768
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle optimization tasks."""
        return task.task_type == TaskType.OPTIMIZATION

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for code optimization."""
        prompt = self.create_base_prompt(task, config)

        # Get code to optimize from context
        code_to_optimize = context_data.get('code', '')

        prompt += "\n\nCode to Optimize:\n"
        prompt += "```python\n"
        prompt += code_to_optimize
        prompt += "\n```\n\n"

        prompt += "Optimization Instructions:\n"
        prompt += "Optimize the code for:\n"
        prompt += "1. Time complexity improvement\n"
        prompt += "2. Space efficiency\n"
        prompt += "3. Reduced cyclomatic complexity\n"
        prompt += "4. Elimination of redundant operations\n\n"

        prompt += "Constraints:\n"
        prompt += "  - Maintain correctness (same output for all inputs)\n"
        prompt += "  - Keep or improve readability\n"
        prompt += f"  - Stay within {task.max_lines} lines\n"
        prompt += f"  - Target complexity <= {task.max_complexity}\n\n"

        if task.performance_req.time_complexity:
            prompt += f"Target time complexity: {task.performance_req.time_complexity}\n"

        if task.performance_req.space_complexity:
            prompt += f"Target space complexity: {task.performance_req.space_complexity}\n"

        prompt += "\nOutput Format:\n"
        prompt += "OPTIMIZED CODE:\n"
        prompt += "```python\n"
        prompt += "[optimized code here]\n"
        prompt += "```\n\n"
        prompt += "IMPROVEMENTS:\n"
        prompt += "  - [List specific optimizations made]\n\n"
        prompt += "COMPLEXITY:\n"
        prompt += "  Before: O(...) time, O(...) space\n"
        prompt += "  After: O(...) time, O(...) space\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process optimization output."""
        if not llm_output or not llm_output.strip():
            return None

        output = llm_output.strip()

        # Extract optimized code
        code_start = output.find('```python')
        if code_start != -1:
            code_start += 9
            code_end = output.find('```', code_start)
            if code_end != -1:
                code = output[code_start:code_end].strip()
            else:
                return None
        else:
            # Try to find code section
            if 'OPTIMIZED CODE:' in output:
                code_section = output.split('OPTIMIZED CODE:')[1]
                code_lines = []
                for line in code_section.split('\n'):
                    if line.strip() and not line.startswith('IMPROVEMENTS'):
                        code_lines.append(line)
                code = '\n'.join(code_lines).strip()
            else:
                return None

        if not code:
            return None

        # Extract improvements if present
        improvements = []
        if 'IMPROVEMENTS:' in output:
            imp_section = output.split('IMPROVEMENTS:')[1]
            if 'COMPLEXITY:' in imp_section:
                imp_section = imp_section.split('COMPLEXITY:')[0]
            improvements = [
                line.strip('- ').strip()
                for line in imp_section.split('\n')
                if line.strip().startswith('-')
            ]

        return {
            'code': code,
            'improvements': improvements,
            'full_output': output
        }

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate optimization output."""
        if output is None:
            return False

        if not isinstance(output, dict):
            return False

        if 'code' not in output:
            return False

        code = output['code']

        if not code or not code.strip():
            return False

        # Must look like actual code
        code_indicators = ['def ', 'return ', '=', 'if ', 'for ', 'while ']
        has_code = any(indicator in code for indicator in code_indicators)

        if not has_code:
            return False

        # Check line count
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) > task.max_lines:
            return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for optimization."""
        if output is None or not isinstance(output, dict):
            return 0.0

        score = 0.3  # Base score

        code = output.get('code', '')
        improvements = output.get('improvements', [])

        # Has improvements listed
        if len(improvements) > 0:
            score += 0.2

        if len(improvements) >= 3:
            score += 0.1

        # Check for complexity analysis
        full_output = output.get('full_output', '')
        if 'COMPLEXITY:' in full_output:
            score += 0.15

        if 'Before:' in full_output and 'After:' in full_output:
            score += 0.15

        # Code quality
        if code:
            # Check for efficient patterns
            efficient_patterns = [
                'set(', 'dict(', 'enumerate(', 'zip(',
                'list comprehension', '['
            ]
            pattern_count = sum(1 for pattern in efficient_patterns if pattern in code)
            score += min(0.1, pattern_count * 0.02)

        return min(1.0, max(0.0, score))
