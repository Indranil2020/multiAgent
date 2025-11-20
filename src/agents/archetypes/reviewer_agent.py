"""
Reviewer Agent Archetype.

This agent specializes in code review, identifying quality issues,
suggesting improvements, and ensuring best practices.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class ReviewerAgent(BaseAgent):
    """
    Agent that performs comprehensive code reviews.

    This agent analyzes code for quality, maintainability, best practices,
    and potential improvements.

    Specialization:
    - Code quality assessment
    - Best practice validation
    - Improvement suggestions
    - Maintainability analysis
    """

    def __init__(self):
        """Initialize reviewer agent with appropriate configuration."""
        system_prompt = """You are an expert code reviewer focusing on quality and best practices.
Your role is to provide constructive, actionable code reviews.

Review for:
1. Code clarity and readability
2. Best practice adherence
3. Potential bugs or issues
4. Performance concerns
5. Maintainability
6. Documentation quality

Provide specific, actionable feedback."""

        super().__init__(
            agent_type="reviewer",
            system_prompt_template=system_prompt,
            default_temperature=0.4,
            default_max_tokens=768
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle review tasks."""
        return task.task_type == TaskType.REVIEW

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for code review."""
        prompt = self.create_base_prompt(task, config)

        # Get code to review from context
        code_to_review = context_data.get('code', '')

        prompt += "\n\nCode to Review:\n"
        prompt += "```python\n"
        prompt += code_to_review
        prompt += "\n```\n\n"

        prompt += "Review Instructions:\n"
        prompt += "Provide a comprehensive code review covering:\n"
        prompt += "1. Overall quality assessment\n"
        prompt += "2. Specific issues found\n"
        prompt += "3. Best practice violations\n"
        prompt += "4. Suggested improvements\n"
        prompt += "5. Positive aspects (what's done well)\n\n"

        prompt += "Quality Metrics to Check:\n"
        prompt += f"  - Cyclomatic complexity (target: <= {task.max_complexity})\n"
        prompt += f"  - Line count (target: <= {task.max_lines})\n"
        prompt += "  - Type annotations present\n"
        prompt += "  - Docstrings present\n"
        prompt += "  - Clear variable names\n\n"

        prompt += "Output Format:\n"
        prompt += "OVERALL: [APPROVE/REQUEST_CHANGES/REJECT]\n"
        prompt += "SCORE: [0-100]\n\n"
        prompt += "STRENGTHS:\n"
        prompt += "  - List positive aspects\n\n"
        prompt += "ISSUES:\n"
        prompt += "  - List specific issues with line numbers if applicable\n\n"
        prompt += "SUGGESTIONS:\n"
        prompt += "  - List actionable improvements\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process code review output."""
        if not llm_output or not llm_output.strip():
            return None

        output = llm_output.strip()

        # Extract overall assessment
        overall_status = 'REQUEST_CHANGES'  # Default
        if 'OVERALL: APPROVE' in output:
            overall_status = 'APPROVE'
        elif 'OVERALL: REJECT' in output:
            overall_status = 'REJECT'

        # Extract score
        score = 50  # Default
        if 'SCORE:' in output:
            score_line = [line for line in output.split('\n') if 'SCORE:' in line]
            if score_line:
                score_text = score_line[0].split('SCORE:')[1].strip()
                score_num = ''.join(filter(lambda x: x.isdigit(), score_text))
                if score_num:
                    score = min(100, max(0, int(score_num)))

        # Count issues and suggestions
        issues_count = output.count('- ') if 'ISSUES:' in output else 0
        suggestions_count = output.count('- ') if 'SUGGESTIONS:' in output else 0

        return {
            'status': overall_status,
            'score': score / 100.0,
            'issues_count': issues_count,
            'suggestions_count': suggestions_count,
            'full_review': output
        }

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate review output."""
        if output is None:
            return False

        if not isinstance(output, dict):
            return False

        if 'status' not in output:
            return False

        if output['status'] not in ['APPROVE', 'REQUEST_CHANGES', 'REJECT']:
            return False

        if 'score' not in output:
            return False

        if not (0.0 <= output['score'] <= 1.0):
            return False

        if 'full_review' not in output:
            return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for code review."""
        if output is None or not isinstance(output, dict):
            return 0.0

        score = 0.2  # Base score

        # Review completeness
        full_review = output.get('full_review', '')

        if 'STRENGTHS:' in full_review:
            score += 0.15
        if 'ISSUES:' in full_review:
            score += 0.15
        if 'SUGGESTIONS:' in full_review:
            score += 0.15

        # Has specific feedback
        issues_count = output.get('issues_count', 0)
        suggestions_count = output.get('suggestions_count', 0)

        if issues_count > 0 or suggestions_count > 0:
            score += 0.2

        # Clear decision
        if output.get('status') in ['APPROVE', 'REQUEST_CHANGES', 'REJECT']:
            score += 0.15

        return min(1.0, max(0.0, score))
