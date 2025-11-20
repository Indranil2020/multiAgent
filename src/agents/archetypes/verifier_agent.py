"""
Verifier Agent Archetype.

This agent specializes in verifying code against specifications, contracts,
and quality requirements.
"""

from typing import Any, Dict

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class VerifierAgent(BaseAgent):
    """
    Agent that verifies code correctness and quality.

    This agent analyzes code to check if it satisfies all specifications,
    contracts, test cases, and quality requirements.

    Specialization:
    - Contract verification
    - Test case validation
    - Quality metrics assessment
    - Specification compliance
    """

    def __init__(self):
        """Initialize verifier agent with appropriate configuration."""
        system_prompt = """You are an expert code verifier and quality analyst.
Your role is to rigorously verify code against specifications.

Check for:
1. Precondition and postcondition satisfaction
2. Test case compliance
3. Quality metrics (complexity, line count)
4. Type safety
5. Edge case handling
6. Specification adherence

Provide detailed verification results."""

        super().__init__(
            agent_type="verifier",
            system_prompt_template=system_prompt,
            default_temperature=0.2,  # Low temperature for consistency
            default_max_tokens=768
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """Check if this agent can handle verification tasks."""
        return task.task_type == TaskType.VERIFICATION

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """Generate prompt for code verification."""
        prompt = self.create_base_prompt(task, config)

        # Get code to verify from context
        code_to_verify = context_data.get('code', '')

        prompt += "\n\nCode to Verify:\n"
        prompt += "```python\n"
        prompt += code_to_verify
        prompt += "\n```\n\n"

        prompt += "Verification Instructions:\n"
        prompt += "Analyze the code and verify:\n"
        prompt += "1. All preconditions are checked\n"
        prompt += "2. All postconditions are guaranteed\n"
        prompt += "3. All test cases would pass\n"
        prompt += f"4. Code is within {task.max_lines} lines\n"
        prompt += f"5. Cyclomatic complexity <= {task.max_complexity}\n"
        prompt += "6. Type annotations are present\n"
        prompt += "7. Edge cases are handled\n\n"

        prompt += "Output Format:\n"
        prompt += "VERDICT: [PASS/FAIL]\n"
        prompt += "CHECKS:\n"
        prompt += "  - Preconditions: [PASS/FAIL] - reasoning\n"
        prompt += "  - Postconditions: [PASS/FAIL] - reasoning\n"
        prompt += "  - Test Cases: [PASS/FAIL] - reasoning\n"
        prompt += "  - Line Count: [PASS/FAIL] - actual count\n"
        prompt += "  - Complexity: [PASS/FAIL] - estimated complexity\n"
        prompt += "  - Type Safety: [PASS/FAIL] - reasoning\n"
        prompt += "  - Edge Cases: [PASS/FAIL] - reasoning\n"
        prompt += "CONFIDENCE: [0-100]%\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """Process verification output."""
        if not llm_output or not llm_output.strip():
            return None

        output = llm_output.strip()

        # Extract verdict
        verdict_pass = 'VERDICT: PASS' in output
        verdict_fail = 'VERDICT: FAIL' in output

        if not (verdict_pass or verdict_fail):
            return None

        # Extract confidence if present
        confidence = 50.0
        if 'CONFIDENCE:' in output:
            conf_line = [line for line in output.split('\n') if 'CONFIDENCE:' in line]
            if conf_line:
                conf_text = conf_line[0].split('CONFIDENCE:')[1].strip()
                # Extract number
                conf_num = ''.join(filter(lambda x: x.isdigit(), conf_text))
                if conf_num:
                    confidence = min(100.0, max(0.0, float(conf_num)))

        return {
            'verdict': 'PASS' if verdict_pass else 'FAIL',
            'confidence': confidence / 100.0,
            'full_output': output
        }

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """Validate verification output."""
        if output is None:
            return False

        if not isinstance(output, dict):
            return False

        if 'verdict' not in output:
            return False

        if output['verdict'] not in ['PASS', 'FAIL']:
            return False

        if 'confidence' not in output:
            return False

        if not (0.0 <= output['confidence'] <= 1.0):
            return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """Calculate quality score for verification."""
        if output is None or not isinstance(output, dict):
            return 0.0

        # Base score
        score = 0.3

        # Confidence in verification
        confidence = output.get('confidence', 0.0)
        score += confidence * 0.4

        # Completeness of checks
        full_output = output.get('full_output', '')
        if 'Preconditions:' in full_output:
            score += 0.05
        if 'Postconditions:' in full_output:
            score += 0.05
        if 'Test Cases:' in full_output:
            score += 0.05
        if 'Line Count:' in full_output:
            score += 0.05
        if 'Complexity:' in full_output:
            score += 0.05
        if 'Type Safety:' in full_output:
            score += 0.05

        return min(1.0, max(0.0, score))
