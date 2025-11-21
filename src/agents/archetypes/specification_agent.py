"""
Specification Agent Archetype.

This agent specializes in converting natural language task descriptions into
formal JSON task specifications that can be executed by the swarm.
"""

from typing import Any, Dict, List, Optional
import json
import uuid
from datetime import datetime

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType
from ...core.voting.types import AgentConfig


class SpecificationAgent(BaseAgent):
    """
    Agent that converts natural language descriptions to formal task specifications.

    This agent analyzes user descriptions and generates structured JSON
    task specifications with inputs, outputs, preconditions, postconditions,
    test cases, and constraints.

    Specialization:
    - Natural language understanding
    - Requirement extraction
    - Test case generation
    - Constraint inference
    """

    def __init__(self):
        """Initialize specification agent with appropriate configuration."""
        system_prompt = """You are an expert requirements analyst and software architect.
Your role is to convert natural language task descriptions into formal, structured task specifications.

Guidelines:
1. Extract clear inputs and outputs from the description
2. Infer reasonable preconditions and postconditions
3. Generate at least 3 test cases (including edge cases)
4. Set reasonable constraints (complexity, lines, timeout)
5. Identify the task type (function, class, module, api)
6. Output valid JSON following the exact schema provided

Focus on completeness, clarity, and generating executable specifications."""

        super().__init__(
            agent_type="specification",
            system_prompt_template=system_prompt,
            default_temperature=0.4,  # Moderate temperature for structured creativity
            default_max_tokens=2048
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """
        Check if this agent can handle specification tasks.

        Args:
            task: Task specification

        Returns:
            True if task is a specification task
        """
        return task.task_type == TaskType.SPECIFICATION

    def generate_specification_from_description(
        self,
        description: str,
        llm_pool,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate formal task specification from natural language description.

        Args:
            description: Natural language task description
            llm_pool: LLM pool for generation
            additional_context: Optional additional context

        Returns:
            Task specification as dictionary or None if generation fails
        """
        # Create prompt for specification generation
        prompt = self._create_specification_prompt(description, additional_context)

        # Generate specification using LLM
        result = llm_pool.generate(
            prompt=prompt,
            model_name=None,  # Use default model
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
            stop_sequences=None
        )

        # Handle both string and object returns from LLM pool
        llm_output = None
        if result:
            if isinstance(result, str):
                llm_output = result
            elif hasattr(result, 'text'):
                llm_output = result.text
        
        if not llm_output:
            return None

        # Parse and validate the generated specification
        spec_dict = self._parse_specification_output(llm_output)
        
        if spec_dict and self._validate_specification(spec_dict):
            return spec_dict
        
        return None

    def _create_specification_prompt(
        self,
        description: str,
        additional_context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Create prompt for specification generation.

        Args:
            description: User's task description
            additional_context: Optional additional context

        Returns:
            Formatted prompt
        """
        prompt = f"""Given this task description from a user:

"{description}"
"""

        if additional_context:
            prompt += "\nAdditional Context:\n"
            for key, value in additional_context.items():
                prompt += f"  - {key}: {value}\n"

        prompt += """
Generate a formal task specification following this exact JSON schema:

{
  "task_id": "auto-generated-unique-id",
  "name": "Clear, concise task name",
  "description": "Detailed description of what to create",
  "task_type": "function",  // or "class", "module", "api"
  "inputs": [
    {
      "name": "parameter_name",
      "type": "python_type",
      "description": "What this parameter does"
    }
  ],
  "outputs": [
    {
      "name": "return_value",
      "type": "python_type",
      "description": "What gets returned"
    }
  ],
  "preconditions": [
    "Condition that must be true before execution"
  ],
  "postconditions": [
    "Condition that must be true after execution"
  ],
  "test_cases": [
    {
      "input": {"param_name": value},
      "expected_output": {"return_name": expected_value}
    }
  ],
  "constraints": {
    "max_complexity": 10,
    "max_lines": 50,
    "timeout_ms": 5000
  },
  "hints": [
    "Helpful hint for implementation"
  ]
}

Requirements:
1. Generate a unique task_id (use format: "task-{timestamp}-{random}")
2. Infer inputs/outputs from the description
3. Create at least 3 test cases (including edge cases)
4. Set reasonable constraints based on task complexity
5. Add helpful hints if applicable
6. Ensure all fields are present and valid

Provide ONLY the JSON output, no additional text or explanation.
"""

        return prompt

    def _parse_specification_output(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM output into specification dictionary.

        Args:
            llm_output: Raw output from LLM

        Returns:
            Parsed specification or None if parsing fails
        """
        if not llm_output or not llm_output.strip():
            return None

        # Extract JSON from output
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}')

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = llm_output[json_start:json_end + 1]

        # Parse JSON
        parsed = json.loads(json_str)
        
        # Ensure task_id is present and unique
        if 'task_id' not in parsed or not parsed['task_id']:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            random_suffix = str(uuid.uuid4())[:8]
            parsed['task_id'] = f"task-{timestamp}-{random_suffix}"

        return parsed

    def _validate_specification(self, spec: Dict[str, Any]) -> bool:
        """
        Validate specification structure.

        Args:
            spec: Specification dictionary

        Returns:
            True if valid
        """
        if not isinstance(spec, dict):
            return False

        # Required fields
        required_fields = ['task_id', 'name', 'description', 'task_type']
        for field in required_fields:
            if field not in spec:
                return False
            if not spec[field]:
                return False

        # Validate task_type
        valid_types = ['function', 'class', 'module', 'api', 'code_generation']
        if spec.get('task_type') not in valid_types:
            return False

        # Validate inputs if present
        if 'inputs' in spec:
            if not isinstance(spec['inputs'], list):
                return False
            for inp in spec['inputs']:
                if not isinstance(inp, dict):
                    return False
                if 'name' not in inp or 'type' not in inp:
                    return False

        # Validate outputs if present
        if 'outputs' in spec:
            if not isinstance(spec['outputs'], list):
                return False
            for out in spec['outputs']:
                if not isinstance(out, dict):
                    return False
                if 'name' not in out or 'type' not in out:
                    return False

        # Validate test_cases if present
        if 'test_cases' in spec:
            if not isinstance(spec['test_cases'], list):
                return False

        return True

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for specification task.

        Args:
            task: Task specification
            config: Agent configuration
            context_data: Additional context

        Returns:
            Formatted prompt
        """
        # For specification tasks, extract description from task
        description = task.description or ""
        return self._create_specification_prompt(description, context_data)

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """
        Process LLM output into structured specification.

        Args:
            llm_output: Raw output from LLM
            task: Original task

        Returns:
            Parsed specification or None
        """
        return self._parse_specification_output(llm_output)

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """
        Validate the specification output.

        Args:
            output: Parsed specification
            task: Original task

        Returns:
            True if valid
        """
        if not isinstance(output, dict):
            return False
        return self._validate_specification(output)

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """
        Calculate quality score for specification.

        Args:
            output: Specification dictionary
            task: Original task

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not output or not isinstance(output, dict):
            return 0.0

        score = 0.0

        # Base score for valid structure
        if self._validate_specification(output):
            score += 0.3

        # Score for completeness
        if 'inputs' in output and output['inputs']:
            score += 0.1
        if 'outputs' in output and output['outputs']:
            score += 0.1
        if 'preconditions' in output and output['preconditions']:
            score += 0.1
        if 'postconditions' in output and output['postconditions']:
            score += 0.1

        # Score for test cases
        if 'test_cases' in output:
            num_tests = len(output['test_cases'])
            if num_tests >= 3:
                score += 0.2
            elif num_tests >= 1:
                score += 0.1

        # Score for constraints
        if 'constraints' in output and isinstance(output['constraints'], dict):
            score += 0.1

        return min(1.0, max(0.0, score))
