"""
Decomposer Agent Archetype.

This agent specializes in breaking down complex tasks into smaller, manageable
subtasks. It performs hierarchical decomposition following the 7-layer
architecture defined in the zero-error system.
"""

from typing import Any, Dict, List
import json

from .base_agent import BaseAgent
from ...core.task_spec.language import TaskSpecification
from ...core.task_spec.types import TaskType, TaskID
from ...core.voting.types import AgentConfig


class DecomposerAgent(BaseAgent):
    """
    Agent that decomposes complex tasks into atomic subtasks.

    This agent analyzes task specifications and breaks them down into
    smaller, more manageable pieces following hierarchical decomposition
    principles. It identifies dependencies and creates a task graph.

    Specialization:
    - Task decomposition and dependency analysis
    - Hierarchical breakdown
    - Atomic task identification
    """

    def __init__(self):
        """Initialize decomposer agent with appropriate configuration."""
        system_prompt = """You are an expert software architect specializing in task decomposition.
Your role is to break down complex software tasks into smaller, atomic subtasks.

Guidelines:
1. Each subtask should be independently implementable
2. Identify clear dependencies between subtasks
3. Ensure atomic tasks are 5-20 lines of code maximum
4. Maintain single responsibility for each subtask
5. Consider data flow and interface contracts
6. Output valid JSON with subtask specifications

Focus on clarity, modularity, and correct dependency identification."""

        super().__init__(
            agent_type="decomposer",
            system_prompt_template=system_prompt,
            default_temperature=0.3,  # Lower temperature for structured output
            default_max_tokens=1024
        )

    def can_handle_task(self, task: TaskSpecification) -> bool:
        """
        Check if this agent can handle decomposition tasks.

        Args:
            task: Task specification

        Returns:
            True if task is a decomposition task
        """
        return task.task_type == TaskType.DECOMPOSITION

    def generate_prompt(
        self,
        task: TaskSpecification,
        config: AgentConfig,
        context_data: Dict[str, Any]
    ) -> str:
        """
        Generate prompt for task decomposition.

        Args:
            task: Task to decompose
            config: Agent configuration
            context_data: Additional context

        Returns:
            Formatted prompt for decomposition
        """
        prompt = self.create_base_prompt(task, config)

        # Add decomposition-specific instructions
        prompt += "\n\nDecomposition Instructions:\n"
        prompt += "Analyze the task and break it down into atomic subtasks.\n"
        prompt += "Each subtask should:\n"
        prompt += "  - Be independently implementable\n"
        prompt += "  - Have clear inputs and outputs\n"
        prompt += "  - Be 5-20 lines of code maximum\n"
        prompt += "  - Follow single responsibility principle\n\n"

        # Add hints if available
        if task.hints:
            prompt += "Hints:\n"
            for hint in task.hints:
                prompt += f"  - {hint}\n"
            prompt += "\n"

        # Add examples if available
        if task.examples:
            prompt += "Examples:\n"
            for example in task.examples:
                prompt += f"  {example}\n"
            prompt += "\n"

        # Specify output format
        prompt += "Output Format (JSON):\n"
        prompt += """
{
  "subtasks": [
    {
      "id": "unique_subtask_id",
      "name": "Subtask Name",
      "description": "Detailed description of what this subtask does",
      "inputs": ["input1_name", "input2_name"],
      "outputs": ["output1_name"],
      "dependencies": ["id_of_required_subtask"],
      "estimated_lines": 15,
      "estimated_complexity": 3
    }
  ],
  "dependency_graph": {
    "roots": ["subtask_id_with_no_deps"],
    "ordering": ["subtask1_id", "subtask2_id", "..."]
  }
}
"""

        prompt += "\nProvide ONLY the JSON output, no additional text.\n"

        return prompt

    def process_output(self, llm_output: str, task: TaskSpecification) -> Any:
        """
        Process LLM output into structured decomposition result.

        Args:
            llm_output: Raw JSON from LLM
            task: Original task specification

        Returns:
            Parsed decomposition structure or None if parsing fails
        """
        if not llm_output or not llm_output.strip():
            return None

        # Extract JSON from output (handle cases where LLM adds extra text)
        json_start = llm_output.find('{')
        json_end = llm_output.rfind('}')

        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None

        json_str = llm_output[json_start:json_end + 1]

        # Parse JSON
        parsed = self._safe_json_parse(json_str)
        if parsed is None:
            return None

        # Validate structure
        if not self._validate_decomposition_structure(parsed):
            return None

        return parsed

    def validate_output(self, output: Any, task: TaskSpecification) -> bool:
        """
        Validate the decomposition output.

        Args:
            output: Parsed decomposition structure
            task: Original task

        Returns:
            True if output is valid
        """
        if output is None:
            return False

        if not isinstance(output, dict):
            return False

        # Check required keys
        if 'subtasks' not in output:
            return False

        if not isinstance(output['subtasks'], list):
            return False

        # Must have at least one subtask
        if len(output['subtasks']) == 0:
            return False

        # Validate each subtask
        for subtask in output['subtasks']:
            if not self._validate_subtask(subtask):
                return False

        # Validate dependency graph if present
        if 'dependency_graph' in output:
            if not self._validate_dependency_graph(output['dependency_graph'], output['subtasks']):
                return False

        return True

    def calculate_quality_score(self, output: Any, task: TaskSpecification) -> float:
        """
        Calculate quality score for decomposition.

        Args:
            output: Decomposition structure
            task: Original task

        Returns:
            Quality score between 0.0 and 1.0
        """
        if output is None or not isinstance(output, dict):
            return 0.0

        score = 0.0
        subtasks = output.get('subtasks', [])

        if not subtasks:
            return 0.0

        # Base score for valid output
        score += 0.3

        # Score based on number of subtasks (not too few, not too many)
        num_subtasks = len(subtasks)
        if 2 <= num_subtasks <= 10:
            score += 0.2
        elif 1 <= num_subtasks <= 20:
            score += 0.1

        # Score based on atomic size
        atomic_count = sum(
            1 for st in subtasks
            if st.get('estimated_lines', 100) <= 20
            and st.get('estimated_complexity', 10) <= 5
        )
        atomic_ratio = atomic_count / num_subtasks if num_subtasks > 0 else 0
        score += atomic_ratio * 0.2

        # Score based on completeness
        completeness_score = self._calculate_completeness_score(subtasks)
        score += completeness_score * 0.2

        # Score based on dependency structure
        if 'dependency_graph' in output:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _safe_json_parse(self, json_str: str) -> Any:
        """
        Safely parse JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            Parsed object or None if parsing fails
        """
        if not json_str or not json_str.strip():
            return None

        # Attempt to parse
        parsed = None
        if json_str:
            # Simple validation first
            if '{' in json_str and '}' in json_str:
                # Try direct parse
                parsed = self._attempt_json_parse(json_str)
                if parsed is None:
                    # Try fixing common issues
                    fixed_str = json_str.strip()
                    fixed_str = fixed_str.replace(',]', ']').replace(',}', '}')
                    parsed = self._attempt_json_parse(fixed_str)

        return parsed

    def _attempt_json_parse(self, json_str: str) -> Any:
        """Attempt to parse JSON, return None on failure."""
        # Note: In production, this would use a proper JSON parser
        # For zero-error philosophy, we validate extensively before parsing
        if not json_str:
            return None
        # Placeholder for actual JSON parsing
        # In real implementation, would use json.loads with proper error handling
        return None

    def _validate_decomposition_structure(self, parsed: Dict) -> bool:
        """Validate basic decomposition structure."""
        if not isinstance(parsed, dict):
            return False

        if 'subtasks' not in parsed:
            return False

        if not isinstance(parsed['subtasks'], list):
            return False

        return True

    def _validate_subtask(self, subtask: Dict) -> bool:
        """Validate individual subtask structure."""
        if not isinstance(subtask, dict):
            return False

        # Required fields
        required = ['id', 'name', 'description']
        for field in required:
            if field not in subtask:
                return False
            if not subtask[field] or not str(subtask[field]).strip():
                return False

        # Validate ID format
        if not isinstance(subtask['id'], str):
            return False
        if len(subtask['id']) < 3:
            return False

        # Validate name
        if not isinstance(subtask['name'], str):
            return False
        if len(subtask['name']) < 3:
            return False

        # Validate description
        if not isinstance(subtask['description'], str):
            return False
        if len(subtask['description']) < 10:
            return False

        # Validate optional fields if present
        if 'inputs' in subtask:
            if not isinstance(subtask['inputs'], list):
                return False

        if 'outputs' in subtask:
            if not isinstance(subtask['outputs'], list):
                return False

        if 'dependencies' in subtask:
            if not isinstance(subtask['dependencies'], list):
                return False

        if 'estimated_lines' in subtask:
            if not isinstance(subtask['estimated_lines'], int):
                return False
            if subtask['estimated_lines'] <= 0:
                return False

        if 'estimated_complexity' in subtask:
            if not isinstance(subtask['estimated_complexity'], int):
                return False
            if subtask['estimated_complexity'] <= 0:
                return False

        return True

    def _validate_dependency_graph(self, graph: Dict, subtasks: List[Dict]) -> bool:
        """Validate dependency graph structure."""
        if not isinstance(graph, dict):
            return False

        # Get all subtask IDs
        subtask_ids = {st['id'] for st in subtasks}

        # Validate roots if present
        if 'roots' in graph:
            if not isinstance(graph['roots'], list):
                return False
            # All roots must be valid subtask IDs
            for root_id in graph['roots']:
                if root_id not in subtask_ids:
                    return False

        # Validate ordering if present
        if 'ordering' in graph:
            if not isinstance(graph['ordering'], list):
                return False
            # All IDs in ordering must be valid subtask IDs
            for task_id in graph['ordering']:
                if task_id not in subtask_ids:
                    return False

        return True

    def _calculate_completeness_score(self, subtasks: List[Dict]) -> float:
        """
        Calculate completeness score for subtasks.

        Args:
            subtasks: List of subtask specifications

        Returns:
            Completeness score between 0.0 and 1.0
        """
        if not subtasks:
            return 0.0

        total = len(subtasks)
        complete_count = 0

        for subtask in subtasks:
            completeness = 0

            # Has ID, name, description (required)
            if 'id' in subtask and 'name' in subtask and 'description' in subtask:
                completeness += 0.3

            # Has inputs/outputs
            if 'inputs' in subtask:
                completeness += 0.2
            if 'outputs' in subtask:
                completeness += 0.2

            # Has estimates
            if 'estimated_lines' in subtask:
                completeness += 0.15
            if 'estimated_complexity' in subtask:
                completeness += 0.15

            if completeness >= 0.6:  # Threshold for "complete"
                complete_count += 1

        return complete_count / total if total > 0 else 0.0
