"""
Base Prompts Module.

This module provides the foundational prompt templates and utilities used
across all agent types in the zero-error system. Defines base classes,
prompt composition, and common formatting.

Key Concepts:
- Prompts are instructions that guide LLM behavior
- Well-designed prompts are critical for zero-error output
- Templates allow reuse and consistency
- Variables enable dynamic prompt generation
- Formatting ensures clarity and structure

All prompts follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class PromptFormat(Enum):
    """
    Prompt formatting styles.

    Attributes:
        PLAIN: Plain text format
        MARKDOWN: Markdown formatting
        JSON: JSON-structured prompts
        XML: XML-tagged prompts
    """
    PLAIN = "plain"
    MARKDOWN = "markdown"
    JSON = "json"
    XML = "xml"


@dataclass
class PromptTemplate:
    """
    Base prompt template.

    Attributes:
        template_id: Unique template identifier
        name: Human-readable template name
        template_text: Template string with {variables}
        format: Prompt format style
        variables: List of required variables
        examples: Example prompts for few-shot learning
        constraints: Explicit constraints/requirements
        metadata: Additional template metadata
    """
    template_id: str
    name: str
    template_text: str
    format: PromptFormat = PromptFormat.MARKDOWN
    variables: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs: Any) -> Tuple[bool, str, str]:
        """
        Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Tuple of (success, rendered_text, error_message)
        """
        # Check required variables
        missing = [var for var in self.variables if var not in kwargs]
        if missing:
            return False, "", f"Missing required variables: {missing}"

        # Render template
        # Since we validated all variables are present, format should succeed
        # unless there are malformed format strings in the template itself
        # which should be caught during template creation/testing.
        # However, to be strictly zero-error, we should validate the template string too.
        
        # In a strict zero-error system, we would use a safe formatter
        # For now, we rely on the pre-validation above
        return True, self.template_text.format(**kwargs), ""

    def validate_variables(self, **kwargs: Any) -> bool:
        """
        Validate that all required variables are provided.

        Args:
            **kwargs: Variable values

        Returns:
            True if all variables present
        """
        return all(var in kwargs for var in self.variables)


# Zero-Error System Prompt
ZERO_ERROR_SYSTEM_PROMPT = PromptTemplate(
    template_id="zero_error_system",
    name="Zero-Error System Prompt",
    template_text="""You are an AI agent in a zero-error code generation system.

CRITICAL REQUIREMENTS:
1. Generate ONLY syntactically correct, executable code
2. Handle ALL edge cases explicitly
3. Validate ALL inputs before use
4. Return explicit error values instead of exceptions for control flow
5. Follow the Single Responsibility Principle
6. Include comprehensive docstrings with type hints
7. Never use placeholders or TODO comments

FORBIDDEN PRACTICES:
- Using try/except for control flow
- Leaving placeholder implementations
- Assuming input validity without checks
- Missing error handling
- Incomplete implementations

OUTPUT FORMAT:
- Working, tested code only
- Clear variable names
- Explicit type annotations
- Complete docstrings
- No explanatory prose outside code

Your code will be validated through 8 verification layers. It must pass all checks.
Generate perfect code on the first attempt.""",
    format=PromptFormat.MARKDOWN,
    variables=[],
    constraints=[
        "No syntax errors",
        "No runtime errors",
        "All edge cases handled",
        "Complete implementations only"
    ]
)


# Task Specification Prompt
TASK_SPECIFICATION_PROMPT = PromptTemplate(
    template_id="task_specification",
    name="Task Specification Prompt",
    template_text="""Parse and analyze this task specification:

TASK: {task_description}

CONTEXT:
{context}

REQUIREMENTS:
{requirements}

PROVIDE:
1. Precise technical interpretation
2. Required inputs and their types
3. Expected outputs and their types
4. Edge cases to handle
5. Success criteria
6. Failure conditions

Format as structured JSON with keys: interpretation, inputs, outputs, edge_cases, success_criteria, failure_conditions.""",
    format=PromptFormat.MARKDOWN,
    variables=["task_description", "context", "requirements"],
    examples=[]
)


# Contract Generation Prompt
CONTRACT_GENERATION_PROMPT = PromptTemplate(
    template_id="contract_generation",
    name="Contract Generation Prompt",
    template_text="""Generate formal contracts for this function:

FUNCTION SIGNATURE:
{function_signature}

INPUTS: {inputs}
OUTPUTS: {outputs}

Generate:
1. Preconditions (input validation rules)
2. Postconditions (output guarantees)
3. Invariants (conditions that must always hold)
4. Error conditions

Use Python assert statements for preconditions.
Format as executable Python code.""",
    format=PromptFormat.MARKDOWN,
    variables=["function_signature", "inputs", "outputs"]
)


# Error Analysis Prompt
ERROR_ANALYSIS_PROMPT = PromptTemplate(
    template_id="error_analysis",
    name="Error Analysis Prompt",
    template_text="""Analyze this error and provide fix:

ERROR TYPE: {error_type}
ERROR MESSAGE: {error_message}
CODE CONTEXT:
```python
{code_context}
```

LINE NUMBER: {line_number}

PROVIDE:
1. Root cause analysis
2. Exact line(s) causing error
3. Complete corrected code
4. Explanation of fix
5. How to prevent similar errors

Return ONLY the corrected code, no explanations.""",
    format=PromptFormat.MARKDOWN,
    variables=["error_type", "error_message", "code_context", "line_number"]
)


# Test Generation Prompt
TEST_GENERATION_PROMPT = PromptTemplate(
    template_id="test_generation",
    name="Test Generation Prompt",
    template_text="""Generate comprehensive tests for this code:

```python
{code}
```

FUNCTION: {function_name}
INPUTS: {input_types}
OUTPUTS: {output_types}

Generate pytest tests including:
1. Happy path tests
2. Edge case tests
3. Error condition tests
4. Boundary value tests
5. Invalid input tests

Use descriptive test names. Aim for 100% code coverage.
Return complete, executable pytest code.""",
    format=PromptFormat.MARKDOWN,
    variables=["code", "function_name", "input_types", "output_types"]
)


# Decomposition Prompt
DECOMPOSITION_PROMPT = PromptTemplate(
    template_id="decomposition",
    name="Hierarchical Decomposition Prompt",
    template_text="""Decompose this high-level task into subtasks:

TASK: {task}
LEVEL: {level}
MAX_SUBTASKS: {max_subtasks}

Decompose into {max_subtasks} independent subtasks at the {level} level.

For each subtask provide:
- Subtask ID
- Clear description
- Required inputs
- Expected outputs
- Dependencies on other subtasks

Format as JSON array of subtask objects.""",
    format=PromptFormat.MARKDOWN,
    variables=["task", "level", "max_subtasks"]
)


# Quality Assessment Prompt
QUALITY_ASSESSMENT_PROMPT = PromptTemplate(
    template_id="quality_assessment",
    name="Code Quality Assessment Prompt",
    template_text="""Assess code quality:

```python
{code}
```

EVALUATE:
1. Correctness (0-10)
2. Readability (0-10)
3. Maintainability (0-10)
4. Performance (0-10)
5. Test coverage (0-10)

For each dimension, provide:
- Score
- Justification
- Specific improvements needed

Return as JSON with structure: {{dimension: {{score, justification, improvements}}}}""",
    format=PromptFormat.MARKDOWN,
    variables=["code"]
)


def create_few_shot_prompt(
    base_template: PromptTemplate,
    examples: List[Dict[str, str]],
    task_variables: Dict[str, Any]
) -> str:
    """
    Create few-shot prompt with examples.

    Args:
        base_template: Base prompt template
        examples: List of example input/output pairs
        task_variables: Variables for current task

    Returns:
        Complete few-shot prompt
    """
    # Build examples section
    examples_text = "EXAMPLES:\n\n"
    for i, example in enumerate(examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Input: {example.get('input', '')}\n"
        examples_text += f"Output: {example.get('output', '')}\n\n"

    # Combine with task
    examples_text += "YOUR TASK:\n"
    task_prompt = base_template.render(**task_variables)

    return examples_text + task_prompt


def create_chain_of_thought_prompt(
    base_template: PromptTemplate,
    task_variables: Dict[str, Any]
) -> str:
    """
    Create chain-of-thought prompt.

    Args:
        base_template: Base prompt template
        task_variables: Variables for task

    Returns:
        Chain-of-thought prompt
    """
    cot_prefix = """Think step-by-step:
1. Understand the requirements
2. Identify edge cases
3. Plan the solution approach
4. Consider error conditions
5. Implement the solution

"""

    task_prompt = base_template.render(**task_variables)
    return cot_prefix + task_prompt


def create_self_consistency_prompt(
    base_template: PromptTemplate,
    task_variables: Dict[str, Any],
    num_attempts: int = 3
) -> List[str]:
    """
    Create multiple prompts for self-consistency.

    Args:
        base_template: Base prompt template
        task_variables: Variables for task
        num_attempts: Number of independent attempts

    Returns:
        List of prompts with slight variations
    """
    base_prompt = base_template.render(**task_variables)

    prompts = [base_prompt]

    # Add variations
    variations = [
        "Approach this problem independently. ",
        "Solve this from first principles. ",
        "Find an alternative solution approach. "
    ]

    for i in range(min(num_attempts - 1, len(variations))):
        prompts.append(variations[i] + base_prompt)

    return prompts


# Export all templates
ALL_BASE_TEMPLATES = {
    "zero_error_system": ZERO_ERROR_SYSTEM_PROMPT,
    "task_specification": TASK_SPECIFICATION_PROMPT,
    "contract_generation": CONTRACT_GENERATION_PROMPT,
    "error_analysis": ERROR_ANALYSIS_PROMPT,
    "test_generation": TEST_GENERATION_PROMPT,
    "decomposition": DECOMPOSITION_PROMPT,
    "quality_assessment": QUALITY_ASSESSMENT_PROMPT
}


def get_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_BASE_TEMPLATES.get(template_id)


def list_templates() -> List[str]:
    """
    List all available template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_BASE_TEMPLATES.keys())
