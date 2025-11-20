"""
Functional decomposition strategy.

This module implements functional decomposition, breaking tasks by
functional capabilities and responsibilities.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys
sys.path.append('/home/niel/git/multiAgent')
from src.core.task_spec.types import TaskType


@dataclass
class FunctionalComponent:
    """
    A functional component in decomposition.
    
    Attributes:
        component_id: Unique identifier
        name: Component name
        description: Component description
        responsibility: Primary responsibility
        inputs: Required inputs
        outputs: Produced outputs
    """
    component_id: str
    name: str
    description: str
    responsibility: str
    inputs: List[str]
    outputs: List[str]
    
    def is_valid(self) -> bool:
        """Check if component is valid."""
        return bool(
            self.component_id and
            self.name and
            self.description and
            self.responsibility
        )


class FunctionalStrategy:
    """
    Functional decomposition strategy.
    
    This strategy decomposes tasks based on functional responsibilities,
    identifying distinct capabilities and grouping related functionality.
    """
    
    def __init__(self):
        """Initialize functional strategy."""
        self.component_counter = 0
    
    def decompose_by_function(
        self,
        task_id: str,
        description: str,
        task_type: TaskType
    ) -> Tuple[bool, List[FunctionalComponent], str]:
        """
        Decompose task by functional responsibilities.
        
        Args:
            task_id: Task ID
            description: Task description
            task_type: Type of task
        
        Returns:
            Tuple of (success, list of components, message)
        """
        if not task_id:
            return (False, [], "task_id cannot be empty")
        
        if not description:
            return (False, [], "description cannot be empty")
        
        # Identify functional components based on task type
        components = self._identify_components(task_id, description, task_type)
        
        if not components:
            return (False, [], "No functional components identified")
        
        return (True, components, f"Identified {len(components)} functional components")
    
    def _identify_components(
        self,
        task_id: str,
        description: str,
        task_type: TaskType
    ) -> List[FunctionalComponent]:
        """
        Identify functional components from description.
        
        Args:
            task_id: Task ID
            description: Task description
            task_type: Task type
        
        Returns:
            List of functional components
        """
        components = []
        desc_lower = description.lower()
        
        # Common functional patterns
        if task_type == TaskType.IMPLEMENTATION:
            # Input validation component
            if any(kw in desc_lower for kw in ['validate', 'check', 'verify']):
                components.append(FunctionalComponent(
                    component_id=f"{task_id}_validation",
                    name="Input Validation",
                    description="Validate input parameters",
                    responsibility="Ensure inputs meet requirements",
                    inputs=["raw_inputs"],
                    outputs=["validated_inputs"]
                ))
            
            # Core logic component
            components.append(FunctionalComponent(
                component_id=f"{task_id}_core",
                name="Core Logic",
                description="Main processing logic",
                responsibility="Implement core functionality",
                inputs=["validated_inputs"],
                outputs=["processed_data"]
            ))
            
            # Output formatting component
            if any(kw in desc_lower for kw in ['format', 'transform', 'convert']):
                components.append(FunctionalComponent(
                    component_id=f"{task_id}_formatting",
                    name="Output Formatting",
                    description="Format output data",
                    responsibility="Transform data to required format",
                    inputs=["processed_data"],
                    outputs=["formatted_output"]
                ))
        
        elif task_type == TaskType.TESTING:
            # Test setup component
            components.append(FunctionalComponent(
                component_id=f"{task_id}_setup",
                name="Test Setup",
                description="Setup test environment",
                responsibility="Prepare test fixtures and data",
                inputs=[],
                outputs=["test_context"]
            ))
            
            # Test execution component
            components.append(FunctionalComponent(
                component_id=f"{task_id}_execution",
                name="Test Execution",
                description="Execute test cases",
                responsibility="Run tests and collect results",
                inputs=["test_context"],
                outputs=["test_results"]
            ))
            
            # Test cleanup component
            components.append(FunctionalComponent(
                component_id=f"{task_id}_cleanup",
                name="Test Cleanup",
                description="Cleanup test environment",
                responsibility="Release resources and cleanup",
                inputs=["test_context"],
                outputs=[]
            ))
        
        # If no specific components identified, create generic ones
        if not components:
            components.append(FunctionalComponent(
                component_id=f"{task_id}_main",
                name="Main Function",
                description=description,
                responsibility="Primary task responsibility",
                inputs=["inputs"],
                outputs=["outputs"]
            ))
        
        return components
    
    def group_related_components(
        self,
        components: List[FunctionalComponent]
    ) -> List[List[FunctionalComponent]]:
        """
        Group related functional components.
        
        Args:
            components: List of components to group
        
        Returns:
            List of component groups
        """
        if not components:
            return []
        
        # Simple grouping by data flow
        groups = []
        current_group = [components[0]]
        
        for i in range(1, len(components)):
            prev_component = components[i-1]
            curr_component = components[i]
            
            # Check if current component uses output from previous
            if any(output in curr_component.inputs for output in prev_component.outputs):
                current_group.append(curr_component)
            else:
                groups.append(current_group)
                current_group = [curr_component]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def validate_components(
        self,
        components: List[FunctionalComponent]
    ) -> Tuple[bool, str]:
        """
        Validate functional components.
        
        Args:
            components: Components to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not components:
            return (False, "No components to validate")
        
        # Check each component is valid
        for component in components:
            if not component.is_valid():
                return (False, f"Invalid component: {component.component_id}")
        
        # Check for duplicate IDs
        component_ids = [c.component_id for c in components]
        if len(component_ids) != len(set(component_ids)):
            return (False, "Duplicate component IDs found")
        
        return (True, "Components are valid")
    
    def reset_counter(self) -> None:
        """Reset the component counter."""
        self.component_counter = 0
