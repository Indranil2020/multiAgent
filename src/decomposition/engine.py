"""
Main decomposition engine.

This module orchestrates the entire decomposition process, using analyzers
and strategies to break down complex tasks into atomic units.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import sys
sys.path.append('/home/niel/git/multiAgent')

from src.core.task_spec.types import (
    TaskType,
    TypedParameter,
    Predicate,
    TestCase,
    Property
)

from .analyzers import (
    ComplexityAnalyzer,
    ComplexityMetrics,
    DependencyAnalyzer,
    TaskDependency,
    DependencyType,
    RiskAnalyzer,
    RiskMetrics
)

from .strategies import (
    HierarchicalStrategy,
    DecompositionNode,
    AtomicTaskCreator,
    AtomicTaskSpec
)

from .dag_builder import DAGBuilder, TaskDAG


class DecompositionStrategy(Enum):
    """Available decomposition strategies."""
    HIERARCHICAL = "hierarchical"
    FUNCTIONAL = "functional"
    DOMAIN_DRIVEN = "domain_driven"
    AUTO = "auto"  # Automatically select best strategy


@dataclass
class DecompositionTree:
    """
    Tree structure for hierarchical decomposition.
    
    Attributes:
        root: Root decomposition node
        atomic_tasks: List of atomic task specifications
        total_nodes: Total number of nodes in tree
        max_depth: Maximum depth of tree
        leaf_count: Number of leaf nodes
    """
    root: DecompositionNode
    atomic_tasks: List[AtomicTaskSpec] = field(default_factory=list)
    total_nodes: int = 0
    max_depth: int = 0
    leaf_count: int = 0
    
    def is_valid(self) -> bool:
        """Check if decomposition tree is valid."""
        return self.root.is_valid() and self.total_nodes > 0


@dataclass
class DecompositionResult:
    """
    Result of decomposition process.
    
    Attributes:
        success: Whether decomposition succeeded
        tree: Decomposition tree
        dag: Execution DAG
        complexity_metrics: Complexity analysis results
        risk_metrics: Risk analysis results
        message: Result message
    """
    success: bool
    tree: Optional[DecompositionTree]
    dag: Optional[TaskDAG]
    complexity_metrics: Optional[ComplexityMetrics]
    risk_metrics: Optional[RiskMetrics]
    message: str


class DecompositionEngine:
    """
    Main orchestrator for task decomposition.
    
    This engine coordinates analyzers and strategies to decompose complex
    tasks into atomic units suitable for agent execution.
    """
    
    MAX_ATOMIC_LINES = 20
    MAX_ATOMIC_COMPLEXITY = 5
    
    def __init__(
        self,
        strategy: DecompositionStrategy = DecompositionStrategy.HIERARCHICAL
    ):
        """
        Initialize decomposition engine.
        
        Args:
            strategy: Decomposition strategy to use
        """
        self.strategy = strategy
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.hierarchical_strategy = HierarchicalStrategy()
        self.atomic_creator = AtomicTaskCreator()
        self.dag_builder = DAGBuilder()
    
    def decompose(
        self,
        task_id: str,
        name: str,
        description: str,
        task_type: TaskType,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        test_cases: List[TestCase],
        properties: List[Property]
    ) -> DecompositionResult:
        """
        Decompose a task into atomic units.
        
        Args:
            task_id: Task identifier
            name: Task name
            description: Task description
            task_type: Type of task
            inputs: Input parameters
            outputs: Output parameters
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
            test_cases: Test cases
            properties: Properties for property-based testing
        
        Returns:
            DecompositionResult with tree and DAG
        """
        # Validate inputs
        if not task_id:
            return DecompositionResult(
                success=False,
                tree=None,
                dag=None,
                complexity_metrics=None,
                risk_metrics=None,
                message="task_id cannot be empty"
            )
        
        if not name:
            return DecompositionResult(
                success=False,
                tree=None,
                dag=None,
                complexity_metrics=None,
                risk_metrics=None,
                message="name cannot be empty"
            )
        
        if not description:
            return DecompositionResult(
                success=False,
                tree=None,
                dag=None,
                complexity_metrics=None,
                risk_metrics=None,
                message="description cannot be empty"
            )
        
        # Step 1: Analyze complexity
        success, complexity_metrics, message = self.complexity_analyzer.analyze(
            description=description,
            inputs=inputs,
            outputs=outputs,
            preconditions=preconditions,
            postconditions=postconditions,
            test_cases=test_cases,
            properties=properties
        )
        
        if not success:
            return DecompositionResult(
                success=False,
                tree=None,
                dag=None,
                complexity_metrics=None,
                risk_metrics=None,
                message=f"Complexity analysis failed: {message}"
            )
        
        # Step 2: Analyze risk
        success, risk_metrics, message = self.risk_analyzer.analyze(
            description=description,
            estimated_complexity=complexity_metrics.estimated_complexity,
            dependency_count=0,  # Will be updated after decomposition
            has_security_requirements=any('security' in p.description.lower() 
                                         for p in preconditions + postconditions),
            has_performance_requirements=any('performance' in p.description.lower() 
                                            for p in preconditions + postconditions),
            is_external_integration='external' in description.lower() or 'api' in description.lower(),
            handles_sensitive_data='password' in description.lower() or 'credential' in description.lower()
        )
        
        if not success:
            return DecompositionResult(
                success=False,
                tree=None,
                dag=None,
                complexity_metrics=complexity_metrics,
                risk_metrics=None,
                message=f"Risk analysis failed: {message}"
            )
        
        # Step 3: Check if task is already atomic
        if complexity_metrics.is_atomic:
            # Create single atomic task
            success, atomic_task, message = self.atomic_creator.create_atomic_task(
                parent_task_id=task_id,
                subtask_index=0,
                name=name,
                description=description,
                task_type=task_type,
                inputs=inputs,
                outputs=outputs,
                preconditions=preconditions,
                postconditions=postconditions
            )
            
            if not success:
                return DecompositionResult(
                    success=False,
                    tree=None,
                    dag=None,
                    complexity_metrics=complexity_metrics,
                    risk_metrics=risk_metrics,
                    message=f"Failed to create atomic task: {message}"
                )
            
            # Create simple tree with single node
            root = DecompositionNode(
                task_id=task_id,
                name=name,
                description=description,
                level=0,
                is_atomic=True
            )
            
            tree = DecompositionTree(
                root=root,
                atomic_tasks=[atomic_task],
                total_nodes=1,
                max_depth=0,
                leaf_count=1
            )
            
            # Create simple DAG with single node
            self.dag_builder.build_from_tasks(
                task_ids=[atomic_task.task_id],
                task_names={atomic_task.task_id: atomic_task.name},
                task_descriptions={atomic_task.task_id: atomic_task.description},
                dependencies={atomic_task.task_id: []}
            )
            
            return DecompositionResult(
                success=True,
                tree=tree,
                dag=self.dag_builder.get_dag(),
                complexity_metrics=complexity_metrics,
                risk_metrics=risk_metrics,
                message="Task is atomic - no decomposition needed"
            )
        
        # Step 4: Decompose using selected strategy
        if self.strategy == DecompositionStrategy.HIERARCHICAL:
            success, root_node, message = self.hierarchical_strategy.decompose(
                task_id=task_id,
                name=name,
                description=description,
                estimated_complexity=complexity_metrics.estimated_complexity
            )
            
            if not success:
                return DecompositionResult(
                    success=False,
                    tree=None,
                    dag=None,
                    complexity_metrics=complexity_metrics,
                    risk_metrics=risk_metrics,
                    message=f"Hierarchical decomposition failed: {message}"
                )
            
            # Get tree statistics
            total_nodes, max_depth, leaf_count = self.hierarchical_strategy.get_tree_statistics(root_node)
            
            # Collect leaf nodes (atomic tasks)
            leaf_nodes = self.hierarchical_strategy.collect_leaf_nodes(root_node)
            
            # Create atomic task specs for each leaf
            atomic_tasks = []
            for i, leaf in enumerate(leaf_nodes):
                success, atomic_task, message = self.atomic_creator.create_atomic_task(
                    parent_task_id=leaf.task_id,
                    subtask_index=0,
                    name=leaf.name,
                    description=leaf.description,
                    task_type=task_type,
                    inputs=inputs if i == 0 else [],  # First task gets inputs
                    outputs=outputs if i == len(leaf_nodes) - 1 else [],  # Last task produces outputs
                    preconditions=preconditions if i == 0 else [],
                    postconditions=postconditions if i == len(leaf_nodes) - 1 else []
                )
                
                if success and atomic_task:
                    atomic_tasks.append(atomic_task)
            
            # Create decomposition tree
            tree = DecompositionTree(
                root=root_node,
                atomic_tasks=atomic_tasks,
                total_nodes=total_nodes,
                max_depth=max_depth,
                leaf_count=leaf_count
            )
            
            # Step 5: Build execution DAG
            task_ids = [task.task_id for task in atomic_tasks]
            task_names = {task.task_id: task.name for task in atomic_tasks}
            task_descriptions = {task.task_id: task.description for task in atomic_tasks}
            
            # Create sequential dependencies (simple approach)
            dependencies = {}
            for i, task in enumerate(atomic_tasks):
                if i == 0:
                    dependencies[task.task_id] = []
                else:
                    dependencies[task.task_id] = [atomic_tasks[i-1].task_id]
            
            success, message = self.dag_builder.build_from_tasks(
                task_ids=task_ids,
                task_names=task_names,
                task_descriptions=task_descriptions,
                dependencies=dependencies
            )
            
            if not success:
                return DecompositionResult(
                    success=False,
                    tree=tree,
                    dag=None,
                    complexity_metrics=complexity_metrics,
                    risk_metrics=risk_metrics,
                    message=f"DAG building failed: {message}"
                )
            
            return DecompositionResult(
                success=True,
                tree=tree,
                dag=self.dag_builder.get_dag(),
                complexity_metrics=complexity_metrics,
                risk_metrics=risk_metrics,
                message=f"Decomposed into {len(atomic_tasks)} atomic tasks"
            )
        
        # Other strategies not yet implemented
        return DecompositionResult(
            success=False,
            tree=None,
            dag=None,
            complexity_metrics=complexity_metrics,
            risk_metrics=risk_metrics,
            message=f"Strategy {self.strategy} not yet implemented"
        )
    
    def is_atomic(
        self,
        description: str,
        inputs: List[TypedParameter],
        outputs: List[TypedParameter],
        preconditions: List[Predicate],
        postconditions: List[Predicate],
        test_cases: List[TestCase],
        properties: List[Property]
    ) -> Tuple[bool, Optional[ComplexityMetrics], str]:
        """
        Check if a task is atomic.
        
        Args:
            description: Task description
            inputs: Input parameters
            outputs: Output parameters
            preconditions: Precondition predicates
            postconditions: Postcondition predicates
            test_cases: Test cases
            properties: Properties
        
        Returns:
            Tuple of (is_atomic, metrics or None, message)
        """
        success, metrics, message = self.complexity_analyzer.analyze(
            description=description,
            inputs=inputs,
            outputs=outputs,
            preconditions=preconditions,
            postconditions=postconditions,
            test_cases=test_cases,
            properties=properties
        )
        
        if not success:
            return (False, None, message)
        
        return (metrics.is_atomic, metrics, "Analysis complete")
    
    def get_decomposition_recommendation(
        self,
        complexity_metrics: ComplexityMetrics
    ) -> Tuple[bool, str]:
        """
        Get recommendation on whether to decompose.
        
        Args:
            complexity_metrics: Complexity metrics
        
        Returns:
            Tuple of (should_decompose, reason)
        """
        return self.complexity_analyzer.get_decomposition_recommendation(complexity_metrics)
    
    def reset(self) -> None:
        """Reset the engine state."""
        self.dependency_analyzer.clear()
        self.hierarchical_strategy.reset_counter()
        self.atomic_creator.reset_counter()
        self.dag_builder.clear()
