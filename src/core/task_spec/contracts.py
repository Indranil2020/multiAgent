"""
Contract definitions for task specifications.

This module provides tools for defining and working with contracts (preconditions,
postconditions, and invariants) in the task specification system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from .types import Predicate


@dataclass
class ContractBuilder:
    """
    Builder for creating contract predicates.
    
    Provides a fluent interface for constructing preconditions, postconditions,
    and invariants with proper validation.
    """
    predicates: List[Predicate] = field(default_factory=list)
    
    def add_predicate(
        self,
        name: str,
        expression: str,
        description: str,
        severity: str = "error"
    ) -> 'ContractBuilder':
        """
        Add a predicate to the contract.
        
        Args:
            name: Unique name for the predicate
            expression: Boolean expression to evaluate
            description: Human-readable description
            severity: Severity level (error, warning, info)
        
        Returns:
            Self for method chaining
        """
        predicate = Predicate(
            name=name,
            expression=expression,
            description=description,
            severity=severity
        )
        
        if not predicate.is_valid():
            return self  # Skip invalid predicates
        
        self.predicates.append(predicate)
        return self
    
    def add_not_null(self, param_name: str, description: str = "") -> 'ContractBuilder':
        """Add a not-null constraint."""
        desc = description or f"Parameter '{param_name}' must not be None"
        return self.add_predicate(
            name=f"{param_name}_not_null",
            expression=f"{param_name} is not None",
            description=desc
        )
    
    def add_positive(self, param_name: str, description: str = "") -> 'ContractBuilder':
        """Add a positive number constraint."""
        desc = description or f"Parameter '{param_name}' must be positive"
        return self.add_predicate(
            name=f"{param_name}_positive",
            expression=f"{param_name} > 0",
            description=desc
        )
    
    def add_non_negative(self, param_name: str, description: str = "") -> 'ContractBuilder':
        """Add a non-negative constraint."""
        desc = description or f"Parameter '{param_name}' must be non-negative"
        return self.add_predicate(
            name=f"{param_name}_non_negative",
            expression=f"{param_name} >= 0",
            description=desc
        )
    
    def add_range(
        self,
        param_name: str,
        min_val: Any,
        max_val: Any,
        description: str = ""
    ) -> 'ContractBuilder':
        """Add a range constraint."""
        desc = description or f"Parameter '{param_name}' must be between {min_val} and {max_val}"
        return self.add_predicate(
            name=f"{param_name}_in_range",
            expression=f"{min_val} <= {param_name} <= {max_val}",
            description=desc
        )
    
    def add_length_constraint(
        self,
        param_name: str,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        description: str = ""
    ) -> 'ContractBuilder':
        """Add a length constraint for strings or collections."""
        if min_len is not None and max_len is not None:
            expr = f"{min_len} <= len({param_name}) <= {max_len}"
            desc = description or f"Length of '{param_name}' must be between {min_len} and {max_len}"
        elif min_len is not None:
            expr = f"len({param_name}) >= {min_len}"
            desc = description or f"Length of '{param_name}' must be at least {min_len}"
        elif max_len is not None:
            expr = f"len({param_name}) <= {max_len}"
            desc = description or f"Length of '{param_name}' must be at most {max_len}"
        else:
            return self  # No constraint specified
        
        return self.add_predicate(
            name=f"{param_name}_length",
            expression=expr,
            description=desc
        )
    
    def add_type_constraint(
        self,
        param_name: str,
        expected_type: str,
        description: str = ""
    ) -> 'ContractBuilder':
        """Add a type constraint."""
        desc = description or f"Parameter '{param_name}' must be of type {expected_type}"
        return self.add_predicate(
            name=f"{param_name}_type",
            expression=f"isinstance({param_name}, {expected_type})",
            description=desc
        )
    
    def add_custom(
        self,
        name: str,
        expression: str,
        description: str,
        severity: str = "error"
    ) -> 'ContractBuilder':
        """Add a custom predicate."""
        return self.add_predicate(name, expression, description, severity)
    
    def build(self) -> List[Predicate]:
        """Build and return the list of predicates."""
        return self.predicates.copy()
    
    def clear(self) -> 'ContractBuilder':
        """Clear all predicates."""
        self.predicates.clear()
        return self


@dataclass
class PreconditionBuilder(ContractBuilder):
    """Builder specifically for preconditions."""
    
    def add_input_validation(
        self,
        param_name: str,
        param_type: str,
        not_null: bool = True,
        min_val: Optional[Any] = None,
        max_val: Optional[Any] = None
    ) -> 'PreconditionBuilder':
        """
        Add standard input validation for a parameter.
        
        Args:
            param_name: Name of the parameter
            param_type: Expected type
            not_null: Whether to enforce not-null
            min_val: Minimum value (for numeric types)
            max_val: Maximum value (for numeric types)
        
        Returns:
            Self for method chaining
        """
        if not_null:
            self.add_not_null(param_name)
        
        self.add_type_constraint(param_name, param_type)
        
        if min_val is not None or max_val is not None:
            if min_val is not None and max_val is not None:
                self.add_range(param_name, min_val, max_val)
            elif min_val is not None:
                self.add_predicate(
                    name=f"{param_name}_min",
                    expression=f"{param_name} >= {min_val}",
                    description=f"Parameter '{param_name}' must be at least {min_val}"
                )
            else:
                self.add_predicate(
                    name=f"{param_name}_max",
                    expression=f"{param_name} <= {max_val}",
                    description=f"Parameter '{param_name}' must be at most {max_val}"
                )
        
        return self


@dataclass
class PostconditionBuilder(ContractBuilder):
    """Builder specifically for postconditions."""
    
    def add_return_not_null(self, description: str = "") -> 'PostconditionBuilder':
        """Add constraint that return value is not null."""
        desc = description or "Return value must not be None"
        return self.add_predicate(
            name="return_not_null",
            expression="__return__ is not None",
            description=desc
        )
    
    def add_return_type(self, expected_type: str, description: str = "") -> 'PostconditionBuilder':
        """Add constraint on return value type."""
        desc = description or f"Return value must be of type {expected_type}"
        return self.add_predicate(
            name="return_type",
            expression=f"isinstance(__return__, {expected_type})",
            description=desc
        )
    
    def add_return_range(
        self,
        min_val: Any,
        max_val: Any,
        description: str = ""
    ) -> 'PostconditionBuilder':
        """Add range constraint on return value."""
        desc = description or f"Return value must be between {min_val} and {max_val}"
        return self.add_predicate(
            name="return_range",
            expression=f"{min_val} <= __return__ <= {max_val}",
            description=desc
        )
    
    def add_state_change(
        self,
        state_var: str,
        expected_change: str,
        description: str
    ) -> 'PostconditionBuilder':
        """Add constraint on state changes."""
        return self.add_predicate(
            name=f"{state_var}_changed",
            expression=expected_change,
            description=description
        )


@dataclass
class InvariantBuilder(ContractBuilder):
    """Builder specifically for invariants."""
    
    def add_consistency_check(
        self,
        expression: str,
        description: str
    ) -> 'InvariantBuilder':
        """Add a consistency invariant."""
        return self.add_predicate(
            name="consistency",
            expression=expression,
            description=description
        )
    
    def add_resource_limit(
        self,
        resource_name: str,
        max_value: Any,
        description: str = ""
    ) -> 'InvariantBuilder':
        """Add a resource limit invariant."""
        desc = description or f"Resource '{resource_name}' must not exceed {max_value}"
        return self.add_predicate(
            name=f"{resource_name}_limit",
            expression=f"{resource_name} <= {max_value}",
            description=desc
        )


class ContractValidator:
    """
    Validator for contract predicates.
    
    Provides validation logic for checking if predicates are well-formed
    and can be evaluated.
    """
    
    def __init__(self):
        """Initialize the contract validator."""
        self.forbidden_keywords = {
            'exec', 'eval', '__import__', 'compile', 'open',
            'file', 'input', 'raw_input', 'execfile'
        }
        self.allowed_operators = {
            '==', '!=', '<', '>', '<=', '>=',
            'and', 'or', 'not', 'in', 'is',
            '+', '-', '*', '/', '//', '%', '**'
        }
    
    def validate_predicate(self, predicate: Predicate) -> bool:
        """
        Validate a single predicate.
        
        Args:
            predicate: The predicate to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not predicate.is_valid():
            return False
        
        if not self.validate_expression_safety(predicate.expression):
            return False
        
        return True
    
    def validate_expression_safety(self, expression: str) -> bool:
        """
        Validate that an expression is safe to evaluate.
        
        Args:
            expression: The expression to validate
        
        Returns:
            True if safe, False otherwise
        """
        if not expression:
            return False
        
        # Check for forbidden keywords
        expr_lower = expression.lower()
        for keyword in self.forbidden_keywords:
            if keyword in expr_lower:
                return False
        
        # Check for suspicious patterns
        if '__' in expression and '__return__' not in expression:
            # Allow __return__ but be suspicious of other dunders
            if any(dunder in expression for dunder in ['__dict__', '__class__', '__bases__']):
                return False
        
        return True
    
    def validate_contract_set(self, predicates: List[Predicate]) -> bool:
        """
        Validate a set of predicates for consistency.
        
        Args:
            predicates: List of predicates to validate
        
        Returns:
            True if all valid and consistent, False otherwise
        """
        if not predicates:
            return True  # Empty set is valid
        
        # Validate each predicate
        for predicate in predicates:
            if not self.validate_predicate(predicate):
                return False
        
        # Check for duplicate names
        names = [p.name for p in predicates]
        if len(names) != len(set(names)):
            return False  # Duplicate names found
        
        return True


def create_standard_preconditions(
    inputs: List[Dict[str, Any]]
) -> List[Predicate]:
    """
    Create standard preconditions for a set of inputs.
    
    Args:
        inputs: List of input parameter specifications
    
    Returns:
        List of standard precondition predicates
    """
    builder = PreconditionBuilder()
    
    for inp in inputs:
        param_name = inp.get('name', '')
        param_type = inp.get('type', 'Any')
        not_null = inp.get('not_null', True)
        
        if param_name and param_type:
            builder.add_input_validation(
                param_name=param_name,
                param_type=param_type,
                not_null=not_null
            )
    
    return builder.build()


def create_standard_postconditions(
    output_type: str,
    not_null: bool = True
) -> List[Predicate]:
    """
    Create standard postconditions for a return value.
    
    Args:
        output_type: Expected type of return value
        not_null: Whether return value must not be null
    
    Returns:
        List of standard postcondition predicates
    """
    builder = PostconditionBuilder()
    
    if not_null:
        builder.add_return_not_null()
    
    if output_type and output_type != 'None':
        builder.add_return_type(output_type)
    
    return builder.build()
