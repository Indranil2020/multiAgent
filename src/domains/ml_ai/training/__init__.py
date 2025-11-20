"""ML training system specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class Optimizer(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"

@dataclass
class TrainingSpec:
    name: str
    optimizer: Optimizer
    learning_rate: float = 0.001
    epochs: int = 10

class TrainingKnowledge:
    def __init__(self):
        self.patterns = {
            "gradient_descent": "Iterative optimization",
            "learning_rate_scheduling": "Adaptive learning rates",
            "early_stopping": "Prevent overfitting",
            "checkpointing": "Save model states",
            "distributed_training": "Multi-GPU training"
        }
    
    def validate_training_spec(self, spec: TrainingSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        if spec.epochs < 1:
            errors.append("Epochs must be at least 1")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_training_loop(self, spec: TrainingSpec) -> Tuple[bool, str, str]:
        code = f"for epoch in range({spec.epochs}):\n    for batch in dataloader:\n        optimizer.zero_grad()\n        loss = model(batch)\n        loss.backward()\n        optimizer.step()\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "optimizers": len(Optimizer)}

__all__ = ["Optimizer", "TrainingSpec", "TrainingKnowledge"]
