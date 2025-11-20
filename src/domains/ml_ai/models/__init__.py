"""ML model specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class LayerType(Enum):
    CONV = "conv"
    POOL = "pool"
    DENSE = "dense"
    ATTENTION = "attention"

@dataclass
class ModelSpec:
    name: str
    layers: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = []

class ModelKnowledge:
    def __init__(self):
        self.patterns = {
            "cnn": "Convolutional neural networks",
            "rnn": "Recurrent neural networks",
            "transformer": "Attention-based models",
            "resnet": "Residual connections"
        }
    
    def validate_model_spec(self, spec: ModelSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Model name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_model_code(self, spec: ModelSpec) -> Tuple[bool, str, str]:
        code = f"class {spec.name}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: Define layers\n\n    def forward(self, x):\n        # TODO: Implement\n        return x\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "layer_types": len(LayerType)}

__all__ = ["LayerType", "ModelSpec", "ModelKnowledge"]
