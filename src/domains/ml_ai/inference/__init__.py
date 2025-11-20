"""ML inference system specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class InferenceOptimization(Enum):
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"

@dataclass
class InferenceSpec:
    name: str
    optimizations: List[InferenceOptimization] = None
    batch_size: int = 1
    
    def __post_init__(self):
        if self.optimizations is None:
            self.optimizations = []

class InferenceKnowledge:
    def __init__(self):
        self.patterns = {
            "quantization": "Reduce precision for speed",
            "pruning": "Remove unnecessary weights",
            "distillation": "Compress to smaller model",
            "batching": "Process multiple inputs together",
            "caching": "Cache frequent predictions"
        }
    
    def validate_inference_spec(self, spec: InferenceSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.batch_size < 1:
            errors.append("Batch size must be at least 1")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_inference_code(self, spec: InferenceSpec) -> Tuple[bool, str, str]:
        code = f"def predict(model, input):\n    with torch.no_grad():\n        output = model(input)\n    return output\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "optimizations": len(InferenceOptimization)}

__all__ = ["InferenceOptimization", "InferenceSpec", "InferenceKnowledge"]
