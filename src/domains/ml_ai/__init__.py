"""ML/AI systems domain knowledge and specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class MLFramework(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    JAX = "jax"

class ModelArchitecture(Enum):
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"

@dataclass
class MLSpec:
    name: str
    framework: MLFramework
    architecture: ModelArchitecture
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

class MLKnowledge:
    def __init__(self):
        self.patterns = {
            "data_preprocessing": "Normalization and augmentation",
            "model_training": "Gradient descent optimization",
            "hyperparameter_tuning": "Grid or random search",
            "model_serving": "Inference optimization",
            "distributed_training": "Data and model parallelism"
        }
    
    def validate_ml_spec(self, spec: MLSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Model name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "frameworks": len(MLFramework)}

__all__ = ["MLFramework", "ModelArchitecture", "MLSpec", "MLKnowledge"]
