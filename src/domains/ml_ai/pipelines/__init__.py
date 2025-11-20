"""ML pipeline specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class PipelineStage(Enum):
    INGEST = "ingest"
    TRANSFORM = "transform"
    TRAIN = "train"
    EVALUATE = "evaluate"

@dataclass
class PipelineSpec:
    name: str
    stages: List[PipelineStage] = None
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = []

class PipelineKnowledge:
    def __init__(self):
        self.patterns = {
            "etl": "Extract, transform, load",
            "feature_engineering": "Feature extraction and selection",
            "data_validation": "Schema and drift detection",
            "orchestration": "Workflow management"
        }
    
    def validate_pipeline_spec(self, spec: PipelineSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Pipeline name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "stages": len(PipelineStage)}

__all__ = ["PipelineStage", "PipelineSpec", "PipelineKnowledge"]
