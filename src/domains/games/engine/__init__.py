"""Game engine specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class ECSPattern(Enum):
    ARCHETYPE = "archetype"
    SPARSE_SET = "sparse_set"

@dataclass
class EngineSpec:
    name: str
    ecs_pattern: ECSPattern
    max_entities: int = 10000

class EngineKnowledge:
    def __init__(self):
        self.patterns = {
            "ecs": "Entity-component-system architecture",
            "update_loop": "Fixed timestep updates",
            "event_system": "Decoupled event handling",
            "resource_management": "Asset loading and caching"
        }
    
    def validate_engine_spec(self, spec: EngineSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.max_entities < 1:
            errors.append("max_entities must be positive")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_entity_code(self) -> Tuple[bool, str, str]:
        code = "struct Entity {\n    uint32_t id;\n    ComponentMask components;\n};\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "ecs_patterns": len(ECSPattern)}

__all__ = ["ECSPattern", "EngineSpec", "EngineKnowledge"]
