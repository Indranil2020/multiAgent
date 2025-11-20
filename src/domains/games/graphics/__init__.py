"""Graphics system specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class RenderingPipeline(Enum):
    FORWARD = "forward"
    DEFERRED = "deferred"
    TILED = "tiled"

@dataclass
class GraphicsSpec:
    name: str
    pipeline: RenderingPipeline
    max_lights: int = 8

class GraphicsKnowledge:
    def __init__(self):
        self.patterns = {
            "pbr": "Physically-based rendering",
            "culling": "Frustum and occlusion culling",
            "lod": "Level of detail management",
            "batching": "Draw call reduction"
        }
    
    def validate_graphics_spec(self, spec: GraphicsSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.max_lights < 1:
            errors.append("max_lights must be positive")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_shader_code(self, shader_type: str) -> Tuple[bool, str, str]:
        code = f"// {shader_type} shader\nvoid main() {{\n    // TODO\n}}\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "pipelines": len(RenderingPipeline)}

__all__ = ["RenderingPipeline", "GraphicsSpec", "GraphicsKnowledge"]
