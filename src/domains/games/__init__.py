"""Game development domain knowledge and specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class GameArchitecture(Enum):
    ENTITY_COMPONENT = "entity_component"
    OBJECT_ORIENTED = "object_oriented"
    DATA_ORIENTED = "data_oriented"

class RenderingAPI(Enum):
    OPENGL = "opengl"
    VULKAN = "vulkan"
    DIRECTX = "directx"
    METAL = "metal"

@dataclass
class GameSpec:
    name: str
    architecture: GameArchitecture
    rendering_api: RenderingAPI
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

class GameKnowledge:
    def __init__(self):
        self.patterns = {
            "game_loop": "Fixed timestep update loop",
            "entity_component_system": "Data-driven architecture",
            "scene_graph": "Hierarchical scene organization",
            "asset_management": "Loading and caching assets",
            "state_machine": "Game state management"
        }
    
    def validate_game_spec(self, spec: GameSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Game name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_game_loop(self) -> Tuple[bool, str, str]:
        code = "void game_loop() {\n    while (running) {\n        process_input();\n        update(delta_time);\n        render();\n    }\n}\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "architectures": len(GameArchitecture)}

__all__ = ["GameArchitecture", "RenderingAPI", "GameSpec", "GameKnowledge"]
