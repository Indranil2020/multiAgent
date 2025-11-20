"""Game AI specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class PathfindingAlgorithm(Enum):
    ASTAR = "astar"
    DIJKSTRA = "dijkstra"
    NAVMESH = "navmesh"

@dataclass
class AISpec:
    name: str
    pathfinding: PathfindingAlgorithm
    max_agents: int = 100

class AIKnowledge:
    def __init__(self):
        self.patterns = {
            "behavior_trees": "Hierarchical AI logic",
            "state_machines": "Finite state machines",
            "pathfinding": "A* and navigation meshes",
            "steering_behaviors": "Flocking and avoidance"
        }
    
    def validate_ai_spec(self, spec: AISpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.max_agents < 1:
            errors.append("max_agents must be positive")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_behavior_tree(self) -> Tuple[bool, str, str]:
        code = "class BehaviorNode {\n    virtual Status execute() = 0;\n};\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "algorithms": len(PathfindingAlgorithm)}

__all__ = ["PathfindingAlgorithm", "AISpec", "AIKnowledge"]
