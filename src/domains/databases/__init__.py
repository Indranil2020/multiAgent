"""Database systems domain knowledge and specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class DatabaseParadigm(Enum):
    RELATIONAL = "relational"
    DOCUMENT = "document"
    GRAPH = "graph"
    KEY_VALUE = "key_value"
    TIME_SERIES = "time_series"

class ConsistencyModel(Enum):
    STRONG = "strong"
    EVENTUAL = "eventual"
    CAUSAL = "causal"

@dataclass
class DatabaseSpec:
    name: str
    paradigm: DatabaseParadigm
    consistency: ConsistencyModel
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

class DatabaseKnowledge:
    def __init__(self):
        self.patterns = {
            "acid_transactions": "Atomicity, Consistency, Isolation, Durability",
            "indexing": "B-tree and hash indexes",
            "query_optimization": "Cost-based query planning",
            "replication": "Master-slave and multi-master",
            "sharding": "Horizontal partitioning",
            "caching": "Query result caching"
        }
    
    def validate_database_spec(self, spec: DatabaseSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Database name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def estimate_complexity(self, spec: DatabaseSpec) -> Tuple[bool, int, str]:
        complexity = 5
        if spec.paradigm == DatabaseParadigm.GRAPH:
            complexity += 3
        if spec.consistency == ConsistencyModel.STRONG:
            complexity += 2
        complexity += len(spec.features)
        return (True, complexity, f"Complexity: {complexity}")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "paradigms": len(DatabaseParadigm)}

__all__ = ["DatabaseParadigm", "ConsistencyModel", "DatabaseSpec", "DatabaseKnowledge"]
