"""Time-series database specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class CompressionAlgorithm(Enum):
    DELTA = "delta"
    GORILLA = "gorilla"
    SNAPPY = "snappy"

@dataclass
class TimeSeriesSpec:
    name: str
    compression: CompressionAlgorithm
    retention_days: int = 30

class TimeSeriesKnowledge:
    def __init__(self):
        self.patterns = {
            "downsampling": "Reduce data resolution over time",
            "continuous_aggregation": "Pre-compute aggregates",
            "time_partitioning": "Partition by time ranges",
            "compression": "Efficient storage"
        }
    
    def validate_timeseries_spec(self, spec: TimeSeriesSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if spec.retention_days < 1:
            errors.append("Retention must be at least 1 day")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_retention_policy(self, spec: TimeSeriesSpec) -> Tuple[bool, str, str]:
        code = f"CREATE RETENTION POLICY rp_{spec.retention_days}d ON {spec.name} DURATION {spec.retention_days}d;\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "algorithms": len(CompressionAlgorithm)}

__all__ = ["CompressionAlgorithm", "TimeSeriesSpec", "TimeSeriesKnowledge"]
