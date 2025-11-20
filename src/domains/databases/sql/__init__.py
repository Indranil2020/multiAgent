"""SQL database specifications."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class SQLDialect(Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"

class IndexType(Enum):
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"

@dataclass
class SQLSpec:
    name: str
    dialect: SQLDialect
    tables: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.tables is None:
            self.tables = []

class SQLKnowledge:
    def __init__(self):
        self.patterns = {
            "normalization": "Reduce data redundancy",
            "indexing": "Speed up queries",
            "transactions": "ACID guarantees",
            "joins": "Combine data from multiple tables",
            "constraints": "Enforce data integrity"
        }
    
    def validate_sql_spec(self, spec: SQLSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Database name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_create_table(self, table_name: str, columns: List[Dict[str, str]]) -> Tuple[bool, str, str]:
        code = f"CREATE TABLE {table_name} (\n"
        for col in columns:
            code += f"    {col['name']} {col['type']},\n"
        code += ");\n"
        return (True, code, "Generated")
    
    def generate_index(self, table: str, column: str, index_type: IndexType) -> Tuple[bool, str, str]:
        code = f"CREATE INDEX idx_{table}_{column} ON {table} USING {index_type.value} ({column});\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "dialects": len(SQLDialect)}

__all__ = ["SQLDialect", "IndexType", "SQLSpec", "SQLKnowledge"]
