"""
Web database integration patterns and specifications.

This module provides comprehensive knowledge for database integration in
web applications including ORM patterns, migrations, connection pooling,
query optimization, and caching strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class ORMFramework(Enum):
    """ORM frameworks."""
    SQLALCHEMY = "sqlalchemy"
    DJANGO_ORM = "django_orm"
    PRISMA = "prisma"
    SEQUELIZE = "sequelize"
    TYPEORM = "typeorm"
    MONGOOSE = "mongoose"


class MigrationTool(Enum):
    """Database migration tools."""
    ALEMBIC = "alembic"
    DJANGO_MIGRATIONS = "django_migrations"
    FLYWAY = "flyway"
    LIQUIBASE = "liquibase"
    KNEX = "knex"


class CachingStrategy(Enum):
    """Caching strategies."""
    REDIS = "redis"
    MEMCACHED = "memcached"
    IN_MEMORY = "in_memory"
    CDN = "cdn"


@dataclass
class ModelSpec:
    """
    Database model specification.
    
    Attributes:
        name: Model name
        table_name: Database table name
        fields: Model fields
        indexes: Database indexes
        relationships: Model relationships
        constraints: Database constraints
    """
    name: str
    table_name: str
    fields: List[Dict[str, Any]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if model spec is valid."""
        return bool(self.name and self.table_name and self.fields)


@dataclass
class QueryOptimization:
    """
    Query optimization recommendation.
    
    Attributes:
        query_type: Type of query
        issue: Performance issue
        recommendation: Optimization recommendation
        impact: Expected impact
    """
    query_type: str
    issue: str
    recommendation: str
    impact: str  # "high", "medium", "low"


class WebDatabaseKnowledge:
    """
    Web database integration knowledge base.
    
    Provides patterns, best practices, and specifications for
    integrating databases with web applications.
    """
    
    def __init__(self):
        """Initialize web database knowledge."""
        self.orm_patterns: Dict[str, str] = {}
        self.query_patterns: List[str] = []
        self.optimization_rules: List[str] = []
        self._initialize_orm_patterns()
        self._initialize_query_patterns()
        self._initialize_optimization()
    
    def _initialize_orm_patterns(self) -> None:
        """Initialize ORM patterns."""
        self.orm_patterns = {
            "active_record": "Model contains both data and behavior",
            "data_mapper": "Separate data access from business logic",
            "repository": "Abstract data access layer",
            "unit_of_work": "Track changes and commit together",
            "lazy_loading": "Load related data on access",
            "eager_loading": "Load related data upfront",
            "identity_map": "Ensure single instance per entity"
        }
    
    def _initialize_query_patterns(self) -> None:
        """Initialize query patterns."""
        self.query_patterns = [
            "Use parameterized queries to prevent SQL injection",
            "Use indexes for frequently queried columns",
            "Avoid N+1 queries with eager loading",
            "Use database transactions for consistency",
            "Implement connection pooling",
            "Use read replicas for read-heavy workloads",
            "Cache frequently accessed data",
            "Paginate large result sets",
            "Use database-specific optimizations",
            "Monitor slow queries"
        ]
    
    def _initialize_optimization(self) -> None:
        """Initialize optimization rules."""
        self.optimization_rules = [
            "Add indexes on foreign keys",
            "Add indexes on frequently filtered columns",
            "Use composite indexes for multi-column queries",
            "Avoid SELECT * - specify columns",
            "Use EXPLAIN to analyze query plans",
            "Denormalize for read performance",
            "Use materialized views for complex queries",
            "Partition large tables",
            "Archive old data",
            "Use database connection pooling"
        ]
    
    def validate_model_spec(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, List[str], str]:
        """
        Validate a model specification.
        
        Args:
            spec: Model specification
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        if not spec.is_valid():
            errors.append("Invalid model specification")
        
        # Check for primary key
        has_pk = any(
            field.get("primary_key", False)
            for field in spec.fields
        )
        if not has_pk:
            errors.append("Model must have a primary key")
        
        # Check for duplicate field names
        field_names = [f["name"] for f in spec.fields]
        if len(field_names) != len(set(field_names)):
            errors.append("Duplicate field names found")
        
        # Check for timestamps
        has_created = any(
            field["name"] in ["created_at", "created"]
            for field in spec.fields
        )
        has_updated = any(
            field["name"] in ["updated_at", "updated", "modified_at"]
            for field in spec.fields
        )
        
        if not has_created or not has_updated:
            errors.append("Model should have created_at and updated_at timestamps")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "Model specification valid")
    
    def generate_model_code(
        self,
        spec: ModelSpec,
        orm: ORMFramework
    ) -> Tuple[bool, str, str]:
        """
        Generate model code from specification.
        
        Args:
            spec: Model specification
            orm: Target ORM framework
        
        Returns:
            Tuple of (success, code, message)
        """
        if not spec.is_valid():
            return (False, "", "Invalid model specification")
        
        if orm == ORMFramework.SQLALCHEMY:
            return self._generate_sqlalchemy_model(spec)
        elif orm == ORMFramework.DJANGO_ORM:
            return self._generate_django_model(spec)
        elif orm == ORMFramework.PRISMA:
            return self._generate_prisma_model(spec)
        else:
            return (False, "", f"ORM {orm.value} not supported")
    
    def _generate_sqlalchemy_model(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, str, str]:
        """Generate SQLAlchemy model code."""
        code = "from sqlalchemy import Column, Integer, String, DateTime, ForeignKey\n"
        code += "from sqlalchemy.orm import relationship\n"
        code += "from sqlalchemy.ext.declarative import declarative_base\n"
        code += "from datetime import datetime\n\n"
        code += "Base = declarative_base()\n\n"
        code += f"class {spec.name}(Base):\n"
        code += f"    __tablename__ = '{spec.table_name}'\n\n"
        
        # Generate fields
        for field in spec.fields:
            field_name = field["name"]
            field_type = field.get("type", "String")
            
            if field.get("primary_key", False):
                code += f"    {field_name} = Column(Integer, primary_key=True)\n"
            else:
                code += f"    {field_name} = Column({field_type})\n"
        
        # Generate relationships
        for rel in spec.relationships:
            rel_name = rel["name"]
            code += f"    {rel_name} = relationship('{rel['model']}')\n"
        
        return (True, code, "SQLAlchemy model generated")
    
    def _generate_django_model(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, str, str]:
        """Generate Django model code."""
        code = "from django.db import models\n\n"
        code += f"class {spec.name}(models.Model):\n"
        
        # Generate fields
        for field in spec.fields:
            field_name = field["name"]
            field_type = field.get("type", "CharField")
            
            if field.get("primary_key", False):
                code += f"    {field_name} = models.AutoField(primary_key=True)\n"
            else:
                code += f"    {field_name} = models.{field_type}()\n"
        
        code += "\n    class Meta:\n"
        code += f"        db_table = '{spec.table_name}'\n"
        
        return (True, code, "Django model generated")
    
    def _generate_prisma_model(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, str, str]:
        """Generate Prisma schema."""
        code = f"model {spec.name} {{\n"
        
        # Generate fields
        for field in spec.fields:
            field_name = field["name"]
            field_type = field.get("type", "String")
            
            if field.get("primary_key", False):
                code += f"  {field_name} Int @id @default(autoincrement())\n"
            else:
                code += f"  {field_name} {field_type}\n"
        
        code += "}\n"
        
        return (True, code, "Prisma model generated")
    
    def generate_migration_code(
        self,
        spec: ModelSpec,
        tool: MigrationTool
    ) -> Tuple[bool, str, str]:
        """
        Generate migration code.
        
        Args:
            spec: Model specification
            tool: Migration tool
        
        Returns:
            Tuple of (success, code, message)
        """
        if tool == MigrationTool.ALEMBIC:
            return self._generate_alembic_migration(spec)
        elif tool == MigrationTool.DJANGO_MIGRATIONS:
            return self._generate_django_migration(spec)
        else:
            return (False, "", f"Migration tool {tool.value} not supported")
    
    def _generate_alembic_migration(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, str, str]:
        """Generate Alembic migration."""
        code = "from alembic import op\n"
        code += "import sqlalchemy as sa\n\n"
        code += "def upgrade():\n"
        code += f"    op.create_table('{spec.table_name}',\n"
        
        for field in spec.fields:
            field_name = field["name"]
            field_type = field.get("type", "String")
            code += f"        sa.Column('{field_name}', sa.{field_type}()),\n"
        
        code += "    )\n\n"
        code += "def downgrade():\n"
        code += f"    op.drop_table('{spec.table_name}')\n"
        
        return (True, code, "Alembic migration generated")
    
    def _generate_django_migration(
        self,
        spec: ModelSpec
    ) -> Tuple[bool, str, str]:
        """Generate Django migration."""
        code = "from django.db import migrations, models\n\n"
        code += "class Migration(migrations.Migration):\n"
        code += "    operations = [\n"
        code += "        migrations.CreateModel(\n"
        code += f"            name='{spec.name}',\n"
        code += "            fields=[\n"
        
        for field in spec.fields:
            field_name = field["name"]
            field_type = field.get("type", "CharField")
            code += f"                ('{field_name}', models.{field_type}()),\n"
        
        code += "            ],\n"
        code += "        ),\n"
        code += "    ]\n"
        
        return (True, code, "Django migration generated")
    
    def recommend_indexes(
        self,
        spec: ModelSpec,
        query_patterns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Recommend indexes based on query patterns.
        
        Args:
            spec: Model specification
            query_patterns: Common query patterns
        
        Returns:
            List of recommended indexes
        """
        indexes = []
        
        # Always index foreign keys
        for field in spec.fields:
            if field.get("foreign_key", False):
                indexes.append({
                    "columns": [field["name"]],
                    "type": "btree",
                    "reason": "Foreign key index"
                })
        
        # Index frequently filtered columns
        for pattern in query_patterns:
            if "WHERE" in pattern.upper():
                # Extract column names (simplified)
                pass
        
        # Add timestamp indexes if present
        for field in spec.fields:
            if field["name"] in ["created_at", "updated_at"]:
                indexes.append({
                    "columns": [field["name"]],
                    "type": "btree",
                    "reason": "Timestamp index for sorting/filtering"
                })
        
        return indexes
    
    def detect_n_plus_one(
        self,
        query_log: List[str]
    ) -> Tuple[bool, List[str], str]:
        """
        Detect N+1 query problems.
        
        Args:
            query_log: List of executed queries
        
        Returns:
            Tuple of (has_issue, problematic_queries, message)
        """
        issues = []
        
        # Look for repeated similar queries
        query_counts: Dict[str, int] = {}
        
        for query in query_log:
            # Normalize query (remove specific IDs)
            normalized = query  # Would normalize in real implementation
            query_counts[normalized] = query_counts.get(normalized, 0) + 1
        
        # Find queries executed many times
        for query, count in query_counts.items():
            if count > 10:
                issues.append(f"Query executed {count} times: {query[:100]}")
        
        if issues:
            return (True, issues, f"Found {len(issues)} potential N+1 issues")
        
        return (False, [], "No N+1 issues detected")
    
    def recommend_caching_strategy(
        self,
        read_write_ratio: float,
        data_volatility: str,  # "high", "medium", "low"
        data_size: str  # "small", "medium", "large"
    ) -> Tuple[bool, CachingStrategy, Dict[str, Any], str]:
        """
        Recommend caching strategy.
        
        Args:
            read_write_ratio: Ratio of reads to writes
            data_volatility: How often data changes
            data_size: Size of data
        
        Returns:
            Tuple of (success, strategy, config, message)
        """
        if read_write_ratio > 10 and data_volatility == "low":
            strategy = CachingStrategy.REDIS
            config = {
                "ttl": 3600,  # 1 hour
                "invalidation": "time_based"
            }
        elif read_write_ratio > 5:
            strategy = CachingStrategy.REDIS
            config = {
                "ttl": 300,  # 5 minutes
                "invalidation": "event_based"
            }
        elif data_size == "small":
            strategy = CachingStrategy.IN_MEMORY
            config = {
                "max_size": 1000,
                "eviction": "lru"
            }
        else:
            strategy = CachingStrategy.REDIS
            config = {
                "ttl": 60,
                "invalidation": "write_through"
            }
        
        return (True, strategy, config, f"Recommended {strategy.value}")
    
    def estimate_query_complexity(
        self,
        num_joins: int,
        has_subquery: bool,
        has_aggregation: bool,
        result_size: int
    ) -> Tuple[bool, int, str]:
        """
        Estimate query complexity.
        
        Args:
            num_joins: Number of joins
            has_subquery: Has subquery
            has_aggregation: Has aggregation
            result_size: Expected result size
        
        Returns:
            Tuple of (success, complexity_score, message)
        """
        complexity = 0
        
        # Add complexity for joins
        complexity += num_joins * 2
        
        # Add complexity for subqueries
        if has_subquery:
            complexity += 3
        
        # Add complexity for aggregation
        if has_aggregation:
            complexity += 2
        
        # Add complexity for large results
        if result_size > 10000:
            complexity += 4
        elif result_size > 1000:
            complexity += 2
        elif result_size > 100:
            complexity += 1
        
        return (True, complexity, f"Estimated complexity: {complexity}")
    
    def get_connection_pool_config(
        self,
        expected_concurrent_users: int,
        avg_query_time_ms: float
    ) -> Dict[str, int]:
        """
        Get recommended connection pool configuration.
        
        Args:
            expected_concurrent_users: Expected concurrent users
            avg_query_time_ms: Average query time in milliseconds
        
        Returns:
            Connection pool configuration
        """
        # Calculate pool size based on Little's Law
        # Pool size = (concurrent users * avg query time) / 1000
        min_pool_size = max(5, int(expected_concurrent_users * avg_query_time_ms / 1000))
        max_pool_size = min_pool_size * 2
        
        return {
            "min_size": min_pool_size,
            "max_size": max_pool_size,
            "timeout": 30,
            "max_overflow": 10
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get web database knowledge statistics."""
        return {
            "orm_patterns": len(self.orm_patterns),
            "query_patterns": len(self.query_patterns),
            "optimization_rules": len(self.optimization_rules),
            "supported_orms": len(ORMFramework),
            "migration_tools": len(MigrationTool)
        }


__all__ = [
    # Enums
    "ORMFramework",
    "MigrationTool",
    "CachingStrategy",
    # Data classes
    "ModelSpec",
    "QueryOptimization",
    # Main class
    "WebDatabaseKnowledge",
]
