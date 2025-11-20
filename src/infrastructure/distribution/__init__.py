"""
Distribution clients for infrastructure.

This package provides distributed systems integration including
Kafka messaging, Redis caching, Prefect workflows, and Dask computing.
"""

from .redis_client import (
    RedisDataType,
    RedisConfig,
    RedisClient,
)

from .kafka_client import (
    MessageFormat,
    KafkaConfig,
    KafkaMessage,
    KafkaClient,
)

from .prefect_client import (
    FlowRunState,
    TaskRunState,
    PrefectConfig,
    FlowRun,
    TaskRun,
    PrefectClient,
)

from .dask_client import (
    TaskStatus,
    WorkerStatus,
    DaskConfig,
    DaskTask,
    Worker,
    DaskClient,
)

__all__ = [
    # Redis
    "RedisDataType",
    "RedisConfig",
    "RedisClient",
    # Kafka
    "MessageFormat",
    "KafkaConfig",
    "KafkaMessage",
    "KafkaClient",
    # Prefect
    "FlowRunState",
    "TaskRunState",
    "PrefectConfig",
    "FlowRun",
    "TaskRun",
    "PrefectClient",
    # Dask
    "TaskStatus",
    "WorkerStatus",
    "DaskConfig",
    "DaskTask",
    "Worker",
    "DaskClient",
]
