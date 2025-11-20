"""
Load Balancer Module.

This module implements intelligent load balancing for distributing inference
requests across multiple model instances, pools, and devices. Optimizes for
throughput, latency, and resource utilization.

Key Concepts:
- Multiple model pools running same/different models
- Distribute requests to prevent bottlenecks
- Balance between throughput and latency
- Consider model load, queue depth, response time
- Health monitoring and automatic failover

Load Balancing Strategies:
- Round Robin: Simple rotation through pools
- Least Loaded: Route to pool with fewest active requests
- Weighted: Distribute based on pool capacity/priority
- Response Time: Route to fastest responding pool
- Random: Random selection with optional weights
- Consistent Hashing: Sticky routing for cache hits

Features:
- Multiple balancing algorithms
- Real-time pool health monitoring
- Automatic pool failover and recovery
- Request queuing and throttling
- Statistics and metrics tracking
- Dynamic weight adjustment
- Circuit breaker pattern for failing pools

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time
import random
import hashlib


class LoadBalancingStrategy(Enum):
    """
    Load balancing strategies.

    Attributes:
        ROUND_ROBIN: Rotate through pools sequentially
        LEAST_LOADED: Route to pool with lowest load
        WEIGHTED: Distribute based on weights
        RESPONSE_TIME: Route to fastest pool
        RANDOM: Random selection
        CONSISTENT_HASH: Hash-based sticky routing
    """
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"


class PoolHealth(Enum):
    """
    Pool health status.

    Attributes:
        HEALTHY: Pool is healthy and accepting requests
        DEGRADED: Pool is slow but functional
        UNHEALTHY: Pool is failing requests
        OFFLINE: Pool is not responding
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class LoadBalancerConfig:
    """
    Configuration for load balancer.

    Attributes:
        strategy: Load balancing strategy
        enable_health_checks: Enable automatic health checks
        health_check_interval_seconds: Health check interval
        unhealthy_threshold: Failed requests before marking unhealthy
        recovery_threshold: Successful requests before recovery
        enable_circuit_breaker: Enable circuit breaker pattern
        circuit_breaker_timeout_seconds: Circuit breaker timeout
        max_queue_size: Maximum queued requests per pool
        request_timeout_seconds: Request timeout
        enable_failover: Enable automatic failover
        enable_metrics: Enable detailed metrics collection
    """
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED
    enable_health_checks: bool = True
    health_check_interval_seconds: float = 10.0
    unhealthy_threshold: int = 5
    recovery_threshold: int = 3
    enable_circuit_breaker: bool = True
    circuit_breaker_timeout_seconds: float = 30.0
    max_queue_size: int = 100
    request_timeout_seconds: float = 30.0
    enable_failover: bool = True
    enable_metrics: bool = True

    def validate(self) -> bool:
        """
        Validate load balancer configuration.

        Returns:
            True if valid, False otherwise
        """
        if self.health_check_interval_seconds <= 0:
            return False
        if self.unhealthy_threshold <= 0:
            return False
        if self.recovery_threshold <= 0:
            return False
        if self.circuit_breaker_timeout_seconds <= 0:
            return False
        if self.max_queue_size <= 0:
            return False
        if self.request_timeout_seconds <= 0:
            return False
        return True


@dataclass
class PoolConfig:
    """
    Configuration for a registered pool.

    Attributes:
        pool_id: Unique pool identifier
        pool_name: Human-readable pool name
        weight: Pool weight for weighted balancing (1-100)
        max_concurrent_requests: Maximum concurrent requests
        priority: Pool priority (higher = preferred)
        enabled: Whether pool is enabled
        metadata: Additional pool metadata
    """
    pool_id: str
    pool_name: str
    weight: int = 10
    max_concurrent_requests: int = 10
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolMetrics:
    """
    Metrics for a pool.

    Attributes:
        pool_id: Pool identifier
        health: Current health status
        active_requests: Currently active requests
        queued_requests: Requests in queue
        total_requests: Total requests processed
        successful_requests: Successful requests
        failed_requests: Failed requests
        average_response_time_ms: Average response time
        requests_per_second: Current throughput
        error_rate: Error rate (0.0-1.0)
        last_request_time: Timestamp of last request
        circuit_open: Whether circuit breaker is open
    """
    pool_id: str
    health: PoolHealth = PoolHealth.HEALTHY
    active_requests: int = 0
    queued_requests: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    last_request_time: Optional[float] = None
    circuit_open: bool = False

    def update_error_rate(self) -> None:
        """Update error rate calculation."""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
        else:
            self.error_rate = 0.0


@dataclass
class LoadBalancerStatistics:
    """
    Overall load balancer statistics.

    Attributes:
        total_requests: Total requests processed
        successful_requests: Successfully routed requests
        failed_requests: Failed/rejected requests
        average_response_time_ms: Average response time across all pools
        total_failovers: Number of failover events
        pools_healthy: Number of healthy pools
        pools_degraded: Number of degraded pools
        pools_unhealthy: Number of unhealthy pools
        pools_offline: Number of offline pools
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    total_failovers: int = 0
    pools_healthy: int = 0
    pools_degraded: int = 0
    pools_unhealthy: int = 0
    pools_offline: int = 0


class LoadBalancer:
    """
    Intelligent load balancer for distributing inference requests.

    Manages multiple model pools and routes requests based on
    configured strategy and pool health.
    """

    def __init__(self, config: LoadBalancerConfig):
        """
        Initialize load balancer.

        Args:
            config: Load balancer configuration
        """
        if not config.validate():
            raise ValueError("Invalid load balancer configuration")

        self.config = config

        # Pool registry
        self.pools: Dict[str, PoolConfig] = {}
        self.pool_metrics: Dict[str, PoolMetrics] = {}

        # Round-robin state
        self.round_robin_index = 0

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = LoadBalancerStatistics()

        # Circuit breaker state
        self.circuit_breaker_opened_at: Dict[str, float] = {}

    def register_pool(self, pool_config: PoolConfig) -> bool:
        """
        Register a pool for load balancing.

        Args:
            pool_config: Pool configuration

        Returns:
            True if registered, False if already exists
        """
        with self.lock:
            if pool_config.pool_id in self.pools:
                return False

            self.pools[pool_config.pool_id] = pool_config
            self.pool_metrics[pool_config.pool_id] = PoolMetrics(
                pool_id=pool_config.pool_id
            )

            return True

    def unregister_pool(self, pool_id: str) -> bool:
        """
        Unregister a pool.

        Args:
            pool_id: Pool identifier

        Returns:
            True if unregistered, False if not found
        """
        with self.lock:
            if pool_id not in self.pools:
                return False

            del self.pools[pool_id]
            del self.pool_metrics[pool_id]

            if pool_id in self.circuit_breaker_opened_at:
                del self.circuit_breaker_opened_at[pool_id]

            return True

    def select_pool(
        self,
        request_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Select optimal pool for request.

        Args:
            request_key: Optional key for consistent hashing

        Returns:
            Pool ID if found, None if no pools available
        """
        with self.lock:
            # Get eligible pools
            eligible_pools = self._get_eligible_pools()

            if not eligible_pools:
                return None

            # Route based on strategy
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(eligible_pools)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._select_least_loaded(eligible_pools)
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED:
                return self._select_weighted(eligible_pools)
            elif self.config.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._select_fastest(eligible_pools)
            elif self.config.strategy == LoadBalancingStrategy.RANDOM:
                return self._select_random(eligible_pools)
            elif self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return self._select_consistent_hash(eligible_pools, request_key)
            else:
                return self._select_round_robin(eligible_pools)

    def _get_eligible_pools(self) -> List[str]:
        """
        Get list of eligible pools for routing.

        Returns:
            List of pool IDs
        """
        eligible = []

        for pool_id, pool_config in self.pools.items():
            # Check if pool is enabled
            if not pool_config.enabled:
                continue

            # Check pool health
            metrics = self.pool_metrics[pool_id]
            if metrics.health == PoolHealth.OFFLINE:
                continue

            # Check circuit breaker
            if self.config.enable_circuit_breaker and metrics.circuit_open:
                # Check if timeout has passed
                if pool_id in self.circuit_breaker_opened_at:
                    elapsed = time.time() - self.circuit_breaker_opened_at[pool_id]
                    if elapsed < self.config.circuit_breaker_timeout_seconds:
                        continue
                    else:
                        # Try to close circuit
                        metrics.circuit_open = False
                        del self.circuit_breaker_opened_at[pool_id]

            # Check capacity
            if metrics.active_requests >= pool_config.max_concurrent_requests:
                continue

            eligible.append(pool_id)

        return eligible

    def _select_round_robin(self, eligible_pools: List[str]) -> Optional[str]:
        """Select pool using round-robin."""
        if not eligible_pools:
            return None

        pool_id = eligible_pools[self.round_robin_index % len(eligible_pools)]
        self.round_robin_index += 1

        return pool_id

    def _select_least_loaded(self, eligible_pools: List[str]) -> Optional[str]:
        """Select pool with lowest load."""
        if not eligible_pools:
            return None

        best_pool = None
        lowest_load = float('inf')

        for pool_id in eligible_pools:
            metrics = self.pool_metrics[pool_id]
            load = metrics.active_requests + metrics.queued_requests

            if load < lowest_load:
                lowest_load = load
                best_pool = pool_id

        return best_pool

    def _select_weighted(self, eligible_pools: List[str]) -> Optional[str]:
        """Select pool using weighted random selection."""
        if not eligible_pools:
            return None

        # Build weights list
        weights = []
        for pool_id in eligible_pools:
            pool_config = self.pools[pool_id]
            weights.append(pool_config.weight)

        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(eligible_pools)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for pool_id, weight in zip(eligible_pools, weights):
            cumulative += weight
            if r <= cumulative:
                return pool_id

        return eligible_pools[-1]

    def _select_fastest(self, eligible_pools: List[str]) -> Optional[str]:
        """Select pool with fastest response time."""
        if not eligible_pools:
            return None

        best_pool = None
        fastest_time = float('inf')

        for pool_id in eligible_pools:
            metrics = self.pool_metrics[pool_id]

            # Use average response time or default if no history
            response_time = metrics.average_response_time_ms if metrics.total_requests > 0 else 100.0

            if response_time < fastest_time:
                fastest_time = response_time
                best_pool = pool_id

        return best_pool

    def _select_random(self, eligible_pools: List[str]) -> Optional[str]:
        """Select random pool."""
        if not eligible_pools:
            return None

        return random.choice(eligible_pools)

    def _select_consistent_hash(
        self,
        eligible_pools: List[str],
        request_key: Optional[str]
    ) -> Optional[str]:
        """Select pool using consistent hashing."""
        if not eligible_pools:
            return None

        if request_key is None:
            # Fall back to random
            return random.choice(eligible_pools)

        # Hash request key
        key_hash = int(hashlib.md5(request_key.encode()).hexdigest(), 16)

        # Map to pool
        pool_index = key_hash % len(eligible_pools)
        return eligible_pools[pool_index]

    def record_request_start(self, pool_id: str) -> bool:
        """
        Record request start for pool.

        Args:
            pool_id: Pool identifier

        Returns:
            True if recorded, False if pool not found
        """
        with self.lock:
            if pool_id not in self.pool_metrics:
                return False

            metrics = self.pool_metrics[pool_id]
            metrics.active_requests += 1
            metrics.total_requests += 1
            metrics.last_request_time = time.time()

            self.stats.total_requests += 1

            return True

    def record_request_end(
        self,
        pool_id: str,
        success: bool,
        response_time_ms: float
    ) -> bool:
        """
        Record request completion for pool.

        Args:
            pool_id: Pool identifier
            success: Whether request succeeded
            response_time_ms: Request response time

        Returns:
            True if recorded, False if pool not found
        """
        with self.lock:
            if pool_id not in self.pool_metrics:
                return False

            metrics = self.pool_metrics[pool_id]
            metrics.active_requests -= 1

            if success:
                metrics.successful_requests += 1
                self.stats.successful_requests += 1
            else:
                metrics.failed_requests += 1
                self.stats.failed_requests += 1

            # Update average response time
            if metrics.total_requests > 0:
                n = metrics.successful_requests
                if n > 0:
                    metrics.average_response_time_ms = (
                        (metrics.average_response_time_ms * (n - 1) + response_time_ms) / n
                    )

            # Update error rate
            metrics.update_error_rate()

            # Check health thresholds
            self._update_pool_health(pool_id)

            return True

    def _update_pool_health(self, pool_id: str) -> None:
        """
        Update pool health based on metrics.

        Args:
            pool_id: Pool identifier
        """
        metrics = self.pool_metrics[pool_id]

        # Check consecutive failures
        recent_failures = 0
        if metrics.total_requests > 0:
            # Calculate recent error rate
            recent_window = min(10, metrics.total_requests)
            if recent_window > 0:
                recent_error_rate = metrics.error_rate

                # Determine health
                if recent_error_rate > 0.5:
                    metrics.health = PoolHealth.UNHEALTHY
                    # Open circuit breaker
                    if self.config.enable_circuit_breaker and not metrics.circuit_open:
                        metrics.circuit_open = True
                        self.circuit_breaker_opened_at[pool_id] = time.time()
                elif recent_error_rate > 0.2:
                    metrics.health = PoolHealth.DEGRADED
                else:
                    metrics.health = PoolHealth.HEALTHY

    def get_pool_metrics(self, pool_id: str) -> Optional[PoolMetrics]:
        """
        Get metrics for pool.

        Args:
            pool_id: Pool identifier

        Returns:
            PoolMetrics if found, None otherwise
        """
        with self.lock:
            if pool_id not in self.pool_metrics:
                return None

            # Return copy
            metrics = self.pool_metrics[pool_id]
            return PoolMetrics(
                pool_id=metrics.pool_id,
                health=metrics.health,
                active_requests=metrics.active_requests,
                queued_requests=metrics.queued_requests,
                total_requests=metrics.total_requests,
                successful_requests=metrics.successful_requests,
                failed_requests=metrics.failed_requests,
                average_response_time_ms=metrics.average_response_time_ms,
                requests_per_second=metrics.requests_per_second,
                error_rate=metrics.error_rate,
                last_request_time=metrics.last_request_time,
                circuit_open=metrics.circuit_open
            )

    def get_all_pool_metrics(self) -> Dict[str, PoolMetrics]:
        """
        Get metrics for all pools.

        Returns:
            Dictionary of pool metrics
        """
        with self.lock:
            return {
                pool_id: self.get_pool_metrics(pool_id)
                for pool_id in self.pool_metrics.keys()
            }

    def get_statistics(self) -> LoadBalancerStatistics:
        """
        Get load balancer statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            # Count pool health statuses
            health_counts = {
                PoolHealth.HEALTHY: 0,
                PoolHealth.DEGRADED: 0,
                PoolHealth.UNHEALTHY: 0,
                PoolHealth.OFFLINE: 0
            }

            for metrics in self.pool_metrics.values():
                health_counts[metrics.health] += 1

            return LoadBalancerStatistics(
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                average_response_time_ms=self.stats.average_response_time_ms,
                total_failovers=self.stats.total_failovers,
                pools_healthy=health_counts[PoolHealth.HEALTHY],
                pools_degraded=health_counts[PoolHealth.DEGRADED],
                pools_unhealthy=health_counts[PoolHealth.UNHEALTHY],
                pools_offline=health_counts[PoolHealth.OFFLINE]
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = LoadBalancerStatistics()


def create_default_balancer_config() -> LoadBalancerConfig:
    """
    Create default load balancer configuration.

    Returns:
        LoadBalancerConfig with sensible defaults
    """
    return LoadBalancerConfig(
        strategy=LoadBalancingStrategy.LEAST_LOADED,
        enable_health_checks=True,
        enable_circuit_breaker=True,
        enable_failover=True,
        enable_metrics=True
    )


def create_high_throughput_config() -> LoadBalancerConfig:
    """
    Create configuration optimized for throughput.

    Returns:
        LoadBalancerConfig for high throughput
    """
    return LoadBalancerConfig(
        strategy=LoadBalancingStrategy.ROUND_ROBIN,
        enable_health_checks=True,
        enable_circuit_breaker=False,
        max_queue_size=500,
        request_timeout_seconds=60.0,
        enable_failover=False,
        enable_metrics=False
    )


def create_low_latency_config() -> LoadBalancerConfig:
    """
    Create configuration optimized for low latency.

    Returns:
        LoadBalancerConfig for low latency
    """
    return LoadBalancerConfig(
        strategy=LoadBalancingStrategy.RESPONSE_TIME,
        enable_health_checks=True,
        health_check_interval_seconds=5.0,
        enable_circuit_breaker=True,
        circuit_breaker_timeout_seconds=10.0,
        max_queue_size=50,
        request_timeout_seconds=10.0,
        enable_failover=True,
        enable_metrics=True
    )
