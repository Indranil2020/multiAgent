"""
Health check implementation.

This module provides comprehensive health check capabilities for monitoring
service health, readiness, liveness, and dependency status.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from enum import Enum
import time


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    DEPENDENCY = "dependency"


@dataclass
class HealthCheckResult:
    """
    Result of a health check.
    
    Attributes:
        check_name: Name of the check
        status: Health status
        message: Status message
        timestamp: When check was performed
        duration_ms: Check duration in milliseconds
        details: Additional details
    """
    check_name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if result is valid."""
        return bool(self.check_name and self.message)


@dataclass
class HealthCheck:
    """
    A health check definition.
    
    Attributes:
        name: Check name
        check_type: Type of check
        check_function: Function to execute
        timeout_seconds: Timeout for check
        interval_seconds: Check interval
        enabled: Whether check is enabled
        last_result: Last check result
        last_check_time: Last check timestamp
    """
    name: str
    check_type: CheckType
    check_function: Optional[Callable[[], Tuple[bool, str]]]
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    enabled: bool = True
    last_result: Optional[HealthCheckResult] = None
    last_check_time: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if health check is valid."""
        return bool(self.name and self.check_function)
    
    def should_run(self) -> bool:
        """Check if health check should run based on interval."""
        if not self.enabled:
            return False
        
        if self.last_check_time == 0:
            return True
        
        elapsed = time.time() - self.last_check_time
        return elapsed >= self.interval_seconds


class HealthChecker:
    """
    Health check manager.
    
    Manages health checks for services and dependencies with
    liveness, readiness, and startup probes.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize health checker.
        
        Args:
            service_name: Name of this service
        """
        self.service_name = service_name
        self.checks: Dict[str, HealthCheck] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.startup_complete = False
    
    def register_check(
        self,
        name: str,
        check_type: CheckType,
        check_function: Callable[[], Tuple[bool, str]],
        timeout_seconds: float = 5.0,
        interval_seconds: float = 30.0
    ) -> Tuple[bool, str]:
        """
        Register a health check.
        
        Args:
            name: Check name
            check_type: Type of check
            check_function: Function that returns (is_healthy, message)
            timeout_seconds: Timeout for check
            interval_seconds: Check interval
        
        Returns:
            Tuple of (success, message)
        """
        if not name:
            return (False, "name cannot be empty")
        
        if not check_function:
            return (False, "check_function cannot be None")
        
        if timeout_seconds <= 0:
            return (False, "timeout_seconds must be positive")
        
        if interval_seconds <= 0:
            return (False, "interval_seconds must be positive")
        
        if name in self.checks:
            return (False, f"Check '{name}' already registered")
        
        check = HealthCheck(
            name=name,
            check_type=check_type,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds
        )
        
        if not check.is_valid():
            return (False, "Invalid health check created")
        
        self.checks[name] = check
        
        return (True, f"Health check '{name}' registered")
    
    def unregister_check(self, name: str) -> Tuple[bool, str]:
        """
        Unregister a health check.
        
        Args:
            name: Check name
        
        Returns:
            Tuple of (success, message)
        """
        if not name:
            return (False, "name cannot be empty")
        
        if name not in self.checks:
            return (False, f"Check '{name}' not found")
        
        self.checks.pop(name)
        
        return (True, f"Health check '{name}' unregistered")
    
    def run_check(self, name: str) -> Tuple[bool, Optional[HealthCheckResult], str]:
        """
        Run a specific health check.
        
        Args:
            name: Check name
        
        Returns:
            Tuple of (success, result or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if name not in self.checks:
            return (False, None, f"Check '{name}' not found")
        
        check = self.checks[name]
        
        if not check.enabled:
            return (False, None, f"Check '{name}' is disabled")
        
        # Run the check
        start_time = time.time()
        
        is_healthy = False
        check_message = ""
        
        # Execute check function
        if check.check_function:
            is_healthy, check_message = check.check_function()
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Determine status
        if is_healthy:
            status = HealthStatus.HEALTHY
        else:
            status = HealthStatus.UNHEALTHY
        
        # Create result
        result = HealthCheckResult(
            check_name=name,
            status=status,
            message=check_message,
            timestamp=time.time(),
            duration_ms=duration_ms
        )
        
        # Update check
        check.last_result = result
        check.last_check_time = time.time()
        
        return (True, result, "Check completed")
    
    def run_all_checks(
        self,
        check_type: Optional[CheckType] = None,
        force: bool = False
    ) -> Tuple[bool, List[HealthCheckResult], str]:
        """
        Run all health checks.
        
        Args:
            check_type: Filter by check type (None for all)
            force: Force run even if interval hasn't elapsed
        
        Returns:
            Tuple of (success, results list, message)
        """
        results = []
        
        for check_name, check in self.checks.items():
            # Filter by type if specified
            if check_type and check.check_type != check_type:
                continue
            
            # Check if should run
            if not force and not check.should_run():
                # Use last result if available
                if check.last_result:
                    results.append(check.last_result)
                continue
            
            # Run check
            success, result, msg = self.run_check(check_name)
            
            if success and result:
                results.append(result)
        
        return (True, results, f"Ran {len(results)} checks")
    
    def get_liveness(self) -> Tuple[bool, HealthStatus, str]:
        """
        Get liveness status.
        
        Liveness checks if the service is alive and should be restarted if not.
        
        Returns:
            Tuple of (success, status, message)
        """
        success, results, msg = self.run_all_checks(check_type=CheckType.LIVENESS)
        
        if not success:
            return (False, HealthStatus.UNKNOWN, msg)
        
        if not results:
            return (True, HealthStatus.HEALTHY, "No liveness checks configured")
        
        # Service is alive if any liveness check passes
        has_healthy = any(r.status == HealthStatus.HEALTHY for r in results)
        
        if has_healthy:
            return (True, HealthStatus.HEALTHY, "Service is alive")
        
        return (True, HealthStatus.UNHEALTHY, "Service is not alive")
    
    def get_readiness(self) -> Tuple[bool, HealthStatus, str]:
        """
        Get readiness status.
        
        Readiness checks if the service is ready to accept traffic.
        
        Returns:
            Tuple of (success, status, message)
        """
        success, results, msg = self.run_all_checks(check_type=CheckType.READINESS)
        
        if not success:
            return (False, HealthStatus.UNKNOWN, msg)
        
        if not results:
            return (True, HealthStatus.HEALTHY, "No readiness checks configured")
        
        # Service is ready if all readiness checks pass
        all_healthy = all(r.status == HealthStatus.HEALTHY for r in results)
        
        if all_healthy:
            return (True, HealthStatus.HEALTHY, "Service is ready")
        
        # Check if any are degraded
        has_degraded = any(r.status == HealthStatus.DEGRADED for r in results)
        
        if has_degraded:
            return (True, HealthStatus.DEGRADED, "Service is degraded")
        
        return (True, HealthStatus.UNHEALTHY, "Service is not ready")
    
    def check_dependency(
        self,
        dependency_name: str
    ) -> Tuple[bool, HealthStatus, str]:
        """
        Check a specific dependency.
        
        Args:
            dependency_name: Name of dependency to check
        
        Returns:
            Tuple of (success, status, message)
        """
        if not dependency_name:
            return (False, HealthStatus.UNKNOWN, "dependency_name cannot be empty")
        
        # Find dependency check
        dependency_check = None
        for check in self.checks.values():
            if check.check_type == CheckType.DEPENDENCY and check.name == dependency_name:
                dependency_check = check
                break
        
        if not dependency_check:
            return (False, HealthStatus.UNKNOWN, f"Dependency '{dependency_name}' not found")
        
        # Run check
        success, result, msg = self.run_check(dependency_check.name)
        
        if not success or not result:
            return (False, HealthStatus.UNKNOWN, msg)
        
        return (True, result.status, result.message)
    
    def enable_check(self, name: str) -> Tuple[bool, str]:
        """
        Enable a health check.
        
        Args:
            name: Check name
        
        Returns:
            Tuple of (success, message)
        """
        if not name:
            return (False, "name cannot be empty")
        
        if name not in self.checks:
            return (False, f"Check '{name}' not found")
        
        self.checks[name].enabled = True
        
        return (True, f"Check '{name}' enabled")
    
    def disable_check(self, name: str) -> Tuple[bool, str]:
        """
        Disable a health check.
        
        Args:
            name: Check name
        
        Returns:
            Tuple of (success, message)
        """
        if not name:
            return (False, "name cannot be empty")
        
        if name not in self.checks:
            return (False, f"Check '{name}' not found")
        
        self.checks[name].enabled = False
        
        return (True, f"Check '{name}' disabled")
    
    def get_overall_health(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Get overall health status.
        
        Returns:
            Tuple of (success, health_report dict, message)
        """
        # Run all checks
        success, all_results, msg = self.run_all_checks(force=False)
        
        if not success:
            return (False, {}, msg)
        
        # Get liveness and readiness
        liveness_success, liveness_status, liveness_msg = self.get_liveness()
        readiness_success, readiness_status, readiness_msg = self.get_readiness()
        
        # Determine overall status
        if not liveness_success or liveness_status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif not readiness_success or readiness_status == HealthStatus.UNHEALTHY:
            overall_status = HealthStatus.UNHEALTHY
        elif readiness_status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        self.overall_status = overall_status
        
        # Build report
        report = {
            "service_name": self.service_name,
            "overall_status": overall_status.value,
            "liveness": liveness_status.value,
            "readiness": readiness_status.value,
            "timestamp": time.time(),
            "checks": [
                {
                    "name": r.check_name,
                    "status": r.status.value,
                    "message": r.message,
                    "duration_ms": r.duration_ms
                }
                for r in all_results
            ]
        }
        
        return (True, report, "Health report generated")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        total_checks = len(self.checks)
        enabled_checks = sum(1 for c in self.checks.values() if c.enabled)
        
        # Count by type
        type_counts = {ct.value: 0 for ct in CheckType}
        for check in self.checks.values():
            type_counts[check.check_type.value] += 1
        
        return {
            "service_name": self.service_name,
            "overall_status": self.overall_status.value,
            "total_checks": total_checks,
            "enabled_checks": enabled_checks,
            "checks_by_type": type_counts
        }
