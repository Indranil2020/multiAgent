"""
Risk analysis for task decomposition.

This module assesses risk levels of tasks to identify high-risk components
that may need special attention or additional verification.
"""

from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RiskCategory(Enum):
    """Categories of risk."""
    COMPLEXITY = "complexity"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA_INTEGRITY = "data_integrity"


@dataclass
class RiskMetrics:
    """
    Risk assessment results for a task.
    
    Attributes:
        overall_risk_level: Overall risk level
        complexity_risk: Risk from complexity
        integration_risk: Risk from integration points
        security_risk: Risk from security concerns
        performance_risk: Risk from performance requirements
        risk_score: Numerical risk score (0.0 to 1.0)
        risk_factors: List of identified risk factors
    """
    overall_risk_level: RiskLevel
    complexity_risk: RiskLevel
    integration_risk: RiskLevel
    security_risk: RiskLevel
    performance_risk: RiskLevel
    risk_score: float
    risk_factors: List[str]
    
    def validate_risk_score(self) -> bool:
        """Validate risk score is in valid range."""
        return 0.0 <= self.risk_score <= 1.0
    
    def is_valid(self) -> bool:
        """Check if metrics are valid."""
        return self.validate_risk_score()


class RiskAnalyzer:
    """
    Analyzes risk levels of tasks.
    
    This analyzer assesses various risk factors to identify tasks that may
    require additional attention, verification, or specialized handling.
    """
    
    # Risk thresholds
    LOW_RISK_THRESHOLD = 0.3
    MEDIUM_RISK_THRESHOLD = 0.6
    HIGH_RISK_THRESHOLD = 0.8
    
    def __init__(self):
        """Initialize risk analyzer."""
        pass
    
    def analyze(
        self,
        description: str,
        estimated_complexity: int,
        dependency_count: int,
        has_security_requirements: bool,
        has_performance_requirements: bool,
        is_external_integration: bool,
        handles_sensitive_data: bool
    ) -> Tuple[bool, Optional[RiskMetrics], str]:
        """
        Analyze task risk.
        
        Args:
            description: Task description
            estimated_complexity: Estimated cyclomatic complexity
            dependency_count: Number of dependencies
            has_security_requirements: Whether task has security requirements
            has_performance_requirements: Whether task has performance requirements
            is_external_integration: Whether task integrates with external systems
            handles_sensitive_data: Whether task handles sensitive data
        
        Returns:
            Tuple of (success, metrics or None, message)
        """
        if not description:
            return (False, None, "description cannot be empty")
        
        if estimated_complexity < 0:
            return (False, None, "estimated_complexity cannot be negative")
        
        if dependency_count < 0:
            return (False, None, "dependency_count cannot be negative")
        
        risk_factors = []
        
        # Assess complexity risk
        complexity_risk, complexity_factors = self._assess_complexity_risk(
            estimated_complexity
        )
        risk_factors.extend(complexity_factors)
        
        # Assess integration risk
        integration_risk, integration_factors = self._assess_integration_risk(
            dependency_count,
            is_external_integration
        )
        risk_factors.extend(integration_factors)
        
        # Assess security risk
        security_risk, security_factors = self._assess_security_risk(
            has_security_requirements,
            handles_sensitive_data,
            description
        )
        risk_factors.extend(security_factors)
        
        # Assess performance risk
        performance_risk, performance_factors = self._assess_performance_risk(
            has_performance_requirements,
            estimated_complexity,
            description
        )
        risk_factors.extend(performance_factors)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            complexity_risk,
            integration_risk,
            security_risk,
            performance_risk
        )
        
        # Determine overall risk level
        overall_risk_level = self._risk_score_to_level(risk_score)
        
        metrics = RiskMetrics(
            overall_risk_level=overall_risk_level,
            complexity_risk=complexity_risk,
            integration_risk=integration_risk,
            security_risk=security_risk,
            performance_risk=performance_risk,
            risk_score=risk_score,
            risk_factors=risk_factors
        )
        
        if not metrics.is_valid():
            return (False, None, "Invalid risk metrics calculated")
        
        return (True, metrics, "Risk analysis complete")
    
    def _assess_complexity_risk(
        self,
        estimated_complexity: int
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess risk from complexity.
        
        Args:
            estimated_complexity: Estimated cyclomatic complexity
        
        Returns:
            Tuple of (risk level, risk factors)
        """
        factors = []
        
        if estimated_complexity <= 5:
            risk_level = RiskLevel.LOW
        elif estimated_complexity <= 10:
            risk_level = RiskLevel.MEDIUM
            factors.append(f"Moderate complexity ({estimated_complexity})")
        elif estimated_complexity <= 15:
            risk_level = RiskLevel.HIGH
            factors.append(f"High complexity ({estimated_complexity})")
        else:
            risk_level = RiskLevel.CRITICAL
            factors.append(f"Critical complexity ({estimated_complexity})")
        
        return (risk_level, factors)
    
    def _assess_integration_risk(
        self,
        dependency_count: int,
        is_external_integration: bool
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess risk from integration points.
        
        Args:
            dependency_count: Number of dependencies
            is_external_integration: Whether integrating with external systems
        
        Returns:
            Tuple of (risk level, risk factors)
        """
        factors = []
        
        # Base risk from dependency count
        if dependency_count == 0:
            risk_level = RiskLevel.LOW
        elif dependency_count <= 3:
            risk_level = RiskLevel.LOW
        elif dependency_count <= 6:
            risk_level = RiskLevel.MEDIUM
            factors.append(f"Multiple dependencies ({dependency_count})")
        else:
            risk_level = RiskLevel.HIGH
            factors.append(f"Many dependencies ({dependency_count})")
        
        # Increase risk for external integrations
        if is_external_integration:
            factors.append("External system integration")
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MEDIUM
            elif risk_level == RiskLevel.MEDIUM:
                risk_level = RiskLevel.HIGH
            elif risk_level == RiskLevel.HIGH:
                risk_level = RiskLevel.CRITICAL
        
        return (risk_level, factors)
    
    def _assess_security_risk(
        self,
        has_security_requirements: bool,
        handles_sensitive_data: bool,
        description: str
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess risk from security concerns.
        
        Args:
            has_security_requirements: Whether task has security requirements
            handles_sensitive_data: Whether task handles sensitive data
            description: Task description
        
        Returns:
            Tuple of (risk level, risk factors)
        """
        factors = []
        risk_level = RiskLevel.LOW
        
        # Check for security-related keywords in description
        desc_lower = description.lower()
        security_keywords = [
            'authentication', 'authorization', 'password', 'token',
            'encryption', 'decrypt', 'certificate', 'credential',
            'permission', 'access control', 'security'
        ]
        
        has_security_keywords = any(kw in desc_lower for kw in security_keywords)
        
        if has_security_requirements or has_security_keywords:
            risk_level = RiskLevel.MEDIUM
            factors.append("Security requirements present")
        
        if handles_sensitive_data:
            risk_level = RiskLevel.HIGH
            factors.append("Handles sensitive data")
        
        if has_security_requirements and handles_sensitive_data:
            risk_level = RiskLevel.CRITICAL
            factors.append("Critical security component")
        
        return (risk_level, factors)
    
    def _assess_performance_risk(
        self,
        has_performance_requirements: bool,
        estimated_complexity: int,
        description: str
    ) -> Tuple[RiskLevel, List[str]]:
        """
        Assess risk from performance requirements.
        
        Args:
            has_performance_requirements: Whether task has performance requirements
            estimated_complexity: Estimated complexity
            description: Task description
        
        Returns:
            Tuple of (risk level, risk factors)
        """
        factors = []
        risk_level = RiskLevel.LOW
        
        # Check for performance-related keywords
        desc_lower = description.lower()
        performance_keywords = [
            'optimize', 'performance', 'fast', 'efficient',
            'cache', 'parallel', 'concurrent', 'real-time',
            'latency', 'throughput'
        ]
        
        has_performance_keywords = any(kw in desc_lower for kw in performance_keywords)
        
        if has_performance_requirements or has_performance_keywords:
            risk_level = RiskLevel.MEDIUM
            factors.append("Performance requirements present")
        
        # High complexity + performance requirements = higher risk
        if (has_performance_requirements or has_performance_keywords) and estimated_complexity > 10:
            risk_level = RiskLevel.HIGH
            factors.append("Complex performance-critical code")
        
        return (risk_level, factors)
    
    def _calculate_risk_score(
        self,
        complexity_risk: RiskLevel,
        integration_risk: RiskLevel,
        security_risk: RiskLevel,
        performance_risk: RiskLevel
    ) -> float:
        """
        Calculate overall risk score.
        
        Args:
            complexity_risk: Complexity risk level
            integration_risk: Integration risk level
            security_risk: Security risk level
            performance_risk: Performance risk level
        
        Returns:
            Risk score (0.0 to 1.0)
        """
        # Convert risk levels to scores
        risk_values = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0
        }
        
        # Weighted average (security and complexity weighted higher)
        score = (
            0.3 * risk_values[complexity_risk] +
            0.2 * risk_values[integration_risk] +
            0.3 * risk_values[security_risk] +
            0.2 * risk_values[performance_risk]
        )
        
        return min(score, 1.0)
    
    def _risk_score_to_level(self, risk_score: float) -> RiskLevel:
        """
        Convert risk score to risk level.
        
        Args:
            risk_score: Numerical risk score
        
        Returns:
            Risk level
        """
        if risk_score < self.LOW_RISK_THRESHOLD:
            return RiskLevel.LOW
        elif risk_score < self.MEDIUM_RISK_THRESHOLD:
            return RiskLevel.MEDIUM
        elif risk_score < self.HIGH_RISK_THRESHOLD:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_mitigation_recommendations(
        self,
        metrics: RiskMetrics
    ) -> List[str]:
        """
        Get risk mitigation recommendations.
        
        Args:
            metrics: Risk metrics
        
        Returns:
            List of recommendations
        """
        if not metrics.is_valid():
            return []
        
        recommendations = []
        
        if metrics.overall_risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Consider additional code review")
            recommendations.append("Increase test coverage")
        
        if metrics.complexity_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Decompose into smaller subtasks")
            recommendations.append("Add comprehensive unit tests")
        
        if metrics.integration_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Add integration tests")
            recommendations.append("Implement retry logic and error handling")
        
        if metrics.security_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Conduct security review")
            recommendations.append("Add security-specific tests")
            recommendations.append("Implement input validation")
        
        if metrics.performance_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            recommendations.append("Add performance benchmarks")
            recommendations.append("Profile and optimize critical paths")
        
        return recommendations
