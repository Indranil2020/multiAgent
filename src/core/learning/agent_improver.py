"""
Agent performance improvement and specialization system.

This module tracks agent performance metrics and identifies optimal agent
configurations for different task types, enabling continuous improvement
through specialization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
from collections import defaultdict


class PerformanceMetric(Enum):
    """Types of performance metrics tracked."""
    SUCCESS_RATE = "success_rate"
    AVERAGE_QUALITY = "average_quality"
    AVERAGE_EXECUTION_TIME = "average_execution_time"
    CONSENSUS_CONTRIBUTION = "consensus_contribution"
    RED_FLAG_RATE = "red_flag_rate"


@dataclass
class AgentPerformanceMetrics:
    """
    Performance metrics for an agent configuration.
    
    Attributes:
        agent_config_signature: Unique signature of the agent configuration
        task_type: Type of tasks these metrics apply to
        total_executions: Total number of executions
        successful_executions: Number of successful executions
        failed_executions: Number of failed executions
        red_flagged_executions: Number of red-flagged executions
        total_quality_score: Sum of quality scores
        total_execution_time_ms: Sum of execution times
        consensus_wins: Number of times this config won consensus
        last_updated_timestamp: When metrics were last updated
    """
    agent_config_signature: str
    task_type: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    red_flagged_executions: int = 0
    total_quality_score: float = 0.0
    total_execution_time_ms: int = 0
    consensus_wins: int = 0
    last_updated_timestamp: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def get_average_quality(self) -> float:
        """Calculate average quality score."""
        if self.successful_executions == 0:
            return 0.0
        return self.total_quality_score / self.successful_executions
    
    def get_average_execution_time(self) -> float:
        """Calculate average execution time in milliseconds."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_executions
    
    def get_red_flag_rate(self) -> float:
        """Calculate red flag rate."""
        if self.total_executions == 0:
            return 0.0
        return self.red_flagged_executions / self.total_executions
    
    def get_consensus_win_rate(self) -> float:
        """Calculate consensus win rate."""
        if self.successful_executions == 0:
            return 0.0
        return self.consensus_wins / self.successful_executions
    
    def validate_counts(self) -> bool:
        """Validate that counts are consistent."""
        if self.total_executions < 0:
            return False
        
        if self.successful_executions < 0 or self.failed_executions < 0:
            return False
        
        if self.red_flagged_executions < 0 or self.consensus_wins < 0:
            return False
        
        # Total should equal sum of successful, failed, and red-flagged
        expected_total = (
            self.successful_executions + 
            self.failed_executions + 
            self.red_flagged_executions
        )
        
        if self.total_executions != expected_total:
            return False
        
        # Consensus wins cannot exceed successful executions
        if self.consensus_wins > self.successful_executions:
            return False
        
        return True
    
    def is_valid(self) -> bool:
        """Check if metrics are valid."""
        return self.validate_counts()


@dataclass
class SpecializationProfile:
    """
    Defines agent specialization for specific task types.
    
    Attributes:
        profile_id: Unique identifier for this profile
        task_type: Task type this profile specializes in
        recommended_model: Recommended model name
        recommended_temperature: Recommended temperature setting
        recommended_system_prompt: Recommended system prompt variant
        confidence: Confidence in this recommendation (0.0 to 1.0)
        based_on_executions: Number of executions this is based on
        performance_metrics: Associated performance metrics
        created_timestamp: When this profile was created
    """
    profile_id: str
    task_type: str
    recommended_model: str
    recommended_temperature: float
    recommended_system_prompt: str
    confidence: float
    based_on_executions: int
    performance_metrics: AgentPerformanceMetrics
    created_timestamp: float
    
    def validate_temperature(self) -> bool:
        """Validate temperature is in valid range."""
        return 0.0 <= self.recommended_temperature <= 1.0
    
    def validate_confidence(self) -> bool:
        """Validate confidence is in valid range."""
        return 0.0 <= self.confidence <= 1.0
    
    def validate_executions(self) -> bool:
        """Validate execution count is positive."""
        return self.based_on_executions > 0
    
    def is_valid(self) -> bool:
        """Check if profile is valid."""
        return (
            self.validate_temperature() and
            self.validate_confidence() and
            self.validate_executions() and
            self.performance_metrics.is_valid()
        )


@dataclass
class ExecutionRecord:
    """
    Record of a single agent execution.
    
    Attributes:
        task_id: ID of the task executed
        task_type: Type of task
        agent_config: Configuration of the agent
        success: Whether execution was successful
        red_flagged: Whether result was red-flagged
        quality_score: Quality score of the result
        execution_time_ms: Execution time in milliseconds
        won_consensus: Whether this result won consensus
        timestamp: When execution occurred
    """
    task_id: str
    task_type: str
    agent_config: Dict[str, str]
    success: bool
    red_flagged: bool
    quality_score: float
    execution_time_ms: int
    won_consensus: bool
    timestamp: float


class AgentImprover:
    """
    Improves agent performance through specialization and optimization.
    
    This class tracks agent performance across different task types and
    identifies optimal configurations, enabling the system to learn which
    agents work best for which tasks.
    """
    
    def __init__(
        self,
        min_executions_for_profile: int = 10,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize agent improver.
        
        Args:
            min_executions_for_profile: Minimum executions before creating profile
            confidence_threshold: Minimum confidence for recommendations
        """
        self.min_executions_for_profile = min_executions_for_profile
        self.confidence_threshold = confidence_threshold
        self.metrics: Dict[Tuple[str, str], AgentPerformanceMetrics] = {}
        self.specialization_profiles: Dict[str, SpecializationProfile] = {}
        self.execution_records: List[ExecutionRecord] = []
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate configuration parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.min_executions_for_profile < 1:
            return (False, "min_executions_for_profile must be at least 1")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            return (False, "confidence_threshold must be between 0.0 and 1.0")
        
        return (True, "")
    
    def record_execution(self, execution: ExecutionRecord) -> Tuple[bool, str]:
        """
        Record an agent execution for performance tracking.
        
        Args:
            execution: The execution record to add
        
        Returns:
            Tuple of (success, message)
        """
        if not execution.task_id:
            return (False, "task_id cannot be empty")
        
        if not execution.task_type:
            return (False, "task_type cannot be empty")
        
        if not execution.agent_config:
            return (False, "agent_config cannot be empty")
        
        if execution.quality_score < 0.0:
            return (False, "quality_score cannot be negative")
        
        if execution.execution_time_ms < 0:
            return (False, "execution_time_ms cannot be negative")
        
        if execution.timestamp <= 0:
            return (False, "timestamp must be positive")
        
        # Add to records
        self.execution_records.append(execution)
        
        # Update metrics
        config_sig = self._create_config_signature(execution.agent_config)
        key = (config_sig, execution.task_type)
        
        if key not in self.metrics:
            self.metrics[key] = AgentPerformanceMetrics(
                agent_config_signature=config_sig,
                task_type=execution.task_type,
                last_updated_timestamp=execution.timestamp
            )
        
        metrics = self.metrics[key]
        metrics.total_executions += 1
        
        if execution.red_flagged:
            metrics.red_flagged_executions += 1
        elif execution.success:
            metrics.successful_executions += 1
            metrics.total_quality_score += execution.quality_score
            if execution.won_consensus:
                metrics.consensus_wins += 1
        else:
            metrics.failed_executions += 1
        
        metrics.total_execution_time_ms += execution.execution_time_ms
        metrics.last_updated_timestamp = execution.timestamp
        
        return (True, "Execution recorded successfully")
    
    def generate_specialization_profiles(self) -> Tuple[bool, str]:
        """
        Generate specialization profiles based on performance data.
        
        Returns:
            Tuple of (success, message)
        """
        if len(self.execution_records) < self.min_executions_for_profile:
            return (
                False,
                f"Need at least {self.min_executions_for_profile} executions"
            )
        
        # Group metrics by task type
        by_task_type: Dict[str, List[Tuple[str, AgentPerformanceMetrics]]] = defaultdict(list)
        
        for (config_sig, task_type), metrics in self.metrics.items():
            if metrics.total_executions >= self.min_executions_for_profile:
                by_task_type[task_type].append((config_sig, metrics))
        
        # For each task type, find best performing configuration
        profiles_created = 0
        
        for task_type, config_metrics_list in by_task_type.items():
            if not config_metrics_list:
                continue
            
            # Score each configuration
            scored_configs = []
            for config_sig, metrics in config_metrics_list:
                score = self._calculate_performance_score(metrics)
                scored_configs.append((score, config_sig, metrics))
            
            # Sort by score (descending)
            scored_configs.sort(reverse=True, key=lambda x: x[0])
            
            # Take best configuration
            best_score, best_config_sig, best_metrics = scored_configs[0]
            
            # Calculate confidence based on score and sample size
            confidence = self._calculate_profile_confidence(
                best_score,
                best_metrics.total_executions
            )
            
            if confidence >= self.confidence_threshold:
                # Extract configuration details from first execution with this config
                config_details = self._get_config_details(best_config_sig)
                
                if config_details:
                    profile_id = f"profile_{task_type}_{best_config_sig}"
                    
                    profile = SpecializationProfile(
                        profile_id=profile_id,
                        task_type=task_type,
                        recommended_model=config_details.get("model", "unknown"),
                        recommended_temperature=float(config_details.get("temperature", "0.7")),
                        recommended_system_prompt=config_details.get("system_prompt", ""),
                        confidence=confidence,
                        based_on_executions=best_metrics.total_executions,
                        performance_metrics=best_metrics,
                        created_timestamp=best_metrics.last_updated_timestamp
                    )
                    
                    self.specialization_profiles[profile_id] = profile
                    profiles_created += 1
        
        return (True, f"Created {profiles_created} specialization profiles")
    
    def get_recommendation(
        self,
        task_type: str
    ) -> Tuple[Optional[SpecializationProfile], str]:
        """
        Get agent configuration recommendation for a task type.
        
        Args:
            task_type: Type of task to get recommendation for
        
        Returns:
            Tuple of (profile or None, message)
        """
        if not task_type:
            return (None, "task_type cannot be empty")
        
        # Find profiles for this task type
        matching_profiles = [
            profile for profile in self.specialization_profiles.values()
            if profile.task_type == task_type
        ]
        
        if not matching_profiles:
            return (None, f"No specialization profile found for {task_type}")
        
        # Return highest confidence profile
        best_profile = max(matching_profiles, key=lambda p: p.confidence)
        
        return (best_profile, "Recommendation found")
    
    def get_metrics(
        self,
        agent_config: Dict[str, str],
        task_type: str
    ) -> Optional[AgentPerformanceMetrics]:
        """
        Get performance metrics for a specific agent configuration and task type.
        
        Args:
            agent_config: Agent configuration
            task_type: Task type
        
        Returns:
            AgentPerformanceMetrics if found, None otherwise
        """
        config_sig = self._create_config_signature(agent_config)
        key = (config_sig, task_type)
        return self.metrics.get(key)
    
    def get_all_profiles(self) -> List[SpecializationProfile]:
        """
        Get all specialization profiles.
        
        Returns:
            List of all profiles
        """
        return list(self.specialization_profiles.values())
    
    def get_profiles_by_task_type(self, task_type: str) -> List[SpecializationProfile]:
        """
        Get all profiles for a specific task type.
        
        Args:
            task_type: Task type to filter by
        
        Returns:
            List of profiles for the task type
        """
        return [
            profile for profile in self.specialization_profiles.values()
            if profile.task_type == task_type
        ]
    
    def get_top_performers(
        self,
        task_type: str,
        limit: int = 5
    ) -> List[Tuple[str, AgentPerformanceMetrics]]:
        """
        Get top performing agent configurations for a task type.
        
        Args:
            task_type: Task type to analyze
            limit: Maximum number of results to return
        
        Returns:
            List of (config_signature, metrics) tuples
        """
        # Filter metrics for this task type
        task_metrics = [
            (config_sig, metrics)
            for (config_sig, tt), metrics in self.metrics.items()
            if tt == task_type
        ]
        
        # Score and sort
        scored = [
            (self._calculate_performance_score(metrics), config_sig, metrics)
            for config_sig, metrics in task_metrics
        ]
        
        scored.sort(reverse=True, key=lambda x: x[0])
        
        # Return top performers
        return [(config_sig, metrics) for _, config_sig, metrics in scored[:limit]]
    
    def _create_config_signature(self, config: Dict[str, str]) -> str:
        """
        Create a unique signature from agent configuration.
        
        Args:
            config: Agent configuration dictionary
        
        Returns:
            String signature representing the configuration
        """
        sorted_items = sorted(config.items())
        signature_parts = [f"{k}={v}" for k, v in sorted_items]
        return "_".join(signature_parts)
    
    def _calculate_performance_score(self, metrics: AgentPerformanceMetrics) -> float:
        """
        Calculate overall performance score for metrics.
        
        Args:
            metrics: Performance metrics to score
        
        Returns:
            Performance score (higher is better)
        """
        # Weighted combination of different metrics
        success_rate = metrics.get_success_rate()
        avg_quality = metrics.get_average_quality()
        consensus_rate = metrics.get_consensus_win_rate()
        red_flag_rate = metrics.get_red_flag_rate()
        
        # Normalize execution time (lower is better, so invert)
        avg_time = metrics.get_average_execution_time()
        time_score = 1.0 / (1.0 + avg_time / 1000.0)  # Normalize to seconds
        
        # Weighted score
        score = (
            0.30 * success_rate +
            0.25 * avg_quality +
            0.25 * consensus_rate +
            0.10 * time_score +
            0.10 * (1.0 - red_flag_rate)  # Penalize red flags
        )
        
        return score
    
    def _calculate_profile_confidence(
        self,
        performance_score: float,
        num_executions: int
    ) -> float:
        """
        Calculate confidence in a specialization profile.
        
        Args:
            performance_score: Performance score of the configuration
            num_executions: Number of executions the score is based on
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from performance score
        base_confidence = performance_score
        
        # Sample size factor (more executions = higher confidence)
        sample_factor = min(
            num_executions / (self.min_executions_for_profile * 3),
            1.0
        )
        
        # Combined confidence
        confidence = 0.7 * base_confidence + 0.3 * sample_factor
        
        return min(confidence, 1.0)
    
    def _get_config_details(self, config_signature: str) -> Optional[Dict[str, str]]:
        """
        Get configuration details from signature.
        
        Args:
            config_signature: Configuration signature
        
        Returns:
            Configuration dictionary if found, None otherwise
        """
        # Find an execution record with this config signature
        for record in self.execution_records:
            if self._create_config_signature(record.agent_config) == config_signature:
                return record.agent_config
        
        return None
    
    def clear_data(self) -> None:
        """Clear all stored metrics and profiles."""
        self.metrics.clear()
        self.specialization_profiles.clear()
        self.execution_records.clear()
