"""
Escalation Module

Handles escalation logic for red-flagged responses.
Determines when and how to escalate issues for human review or formal verification.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .detector import RedFlagResult


class EscalationLevel(Enum):
    """Levels of escalation"""
    NONE = "none"
    RETRY = "retry"
    HUMAN_REVIEW = "human_review"
    FORMAL_VERIFICATION = "formal_verification"
    EXPERT_CONSULTATION = "expert_consultation"


class EscalationReason(Enum):
    """Reasons for escalation"""
    HIGH_UNCERTAINTY = "high_uncertainty"
    REPEATED_FAILURES = "repeated_failures"
    SECURITY_CONCERN = "security_concern"
    COMPLEXITY_EXCEEDED = "complexity_exceeded"
    NO_CONSENSUS = "no_consensus"
    CRITICAL_COMPONENT = "critical_component"
    PATTERN_VIOLATIONS = "pattern_violations"


@dataclass
class EscalationDecision:
    """Decision about how to handle a red-flagged response"""
    level: EscalationLevel
    reasons: List[EscalationReason] = field(default_factory=list)
    action: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 0-10, higher is more urgent
    
    def should_retry(self) -> bool:
        """Check if should retry with different agent"""
        return self.level == EscalationLevel.RETRY
    
    def needs_human(self) -> bool:
        """Check if needs human intervention"""
        return self.level in [
            EscalationLevel.HUMAN_REVIEW,
            EscalationLevel.EXPERT_CONSULTATION
        ]
    
    def needs_formal_verification(self) -> bool:
        """Check if needs formal verification"""
        return self.level == EscalationLevel.FORMAL_VERIFICATION
    
    def __str__(self) -> str:
        return f"Escalation({self.level.value}, priority={self.priority}): {self.action}"


class EscalationPolicy:
    """
    Policy for determining escalation decisions.
    
    Defines rules for when and how to escalate red-flagged responses.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        uncertainty_threshold_critical: float = 0.8,
        enable_formal_verification: bool = True,
        enable_human_review: bool = True
    ):
        """
        Initialize escalation policy.
        
        Args:
            max_retries: Maximum number of retries before escalation
            uncertainty_threshold_critical: Uncertainty score requiring immediate escalation
            enable_formal_verification: Whether formal verification is available
            enable_human_review: Whether human review is available
        """
        self.max_retries = max_retries
        self.uncertainty_threshold_critical = uncertainty_threshold_critical
        self.enable_formal_verification = enable_formal_verification
        self.enable_human_review = enable_human_review
    
    def decide(
        self,
        red_flag_result: RedFlagResult,
        attempt_count: int = 0,
        task_criticality: str = "medium"
    ) -> EscalationDecision:
        """
        Decide how to handle a red-flagged response.
        
        Args:
            red_flag_result: The red flag detection result
            attempt_count: Number of attempts so far
            task_criticality: Criticality of the task (low, medium, high, critical)
            
        Returns:
            EscalationDecision with recommended action
        """
        reasons = []
        priority = 0
        
        # Not flagged - no escalation needed
        if not red_flag_result.is_flagged:
            return EscalationDecision(
                level=EscalationLevel.NONE,
                action="No escalation needed"
            )
        
        # Critical severity - immediate escalation
        if red_flag_result.severity == "critical":
            if self.enable_formal_verification:
                return EscalationDecision(
                    level=EscalationLevel.FORMAL_VERIFICATION,
                    reasons=[EscalationReason.PATTERN_VIOLATIONS],
                    action="Critical issues detected - escalate to formal verification",
                    priority=10
                )
            elif self.enable_human_review:
                return EscalationDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    reasons=[EscalationReason.PATTERN_VIOLATIONS],
                    action="Critical issues detected - escalate to human review",
                    priority=10
                )
        
        # Very high uncertainty
        if red_flag_result.uncertainty_score >= self.uncertainty_threshold_critical:
            reasons.append(EscalationReason.HIGH_UNCERTAINTY)
            priority = max(priority, 8)
        
        # Check for security concerns
        if "security" in str(red_flag_result.reasons).lower():
            reasons.append(EscalationReason.SECURITY_CONCERN)
            priority = max(priority, 9)
            
            if self.enable_human_review:
                return EscalationDecision(
                    level=EscalationLevel.EXPERT_CONSULTATION,
                    reasons=reasons,
                    action="Security concerns detected - escalate to security expert",
                    priority=priority
                )
        
        # Check retry count
        if attempt_count < self.max_retries:
            return EscalationDecision(
                level=EscalationLevel.RETRY,
                reasons=[EscalationReason.PATTERN_VIOLATIONS],
                action=f"Retry with different agent (attempt {attempt_count + 1}/{self.max_retries})",
                metadata={"attempt_count": attempt_count},
                priority=3
            )
        
        # Exceeded retries
        if attempt_count >= self.max_retries:
            reasons.append(EscalationReason.REPEATED_FAILURES)
            priority = max(priority, 7)
        
        # High severity or critical task
        if red_flag_result.severity == "high" or task_criticality in ["high", "critical"]:
            if self.enable_formal_verification and task_criticality == "critical":
                return EscalationDecision(
                    level=EscalationLevel.FORMAL_VERIFICATION,
                    reasons=reasons if reasons else [EscalationReason.CRITICAL_COMPONENT],
                    action="Critical task with issues - escalate to formal verification",
                    priority=priority if priority > 0 else 8
                )
            elif self.enable_human_review:
                return EscalationDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    reasons=reasons if reasons else [EscalationReason.REPEATED_FAILURES],
                    action="High severity issues - escalate to human review",
                    priority=priority if priority > 0 else 6
                )
        
        # Medium severity - retry or human review
        if red_flag_result.severity == "medium":
            if self.enable_human_review and attempt_count >= self.max_retries:
                return EscalationDecision(
                    level=EscalationLevel.HUMAN_REVIEW,
                    reasons=[EscalationReason.REPEATED_FAILURES],
                    action="Multiple retries failed - escalate to human review",
                    priority=5
                )
        
        # Default: retry
        return EscalationDecision(
            level=EscalationLevel.RETRY,
            reasons=[EscalationReason.PATTERN_VIOLATIONS],
            action="Retry with different agent configuration",
            priority=2
        )


class EscalationManager:
    """
    Manages escalation workflow for red-flagged responses.
    
    Tracks escalation history and coordinates escalation actions.
    """
    
    def __init__(self, policy: Optional[EscalationPolicy] = None):
        """
        Initialize escalation manager.
        
        Args:
            policy: Escalation policy to use (creates default if None)
        """
        self.policy = policy if policy else EscalationPolicy()
        self.escalation_history: Dict[str, List[EscalationDecision]] = {}
        self.retry_counts: Dict[str, int] = {}
    
    def handle_red_flag(
        self,
        task_id: str,
        red_flag_result: RedFlagResult,
        task_criticality: str = "medium"
    ) -> EscalationDecision:
        """
        Handle a red-flagged response.
        
        Args:
            task_id: ID of the task
            red_flag_result: Red flag detection result
            task_criticality: Criticality level of the task
            
        Returns:
            EscalationDecision with recommended action
        """
        # Get current retry count
        attempt_count = self.retry_counts.get(task_id, 0)
        
        # Make escalation decision
        decision = self.policy.decide(
            red_flag_result,
            attempt_count,
            task_criticality
        )
        
        # Update history
        if task_id not in self.escalation_history:
            self.escalation_history[task_id] = []
        self.escalation_history[task_id].append(decision)
        
        # Update retry count
        if decision.should_retry():
            self.retry_counts[task_id] = attempt_count + 1
        
        return decision
    
    def get_escalation_history(self, task_id: str) -> List[EscalationDecision]:
        """Get escalation history for a task"""
        return self.escalation_history.get(task_id, [])
    
    def get_retry_count(self, task_id: str) -> int:
        """Get number of retries for a task"""
        return self.retry_counts.get(task_id, 0)
    
    def reset_task(self, task_id: str) -> None:
        """Reset escalation state for a task"""
        if task_id in self.escalation_history:
            del self.escalation_history[task_id]
        if task_id in self.retry_counts:
            del self.retry_counts[task_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        total_tasks = len(self.escalation_history)
        
        if total_tasks == 0:
            return {
                "total_tasks": 0,
                "escalation_levels": {},
                "average_retries": 0,
                "max_retries": 0
            }
        
        # Count escalation levels
        level_counts = {}
        for decisions in self.escalation_history.values():
            for decision in decisions:
                level = decision.level.value
                level_counts[level] = level_counts.get(level, 0) + 1
        
        # Calculate retry statistics
        retry_values = list(self.retry_counts.values())
        avg_retries = sum(retry_values) / len(retry_values) if retry_values else 0
        max_retries = max(retry_values) if retry_values else 0
        
        return {
            "total_tasks": total_tasks,
            "escalation_levels": level_counts,
            "average_retries": avg_retries,
            "max_retries": max_retries,
            "tasks_needing_human": sum(
                1 for decisions in self.escalation_history.values()
                if any(d.needs_human() for d in decisions)
            ),
            "tasks_needing_formal_verification": sum(
                1 for decisions in self.escalation_history.values()
                if any(d.needs_formal_verification() for d in decisions)
            )
        }


class EscalationRouter:
    """Routes escalated tasks to appropriate handlers"""
    
    def __init__(self):
        self.human_review_queue: List[Dict[str, Any]] = []
        self.formal_verification_queue: List[Dict[str, Any]] = []
        self.expert_consultation_queue: List[Dict[str, Any]] = []
    
    def route(
        self,
        task_id: str,
        decision: EscalationDecision,
        context: Dict[str, Any]
    ) -> str:
        """
        Route an escalated task to the appropriate queue.
        
        Args:
            task_id: ID of the task
            decision: Escalation decision
            context: Additional context
            
        Returns:
            Queue name where task was routed
        """
        escalation_item = {
            "task_id": task_id,
            "decision": decision,
            "context": context,
            "priority": decision.priority
        }
        
        if decision.level == EscalationLevel.HUMAN_REVIEW:
            self.human_review_queue.append(escalation_item)
            self._sort_queue(self.human_review_queue)
            return "human_review"
        
        elif decision.level == EscalationLevel.FORMAL_VERIFICATION:
            self.formal_verification_queue.append(escalation_item)
            self._sort_queue(self.formal_verification_queue)
            return "formal_verification"
        
        elif decision.level == EscalationLevel.EXPERT_CONSULTATION:
            self.expert_consultation_queue.append(escalation_item)
            self._sort_queue(self.expert_consultation_queue)
            return "expert_consultation"
        
        return "none"
    
    def _sort_queue(self, queue: List[Dict[str, Any]]) -> None:
        """Sort queue by priority (highest first)"""
        queue.sort(key=lambda x: x["priority"], reverse=True)
    
    def get_next_item(self, queue_name: str) -> Optional[Dict[str, Any]]:
        """Get next item from specified queue"""
        queue_map = {
            "human_review": self.human_review_queue,
            "formal_verification": self.formal_verification_queue,
            "expert_consultation": self.expert_consultation_queue
        }
        
        queue = queue_map.get(queue_name)
        if queue and len(queue) > 0:
            return queue.pop(0)
        return None
    
    def get_queue_size(self, queue_name: str) -> int:
        """Get size of specified queue"""
        queue_map = {
            "human_review": self.human_review_queue,
            "formal_verification": self.formal_verification_queue,
            "expert_consultation": self.expert_consultation_queue
        }
        
        queue = queue_map.get(queue_name, [])
        return len(queue)
    
    def get_all_queue_sizes(self) -> Dict[str, int]:
        """Get sizes of all queues"""
        return {
            "human_review": len(self.human_review_queue),
            "formal_verification": len(self.formal_verification_queue),
            "expert_consultation": len(self.expert_consultation_queue)
        }
