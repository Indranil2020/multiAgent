"""
Red Flag Detection Module - Demonstration Script

This script demonstrates the complete functionality of the red flag detection module.
"""

import sys
sys.path.insert(0, '/home/niel/git/multiAgent')

from src.core.task_spec.types import (
    TaskSpecification,
    TaskType,
    AgentResponse,
    Priority,
    TypedParameter,
    Predicate,
    TestCase
)
from src.core.red_flag import (
    RedFlagDetector,
    EscalationManager,
    EscalationPolicy,
    UncertaintyDetector,
    PatternRegistry
)


def demo_basic_detection():
    """Demonstrate basic red flag detection"""
    print("=" * 80)
    print("DEMO 1: Basic Red Flag Detection")
    print("=" * 80)
    print()
    
    # Create a task specification
    task_spec = TaskSpecification(
        id="task_001",
        name="Implement user authentication",
        description="Create a secure user authentication function",
        task_type=TaskType.CODE_GENERATION,
        max_lines=20,
        max_complexity=10
    )
    
    # Create detector
    detector = RedFlagDetector()
    
    # Test Case 1: Good response
    print("Test 1: Clean response (should pass)")
    print("-" * 80)
    good_response = AgentResponse(
        task_id="task_001",
        agent_id="agent_001",
        content="""
def authenticate_user(username: str, password: str) -> bool:
    \"\"\"Authenticate user with username and password\"\"\"
    hashed_password = hash_password(password)
    stored_hash = get_stored_password_hash(username)
    return hashed_password == stored_hash
"""
    )
    
    result = detector.check(good_response, task_spec)
    print(f"Result: {result}")
    print(f"Is Flagged: {result.is_flagged}")
    print()
    
    # Test Case 2: Uncertain response
    print("Test 2: Uncertain response (should be flagged)")
    print("-" * 80)
    uncertain_response = AgentResponse(
        task_id="task_001",
        agent_id="agent_002",
        content="""
def authenticate_user(username: str, password: str) -> bool:
    # I'm not sure if this is the best approach
    # Maybe we should use bcrypt? I think this might work
    return username == "admin" and password == "password"  # TODO: fix this
"""
    )
    
    result = detector.check(uncertain_response, task_spec)
    print(f"Result: {result}")
    print(f"Is Flagged: {result.is_flagged}")
    print(f"Reasons: {result.reasons}")
    print(f"Uncertainty Score: {result.uncertainty_score:.2f}")
    print()
    
    # Test Case 3: Format errors
    print("Test 3: Format errors (should be flagged)")
    print("-" * 80)
    format_error_response = AgentResponse(
        task_id="task_001",
        agent_id="agent_003",
        content="""
def authenticate_user(username: str, password: str) -> bool:
    if username == "admin":
        return True
    # Missing closing brace
    return False
}
"""
    )
    
    result = detector.check(format_error_response, task_spec)
    print(f"Result: {result}")
    print(f"Is Flagged: {result.is_flagged}")
    print(f"Reasons: {result.reasons}")
    print()


def demo_uncertainty_detection():
    """Demonstrate uncertainty detection"""
    print("=" * 80)
    print("DEMO 2: Uncertainty Detection")
    print("=" * 80)
    print()
    
    detector = UncertaintyDetector()
    
    test_texts = [
        ("Confident: This function correctly implements authentication.", 0.0),
        ("Uncertain: I think this might work, but I'm not sure.", 0.8),
        ("Hedging: This could potentially work in most cases.", 0.5),
        ("Mixed: This is correct. However, maybe we should consider alternatives.", 0.3)
    ]
    
    for text, expected_range in test_texts:
        score = detector.detect(text)
        print(f"Text: {text}")
        print(f"  Score: {score.score:.2f}")
        print(f"  Confidence Level: {score.confidence_level}")
        print(f"  Markers Found: {score.markers_found}")
        print()


def demo_pattern_detection():
    """Demonstrate pattern detection"""
    print("=" * 80)
    print("DEMO 3: Pattern Detection")
    print("=" * 80)
    print()
    
    registry = PatternRegistry()
    
    test_texts = [
        "This code has a TODO that needs to be fixed",
        "This is a temporary workaround for the bug",
        "Security risk: SQL injection vulnerability present",
        "This implementation is incomplete and needs work"
    ]
    
    for text in test_texts:
        matches = registry.check_all_patterns(text)
        print(f"Text: {text}")
        print(f"  Patterns Matched: {list(matches.keys())}")
        for pattern_name, pattern_matches in matches.items():
            print(f"    - {pattern_name}: {pattern_matches}")
        print()


def demo_escalation():
    """Demonstrate escalation management"""
    print("=" * 80)
    print("DEMO 4: Escalation Management")
    print("=" * 80)
    print()
    
    # Create components
    detector = RedFlagDetector()
    policy = EscalationPolicy(max_retries=3)
    manager = EscalationManager(policy)
    
    task_spec = TaskSpecification(
        id="task_002",
        name="Critical security function",
        description="Implement encryption",
        task_type=TaskType.CODE_GENERATION
    )
    
    # Simulate multiple failed attempts
    for attempt in range(5):
        print(f"Attempt {attempt + 1}:")
        print("-" * 80)
        
        # Create a problematic response
        response = AgentResponse(
            task_id="task_002",
            agent_id=f"agent_{attempt}",
            content=f"# TODO: implement encryption (attempt {attempt + 1})"
        )
        
        # Check for red flags
        red_flag_result = detector.check(response, task_spec)
        
        # Handle escalation
        decision = manager.handle_red_flag(
            task_id="task_002",
            red_flag_result=red_flag_result,
            task_criticality="critical"
        )
        
        print(f"  Red Flag: {red_flag_result.is_flagged}")
        print(f"  Escalation: {decision}")
        print(f"  Retry Count: {manager.get_retry_count('task_002')}")
        print()
        
        if decision.needs_human() or decision.needs_formal_verification():
            print(f"  ⚠️  Escalated to: {decision.level.value}")
            break
    
    # Show statistics
    print("Escalation Statistics:")
    print("-" * 80)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def demo_batch_processing():
    """Demonstrate batch processing"""
    print("=" * 80)
    print("DEMO 5: Batch Processing")
    print("=" * 80)
    print()
    
    detector = RedFlagDetector()
    
    task_spec = TaskSpecification(
        id="task_003",
        name="Batch task",
        description="Process multiple responses",
        task_type=TaskType.CODE_GENERATION
    )
    
    # Create multiple responses
    responses = [
        AgentResponse("task_003", "agent_1", "def good_function(): return True"),
        AgentResponse("task_003", "agent_2", "# TODO: implement this"),
        AgentResponse("task_003", "agent_3", "def another_good(): pass"),
        AgentResponse("task_003", "agent_4", "I'm not sure how to do this"),
        AgentResponse("task_003", "agent_5", "def clean_code(): return 42"),
    ]
    
    # Batch check
    results = detector.batch_check(responses, task_spec)
    
    # Show results
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {'FLAGGED' if result.is_flagged else 'PASSED'}")
        if result.is_flagged:
            print(f"  Reasons: {result.reasons}")
    print()
    
    # Show statistics
    stats = detector.get_statistics(results)
    print("Batch Statistics:")
    print("-" * 80)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 80)
    print("RED FLAG DETECTION MODULE - COMPREHENSIVE DEMONSTRATION")
    print("█" * 80)
    print("\n")
    
    demo_basic_detection()
    demo_uncertainty_detection()
    demo_pattern_detection()
    demo_escalation()
    demo_batch_processing()
    
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("✅ All red flag detection features demonstrated successfully!")
    print()


if __name__ == '__main__':
    main()
