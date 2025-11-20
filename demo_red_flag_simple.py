"""
Simple demonstration of red flag detection module functionality.
"""

import sys
sys.path.insert(0, '/home/niel/git/multiAgent')

from src.core.red_flag import (
    RedFlagDetector,
    PatternRegistry,
    UncertaintyDetector,
    EscalationManager,
    EscalationPolicy
)


def demo_pattern_detection():
    """Demonstrate pattern detection"""
    print("=" * 80)
    print("PATTERN DETECTION DEMO")
    print("=" * 80)
    print()
    
    registry = PatternRegistry()
    
    test_cases = [
        ("Clean code with no issues", False),
        ("This code has a TODO that needs fixing", True),
        ("I'm not sure if this approach is correct", True),
        ("This is a temporary workaround for the bug", True),
        ("Security risk: potential SQL injection", True),
        ("Well-tested and documented implementation", False),
    ]
    
    for text, should_flag in test_cases:
        matches = registry.check_all_patterns(text)
        flagged = len(matches) > 0
        status = "✅ PASS" if flagged == should_flag else "❌ FAIL"
        
        print(f"{status} | Expected: {'FLAGGED' if should_flag else 'CLEAN'} | Got: {'FLAGGED' if flagged else 'CLEAN'}")
        print(f"  Text: {text}")
        if matches:
            print(f"  Patterns: {list(matches.keys())}")
        print()


def demo_uncertainty_detection():
    """Demonstrate uncertainty detection"""
    print("=" * 80)
    print("UNCERTAINTY DETECTION DEMO")
    print("=" * 80)
    print()
    
    detector = UncertaintyDetector()
    
    test_cases = [
        "This function correctly implements the algorithm.",
        "I think this might work, but I'm not entirely sure.",
        "This could potentially solve the problem in most cases.",
        "Definitely the right approach for this use case.",
    ]
    
    for text in test_cases:
        score = detector.detect(text)
        print(f"Text: {text}")
        print(f"  Uncertainty Score: {score.score:.2f}")
        print(f"  Confidence Level: {score.confidence_level}")
        print(f"  Acceptable: {'✅ YES' if score.is_acceptable() else '❌ NO'}")
        if score.markers_found:
            print(f"  Markers: {score.markers_found[:3]}")
        print()


def demo_escalation():
    """Demonstrate escalation logic"""
    print("=" * 80)
    print("ESCALATION MANAGEMENT DEMO")
    print("=" * 80)
    print()
    
    policy = EscalationPolicy(max_retries=3)
    manager = EscalationManager(policy)
    
    # Simulate escalation scenario
    from src.core.red_flag.detector import RedFlagResult
    
    print("Simulating 5 failed attempts...")
    print()
    
    for attempt in range(5):
        # Create a red flag result
        result = RedFlagResult(
            is_flagged=True,
            reasons=["Pattern violations detected"],
            severity="medium" if attempt < 3 else "high"
        )
        
        decision = manager.handle_red_flag(
            task_id="demo_task",
            red_flag_result=result,
            task_criticality="high"
        )
        
        print(f"Attempt {attempt + 1}:")
        print(f"  Severity: {result.severity}")
        print(f"  Decision: {decision.level.value}")
        print(f"  Action: {decision.action}")
        print(f"  Retry Count: {manager.get_retry_count('demo_task')}")
        print()
        
        if decision.needs_human() or decision.needs_formal_verification():
            print(f"  🚨 Escalated to: {decision.level.value.upper()}")
            break
    
    # Show statistics
    stats = manager.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def main():
    """Run all demonstrations"""
    print("\n")
    print("█" * 80)
    print("RED FLAG DETECTION MODULE - DEMONSTRATION")
    print("█" * 80)
    print("\n")
    
    demo_pattern_detection()
    demo_uncertainty_detection()
    demo_escalation()
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
