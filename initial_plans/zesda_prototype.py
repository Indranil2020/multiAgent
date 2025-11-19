"""
ZESDA: Zero-Error Software Development Agentic System
Core Implementation Prototype

This is a working prototype demonstrating the key concepts.
"""

import hashlib
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from collections import Counter
from enum import Enum
import ast
import subprocess
import tempfile
import os

# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class TypedParameter:
    name: str
    type_hint: str
    description: str = ""

@dataclass
class TestCase:
    name: str
    inputs: Dict[str, Any]
    expected_output: Any
    setup_code: str = ""

@dataclass
class Property:
    """For property-based testing"""
    name: str
    predicate: str  # Code that returns bool
    generators: Dict[str, str]  # Parameter name -> generator expression

@dataclass
class TaskSpecification:
    """Formal specification for an atomic code unit"""
    id: str
    name: str
    description: str
    
    # Type information
    inputs: List[TypedParameter]
    outputs: List[TypedParameter]
    
    # Contracts
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    
    # Testing
    test_cases: List[TestCase] = field(default_factory=list)
    properties: List[Property] = field(default_factory=list)
    
    # Constraints
    max_lines: int = 20
    max_complexity: int = 5
    
    # Context
    parent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    context_code: str = ""  # Supporting code/imports

@dataclass
class GeneratedCode:
    """Result from a code generation agent"""
    code: str
    agent_id: str
    temperature: float
    prompt_variant: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VerificationResult:
    passed: bool
    layer: str
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VotingResult:
    winner: GeneratedCode
    vote_count: int
    total_candidates: int
    semantic_groups: int
    confidence: float

# =============================================================================
# Verification Stack
# =============================================================================

class VerificationStack:
    """Multi-layer verification for generated code"""
    
    def verify(self, code: str, spec: TaskSpecification) -> List[VerificationResult]:
        """Run all verification layers"""
        results = []
        
        # Layer 1: Syntax Check
        syntax_result = self._verify_syntax(code)
        results.append(syntax_result)
        if not syntax_result.passed:
            return results
        
        # Layer 2: Type Check (basic)
        type_result = self._verify_types(code, spec)
        results.append(type_result)
        if not type_result.passed:
            return results
        
        # Layer 3: Complexity Check
        complexity_result = self._verify_complexity(code, spec)
        results.append(complexity_result)
        if not complexity_result.passed:
            return results
        
        # Layer 4: Unit Tests
        test_result = self._run_tests(code, spec)
        results.append(test_result)
        if not test_result.passed:
            return results
        
        # Layer 5: Contract Check
        contract_result = self._verify_contracts(code, spec)
        results.append(contract_result)
        
        return results
    
    def _verify_syntax(self, code: str) -> VerificationResult:
        """Check if code parses correctly"""
        try:
            ast.parse(code)
            return VerificationResult(passed=True, layer="syntax")
        except SyntaxError as e:
            return VerificationResult(
                passed=False, 
                layer="syntax",
                message=str(e)
            )
    
    def _verify_types(self, code: str, spec: TaskSpecification) -> VerificationResult:
        """Basic type checking using mypy"""
        # In production, would use mypy programmatically
        # For now, just check that type hints are present
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has return annotation
                if node.returns is None and spec.outputs:
                    return VerificationResult(
                        passed=False,
                        layer="types",
                        message=f"Function {node.name} missing return type annotation"
                    )
        
        return VerificationResult(passed=True, layer="types")
    
    def _verify_complexity(self, code: str, spec: TaskSpecification) -> VerificationResult:
        """Check cyclomatic complexity and line count"""
        lines = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')])
        
        if lines > spec.max_lines:
            return VerificationResult(
                passed=False,
                layer="complexity",
                message=f"Code has {lines} lines, max is {spec.max_lines}"
            )
        
        # Simple complexity estimate based on control flow
        tree = ast.parse(code)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        if complexity > spec.max_complexity:
            return VerificationResult(
                passed=False,
                layer="complexity",
                message=f"Cyclomatic complexity {complexity} exceeds max {spec.max_complexity}"
            )
        
        return VerificationResult(
            passed=True, 
            layer="complexity",
            details={"lines": lines, "complexity": complexity}
        )
    
    def _run_tests(self, code: str, spec: TaskSpecification) -> VerificationResult:
        """Execute test cases"""
        if not spec.test_cases:
            return VerificationResult(passed=True, layer="tests", message="No tests defined")
        
        # Create test execution environment
        test_code = f"""
{spec.context_code}

{code}

# Test execution
import json
results = []
"""
        
        for test in spec.test_cases:
            test_code += f"""
try:
    {test.setup_code}
    result = {spec.name}(**{json.dumps(test.inputs)})
    expected = {json.dumps(test.expected_output)}
    passed = result == expected
    results.append({{"name": "{test.name}", "passed": passed, "result": str(result), "expected": str(expected)}})
except Exception as e:
    results.append({{"name": "{test.name}", "passed": False, "error": str(e)}})
"""
        
        test_code += "\nprint(json.dumps(results))"
        
        # Execute in subprocess for isolation
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            os.unlink(temp_path)
            
            if result.returncode != 0:
                return VerificationResult(
                    passed=False,
                    layer="tests",
                    message=f"Test execution failed: {result.stderr}"
                )
            
            test_results = json.loads(result.stdout)
            failed = [t for t in test_results if not t['passed']]
            
            if failed:
                return VerificationResult(
                    passed=False,
                    layer="tests",
                    message=f"Failed tests: {[t['name'] for t in failed]}",
                    details={"results": test_results}
                )
            
            return VerificationResult(
                passed=True,
                layer="tests",
                details={"results": test_results}
            )
            
        except Exception as e:
            return VerificationResult(
                passed=False,
                layer="tests",
                message=f"Test execution error: {str(e)}"
            )
    
    def _verify_contracts(self, code: str, spec: TaskSpecification) -> VerificationResult:
        """Verify pre/post conditions"""
        # In production, would generate contract-checking wrapper
        # For now, just ensure contracts are checkable
        
        if not spec.preconditions and not spec.postconditions:
            return VerificationResult(passed=True, layer="contracts")
        
        # Check that contract expressions are valid Python
        for pre in spec.preconditions:
            try:
                ast.parse(pre)
            except SyntaxError:
                return VerificationResult(
                    passed=False,
                    layer="contracts",
                    message=f"Invalid precondition: {pre}"
                )
        
        for post in spec.postconditions:
            try:
                ast.parse(post)
            except SyntaxError:
                return VerificationResult(
                    passed=False,
                    layer="contracts",
                    message=f"Invalid postcondition: {post}"
                )
        
        return VerificationResult(passed=True, layer="contracts")


# =============================================================================
# Red-Flag Detector
# =============================================================================

class RedFlagDetector:
    """Detect problematic outputs that should be discarded"""
    
    UNCERTAINTY_MARKERS = [
        "# TODO",
        "# FIXME",
        "# XXX",
        "# HACK",
        "pass  # not implemented",
        "raise NotImplementedError",
        "...",  # Ellipsis as placeholder
    ]
    
    def check(self, code: str, spec: TaskSpecification) -> tuple[bool, List[str]]:
        """
        Check for red flags.
        Returns (has_flags, list_of_flags)
        """
        flags = []
        
        # Check for uncertainty markers
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in code:
                flags.append(f"Contains uncertainty marker: {marker}")
        
        # Check response length
        lines = code.count('\n') + 1
        if lines > spec.max_lines * 2:
            flags.append(f"Response too long: {lines} lines")
        
        if lines < 2:
            flags.append("Response too short")
        
        # Check for common error patterns
        if "import *" in code:
            flags.append("Uses wildcard import")
        
        if code.count("try:") > code.count("except"):
            flags.append("Unbalanced try/except")
        
        # Check for hardcoded test values
        if "test" in code.lower() and "def " not in code.lower():
            flags.append("Possible hardcoded test values")
        
        return len(flags) > 0, flags


# =============================================================================
# Semantic Equivalence Checker
# =============================================================================

class SemanticEquivalenceChecker:
    """Group implementations by their behavior"""
    
    def get_semantic_signature(self, code: str, spec: TaskSpecification) -> str:
        """
        Generate a signature based on the code's behavior.
        Two implementations with the same signature behave identically.
        """
        if not spec.test_cases:
            # Fall back to syntax-normalized hash
            return self._normalize_and_hash(code)
        
        # Run tests and use outputs as signature
        outputs = []
        
        test_code = f"""
{spec.context_code}
{code}

import json
outputs = []
"""
        
        for test in spec.test_cases:
            test_code += f"""
try:
    result = {spec.name}(**{json.dumps(test.inputs)})
    outputs.append(("success", str(result)))
except Exception as e:
    outputs.append(("error", type(e).__name__))
"""
        
        test_code += "print(json.dumps(outputs))"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                temp_path = f.name
            
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                outputs = json.loads(result.stdout)
                return hashlib.md5(json.dumps(outputs).encode()).hexdigest()
        except:
            pass
        
        return self._normalize_and_hash(code)
    
    def _normalize_and_hash(self, code: str) -> str:
        """Normalize code and hash for syntactic comparison"""
        try:
            tree = ast.parse(code)
            # Remove docstrings and comments for comparison
            normalized = ast.dump(tree)
            return hashlib.md5(normalized.encode()).hexdigest()
        except:
            return hashlib.md5(code.encode()).hexdigest()


# =============================================================================
# Voting Engine
# =============================================================================

class VotingEngine:
    """First-to-ahead-by-k voting system"""
    
    def __init__(self, k: int = 3):
        self.k = k
        self.equivalence_checker = SemanticEquivalenceChecker()
    
    def vote(self, 
             candidates: List[GeneratedCode], 
             spec: TaskSpecification) -> Optional[VotingResult]:
        """
        Vote on candidates using first-to-ahead-by-k rule.
        Returns the winner or None if no consensus.
        """
        if not candidates:
            return None
        
        # Group by semantic equivalence
        groups: Dict[str, List[GeneratedCode]] = {}
        
        for candidate in candidates:
            sig = self.equivalence_checker.get_semantic_signature(
                candidate.code, spec
            )
            if sig not in groups:
                groups[sig] = []
            groups[sig].append(candidate)
        
        # Count votes per group
        vote_counts = {sig: len(impls) for sig, impls in groups.items()}
        
        # Check for winner (ahead by k)
        sorted_groups = sorted(vote_counts.items(), key=lambda x: -x[1])
        
        if len(sorted_groups) == 1:
            # Only one interpretation
            winner_sig = sorted_groups[0][0]
            winner_count = sorted_groups[0][1]
        elif sorted_groups[0][1] >= sorted_groups[1][1] + self.k:
            # Clear winner
            winner_sig = sorted_groups[0][0]
            winner_count = sorted_groups[0][1]
        else:
            # No clear winner yet
            return None
        
        # Select best implementation from winning group
        winner_group = groups[winner_sig]
        best = self._select_best_quality(winner_group)
        
        confidence = winner_count / len(candidates)
        
        return VotingResult(
            winner=best,
            vote_count=winner_count,
            total_candidates=len(candidates),
            semantic_groups=len(groups),
            confidence=confidence
        )
    
    def _select_best_quality(self, implementations: List[GeneratedCode]) -> GeneratedCode:
        """Select the highest quality implementation from equivalent options"""
        
        def quality_score(impl: GeneratedCode) -> float:
            code = impl.code
            score = 0
            
            # Prefer shorter code
            score -= len(code) * 0.001
            
            # Prefer code with docstrings
            if '"""' in code or "'''" in code:
                score += 10
            
            # Prefer code with type hints
            if '->' in code:
                score += 5
            
            # Penalize excessive nesting
            max_indent = max(len(line) - len(line.lstrip()) 
                           for line in code.split('\n') if line.strip())
            score -= max_indent * 0.5
            
            return score
        
        return max(implementations, key=quality_score)


# =============================================================================
# Code Generation Agent (Mock)
# =============================================================================

class CodeGenerationAgent:
    """
    Agent that generates code for a task specification.
    In production, this would call an LLM API.
    """
    
    def __init__(self, agent_id: str, model: str = "gpt-4"):
        self.agent_id = agent_id
        self.model = model
        self.prompt_variants = self._load_prompt_variants()
    
    def _load_prompt_variants(self) -> List[str]:
        """Different ways to phrase the code generation request"""
        return [
            "Implement the following function according to the specification:",
            "Write Python code for this function. Follow the specification exactly:",
            "Create a function implementation based on these requirements:",
            "Code the following function. Ensure all contracts are satisfied:",
        ]
    
    def generate(self, 
                 spec: TaskSpecification, 
                 temperature: float = 0.1,
                 prompt_variant: int = 0) -> GeneratedCode:
        """Generate code for the specification"""
        
        prompt = self._build_prompt(spec, prompt_variant)
        
        # In production, call LLM API here
        # For this prototype, return mock implementation
        code = self._mock_generate(spec)
        
        return GeneratedCode(
            code=code,
            agent_id=self.agent_id,
            temperature=temperature,
            prompt_variant=prompt_variant
        )
    
    def _build_prompt(self, spec: TaskSpecification, variant: int) -> str:
        """Build the prompt for code generation"""
        
        prompt = f"{self.prompt_variants[variant % len(self.prompt_variants)]}\n\n"
        
        prompt += f"Function Name: {spec.name}\n"
        prompt += f"Description: {spec.description}\n\n"
        
        prompt += "Parameters:\n"
        for param in spec.inputs:
            prompt += f"  - {param.name}: {param.type_hint} - {param.description}\n"
        
        prompt += "\nReturns:\n"
        for out in spec.outputs:
            prompt += f"  - {out.type_hint} - {out.description}\n"
        
        if spec.preconditions:
            prompt += "\nPreconditions:\n"
            for pre in spec.preconditions:
                prompt += f"  - {pre}\n"
        
        if spec.postconditions:
            prompt += "\nPostconditions:\n"
            for post in spec.postconditions:
                prompt += f"  - {post}\n"
        
        if spec.test_cases:
            prompt += "\nTest Cases:\n"
            for test in spec.test_cases[:3]:  # Limit examples
                prompt += f"  - Input: {test.inputs} -> Expected: {test.expected_output}\n"
        
        prompt += f"\nConstraints:\n"
        prompt += f"  - Maximum {spec.max_lines} lines\n"
        prompt += f"  - Maximum cyclomatic complexity: {spec.max_complexity}\n"
        
        return prompt
    
    def _mock_generate(self, spec: TaskSpecification) -> str:
        """Mock implementation for testing"""
        # This would be replaced by actual LLM call
        
        # Generate type hints
        params = ", ".join(
            f"{p.name}: {p.type_hint}" for p in spec.inputs
        )
        returns = spec.outputs[0].type_hint if spec.outputs else "None"
        
        return f'''def {spec.name}({params}) -> {returns}:
    """
    {spec.description}
    """
    # Implementation would be generated by LLM
    pass
'''


# =============================================================================
# Main Orchestrator
# =============================================================================

class ZESDAOrchestrator:
    """
    Main orchestrator for the Zero-Error Software Development system.
    Coordinates agents, voting, and verification.
    """
    
    def __init__(self, 
                 n_agents: int = 5,
                 k: int = 3,
                 max_attempts: int = 50):
        self.n_agents = n_agents
        self.k = k
        self.max_attempts = max_attempts
        
        # Initialize components
        self.agents = [
            CodeGenerationAgent(f"agent_{i}") 
            for i in range(n_agents)
        ]
        self.verification_stack = VerificationStack()
        self.red_flag_detector = RedFlagDetector()
        self.voting_engine = VotingEngine(k=k)
    
    async def execute_task(self, spec: TaskSpecification) -> Optional[GeneratedCode]:
        """
        Execute an atomic task with voting and verification.
        """
        
        candidates = []
        attempts = 0
        
        while attempts < self.max_attempts:
            # Generate from all agents with varying parameters
            for i, agent in enumerate(self.agents):
                # Vary temperature and prompt
                temperature = 0.1 + (attempts * 0.05) % 0.5
                prompt_variant = (attempts + i) % 4
                
                generated = agent.generate(
                    spec, 
                    temperature=temperature,
                    prompt_variant=prompt_variant
                )
                
                # Red-flag check
                has_flags, flags = self.red_flag_detector.check(
                    generated.code, spec
                )
                if has_flags:
                    print(f"  Red-flagged: {flags}")
                    continue
                
                # Verification
                results = self.verification_stack.verify(generated.code, spec)
                
                if all(r.passed for r in results):
                    candidates.append(generated)
                    print(f"  Verified candidate from {agent.agent_id}")
                else:
                    failed = [r for r in results if not r.passed]
                    print(f"  Verification failed: {[f.layer for f in failed]}")
            
            # Try to vote
            vote_result = self.voting_engine.vote(candidates, spec)
            
            if vote_result:
                print(f"\nVoting complete!")
                print(f"  Winner with {vote_result.vote_count}/{vote_result.total_candidates} votes")
                print(f"  {vote_result.semantic_groups} distinct implementations")
                print(f"  Confidence: {vote_result.confidence:.2%}")
                return vote_result.winner
            
            attempts += 1
            print(f"Attempt {attempts}: {len(candidates)} candidates, no winner yet")
        
        print(f"Failed to reach consensus after {self.max_attempts} attempts")
        return None
    
    def execute_task_sync(self, spec: TaskSpecification) -> Optional[GeneratedCode]:
        """Synchronous wrapper"""
        return asyncio.run(self.execute_task(spec))


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Demonstrate the system with a simple example"""
    
    # Define a task specification
    spec = TaskSpecification(
        id="math_001",
        name="safe_divide",
        description="Safely divide two numbers, returning None if divisor is zero",
        
        inputs=[
            TypedParameter("a", "float", "Dividend"),
            TypedParameter("b", "float", "Divisor"),
        ],
        outputs=[
            TypedParameter("result", "Optional[float]", "Quotient or None"),
        ],
        
        preconditions=[
            "isinstance(a, (int, float))",
            "isinstance(b, (int, float))",
        ],
        postconditions=[
            "result is None if b == 0 else result == a / b",
        ],
        
        test_cases=[
            TestCase("normal_division", {"a": 10.0, "b": 2.0}, 5.0),
            TestCase("zero_divisor", {"a": 10.0, "b": 0.0}, None),
            TestCase("negative", {"a": -6.0, "b": 2.0}, -3.0),
            TestCase("decimal", {"a": 1.0, "b": 3.0}, 1/3),
        ],
        
        max_lines=10,
        max_complexity=3,
        
        context_code="from typing import Optional"
    )
    
    # Create orchestrator and execute
    orchestrator = ZESDAOrchestrator(n_agents=3, k=2, max_attempts=10)
    
    print("=" * 60)
    print("ZESDA: Executing atomic task")
    print("=" * 60)
    print(f"Task: {spec.name}")
    print(f"Description: {spec.description}")
    print("=" * 60)
    
    result = orchestrator.execute_task_sync(spec)
    
    if result:
        print("\n" + "=" * 60)
        print("SUCCESS! Generated code:")
        print("=" * 60)
        print(result.code)
    else:
        print("\nFailed to generate verified code")

if __name__ == "__main__":
    example_usage()
