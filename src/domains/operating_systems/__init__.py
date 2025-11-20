"""
Operating systems domain knowledge and specifications.

This module provides comprehensive OS development patterns including
kernel architecture, process management, memory management, and
system-level programming specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class OSArchitecture(Enum):
    """Operating system architectures."""
    MONOLITHIC = "monolithic"
    MICROKERNEL = "microkernel"
    HYBRID = "hybrid"
    EXOKERNEL = "exokernel"
    UNIKERNEL = "unikernel"


class OSType(Enum):
    """Operating system types."""
    UNIX_LIKE = "unix_like"
    WINDOWS_LIKE = "windows_like"
    RTOS = "rtos"
    EMBEDDED = "embedded"


class SchedulingAlgorithm(Enum):
    """Process scheduling algorithms."""
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"
    CFS = "completely_fair_scheduler"
    MULTILEVEL_QUEUE = "multilevel_queue"
    LOTTERY = "lottery"


@dataclass
class OSSpec:
    """
    Operating system specification.
    
    Attributes:
        name: OS name
        architecture: OS architecture
        os_type: OS type
        target_platform: Target hardware platform
        features: OS features
        constraints: Resource constraints
    """
    name: str
    architecture: OSArchitecture
    os_type: OSType
    target_platform: str
    features: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if OS spec is valid."""
        return bool(self.name and self.target_platform)


class OSDomainKnowledge:
    """
    Operating systems development knowledge base.
    
    Provides patterns, best practices, and specifications for
    building operating systems and system-level software.
    """
    
    def __init__(self):
        """Initialize OS domain knowledge."""
        self.kernel_patterns: Dict[str, str] = {}
        self.memory_patterns: List[str] = []
        self.concurrency_patterns: List[str] = []
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize OS patterns."""
        self.kernel_patterns = {
            "interrupt_handling": "Top-half and bottom-half processing",
            "system_calls": "User-kernel mode transition",
            "context_switching": "Save and restore process state",
            "virtual_memory": "Page tables and TLB management",
            "process_scheduling": "Fair CPU time allocation",
            "ipc": "Inter-process communication mechanisms"
        }
        
        self.memory_patterns = [
            "Paging for virtual memory",
            "Segmentation for memory protection",
            "Demand paging for efficiency",
            "Copy-on-write for fork()",
            "Memory-mapped files",
            "Slab allocation for kernel objects",
            "Buddy system for page allocation",
            "TLB shootdown for multicore"
        ]
        
        self.concurrency_patterns = [
            "Spinlocks for short critical sections",
            "Mutexes for longer critical sections",
            "Semaphores for resource counting",
            "Read-write locks for reader-writer problem",
            "RCU for read-mostly data structures",
            "Atomic operations for lock-free programming",
            "Memory barriers for ordering",
            "Futexes for user-space synchronization"
        ]
    
    def validate_os_spec(
        self,
        spec: OSSpec
    ) -> Tuple[bool, List[str], str]:
        """
        Validate OS specification.
        
        Args:
            spec: OS specification
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        if not spec.is_valid():
            errors.append("Invalid OS specification")
        
        # Check for essential features
        essential_features = [
            "process_management",
            "memory_management",
            "file_system"
        ]
        
        for feature in essential_features:
            if feature not in spec.features:
                errors.append(f"Missing essential feature: {feature}")
        
        # Check constraints
        if "max_processes" in spec.constraints:
            if spec.constraints["max_processes"] < 1:
                errors.append("max_processes must be at least 1")
        
        if "memory_size_mb" in spec.constraints:
            if spec.constraints["memory_size_mb"] < 1:
                errors.append("memory_size_mb must be at least 1")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "OS specification valid")
    
    def estimate_complexity(
        self,
        spec: OSSpec
    ) -> Tuple[bool, int, str]:
        """
        Estimate OS implementation complexity.
        
        Args:
            spec: OS specification
        
        Returns:
            Tuple of (success, complexity_score, message)
        """
        complexity = 0
        
        # Base complexity by architecture
        arch_complexity = {
            OSArchitecture.MONOLITHIC: 5,
            OSArchitecture.MICROKERNEL: 7,
            OSArchitecture.HYBRID: 6,
            OSArchitecture.EXOKERNEL: 8,
            OSArchitecture.UNIKERNEL: 4
        }
        
        complexity += arch_complexity.get(spec.architecture, 5)
        
        # Add complexity for features
        feature_complexity = {
            "multiprocessing": 3,
            "virtual_memory": 4,
            "networking": 3,
            "device_drivers": 2,
            "file_system": 3,
            "security": 2,
            "real_time": 4
        }
        
        for feature in spec.features:
            complexity += feature_complexity.get(feature, 1)
        
        return (True, complexity, f"Estimated complexity: {complexity}")
    
    def get_recommended_scheduler(
        self,
        os_type: OSType,
        is_real_time: bool
    ) -> SchedulingAlgorithm:
        """
        Recommend scheduling algorithm.
        
        Args:
            os_type: OS type
            is_real_time: Is real-time OS
        
        Returns:
            Recommended scheduling algorithm
        """
        if is_real_time:
            return SchedulingAlgorithm.PRIORITY
        elif os_type == OSType.RTOS:
            return SchedulingAlgorithm.PRIORITY
        else:
            return SchedulingAlgorithm.CFS
    
    def generate_syscall_stub(
        self,
        syscall_name: str,
        parameters: List[Dict[str, str]]
    ) -> Tuple[bool, str, str]:
        """
        Generate system call stub code.
        
        Args:
            syscall_name: System call name
            parameters: Parameter specifications
        
        Returns:
            Tuple of (success, code, message)
        """
        code = f"// System call: {syscall_name}\n"
        code += f"long sys_{syscall_name}("
        
        param_strs = []
        for param in parameters:
            param_strs.append(f"{param['type']} {param['name']}")
        
        code += ", ".join(param_strs)
        code += ") {\n"
        code += "    // TODO: Implement system call\n"
        code += "    return 0;\n"
        code += "}\n"
        
        return (True, code, "System call stub generated")
    
    def get_memory_layout(
        self,
        architecture: str,
        address_space_bits: int
    ) -> Dict[str, Any]:
        """
        Get recommended memory layout.
        
        Args:
            architecture: CPU architecture (x86_64, arm64, etc.)
            address_space_bits: Address space size in bits
        
        Returns:
            Memory layout specification
        """
        if architecture == "x86_64" and address_space_bits == 64:
            return {
                "kernel_space": {
                    "start": "0xFFFF800000000000",
                    "end": "0xFFFFFFFFFFFFFFFF",
                    "size_gb": 128
                },
                "user_space": {
                    "start": "0x0000000000000000",
                    "end": "0x00007FFFFFFFFFFF",
                    "size_gb": 128
                },
                "page_size": 4096
            }
        else:
            return {
                "kernel_space": {"size_gb": 1},
                "user_space": {"size_gb": 3},
                "page_size": 4096
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get OS domain knowledge statistics."""
        return {
            "kernel_patterns": len(self.kernel_patterns),
            "memory_patterns": len(self.memory_patterns),
            "concurrency_patterns": len(self.concurrency_patterns),
            "supported_architectures": len(OSArchitecture)
        }


__all__ = [
    "OSArchitecture",
    "OSType",
    "SchedulingAlgorithm",
    "OSSpec",
    "OSDomainKnowledge",
]
