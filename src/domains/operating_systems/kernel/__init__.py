"""Kernel component specifications for OS development."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class KernelType(Enum):
    MONOLITHIC = "monolithic"
    MICRO = "micro"
    HYBRID = "hybrid"

@dataclass
class KernelSpec:
    name: str
    kernel_type: KernelType
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

class KernelKnowledge:
    def __init__(self):
        self.patterns = {
            "interrupt_handling": "Top/bottom half processing",
            "process_scheduling": "Fair CPU allocation",
            "memory_management": "Virtual memory with paging",
            "system_calls": "User-kernel transitions",
            "device_management": "Driver framework"
        }
    
    def validate_kernel_spec(self, spec: KernelSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Kernel name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def estimate_complexity(self, spec: KernelSpec) -> Tuple[bool, int, str]:
        complexity = 5 if spec.kernel_type == KernelType.MONOLITHIC else 7
        complexity += len(spec.features)
        return (True, complexity, f"Complexity: {complexity}")
    
    def get_scheduling_code(self, algorithm: str) -> Tuple[bool, str, str]:
        code = f"// {algorithm} scheduler\nvoid schedule() {{\n    // TODO: Implement\n}}\n"
        return (True, code, "Generated")
    
    def get_memory_management_code(self) -> Tuple[bool, str, str]:
        code = "// Memory management\nvoid* kmalloc(size_t size) {\n    // TODO: Implement\n    return NULL;\n}\n"
        return (True, code, "Generated")
    
    def get_interrupt_handler_code(self, irq_num: int) -> Tuple[bool, str, str]:
        code = f"// IRQ {irq_num} handler\nvoid irq{irq_num}_handler() {{\n    // TODO: Implement\n}}\n"
        return (True, code, "Generated")
    
    def get_syscall_table_code(self, syscalls: List[str]) -> Tuple[bool, str, str]:
        code = "// System call table\nvoid* syscall_table[] = {\n"
        for sc in syscalls:
            code += f"    sys_{sc},\n"
        code += "};\n"
        return (True, code, "Generated")
    
    def get_context_switch_code(self) -> Tuple[bool, str, str]:
        code = "// Context switch\nvoid context_switch(task_t *prev, task_t *next) {\n    // Save prev state\n    // Load next state\n}\n"
        return (True, code, "Generated")
    
    def get_process_creation_code(self) -> Tuple[bool, str, str]:
        code = "// Process creation\nint fork() {\n    // TODO: Implement\n    return 0;\n}\n"
        return (True, code, "Generated")
    
    def get_ipc_code(self, mechanism: str) -> Tuple[bool, str, str]:
        code = f"// IPC: {mechanism}\nint ipc_{mechanism}() {{\n    // TODO: Implement\n    return 0;\n}}\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "kernel_types": len(KernelType)}

__all__ = ["KernelType", "KernelSpec", "KernelKnowledge"]
