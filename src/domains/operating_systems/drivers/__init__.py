"""Device driver specifications for OS development."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

class DriverType(Enum):
    CHARACTER = "character"
    BLOCK = "block"
    NETWORK = "network"

@dataclass
class DriverSpec:
    name: str
    driver_type: DriverType
    irq_number: Optional[int] = None
    io_ports: List[int] = None
    
    def __post_init__(self):
        if self.io_ports is None:
            self.io_ports = []

class DriverKnowledge:
    def __init__(self):
        self.patterns = {
            "probe_remove": "Device discovery and cleanup",
            "interrupt_handling": "IRQ handling",
            "dma": "Direct memory access",
            "power_management": "Suspend/resume support",
            "hotplug": "Dynamic device addition/removal"
        }
    
    def validate_driver_spec(self, spec: DriverSpec) -> Tuple[bool, List[str], str]:
        errors = []
        if not spec.name:
            errors.append("Driver name required")
        if errors:
            return (False, errors, f"{len(errors)} errors")
        return (True, [], "Valid")
    
    def generate_driver_skeleton(self, spec: DriverSpec) -> Tuple[bool, str, str]:
        code = f"// {spec.name} driver\n"
        code += f"#include <linux/module.h>\n\n"
        code += f"static int {spec.name}_probe(void) {{\n    return 0;\n}}\n\n"
        code += f"static void {spec.name}_remove(void) {{}}\n\n"
        code += f"module_init({spec.name}_probe);\n"
        code += f"module_exit({spec.name}_remove);\n"
        return (True, code, "Generated")
    
    def generate_interrupt_handler(self, spec: DriverSpec) -> Tuple[bool, str, str]:
        if spec.irq_number is None:
            return (False, "", "IRQ number required")
        code = f"irqreturn_t {spec.name}_irq_handler(int irq, void *dev_id) {{\n"
        code += "    // Handle interrupt\n    return IRQ_HANDLED;\n}\n"
        return (True, code, "Generated")
    
    def generate_file_operations(self, spec: DriverSpec) -> Tuple[bool, str, str]:
        code = f"static struct file_operations {spec.name}_fops = {{\n"
        code += "    .owner = THIS_MODULE,\n"
        code += "    .open = device_open,\n"
        code += "    .release = device_release,\n"
        code += "    .read = device_read,\n"
        code += "    .write = device_write,\n"
        code += "};\n"
        return (True, code, "Generated")
    
    def generate_dma_code(self, spec: DriverSpec) -> Tuple[bool, str, str]:
        code = f"// DMA setup for {spec.name}\n"
        code += "void setup_dma() {\n"
        code += "    // Configure DMA\n}\n"
        return (True, code, "Generated")
    
    def get_stats(self) -> Dict[str, Any]:
        return {"patterns": len(self.patterns), "driver_types": len(DriverType)}

__all__ = ["DriverType", "DriverSpec", "DriverKnowledge"]
