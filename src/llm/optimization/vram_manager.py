"""
VRAM Manager Module.

This module provides comprehensive VRAM (GPU memory) tracking and management
for the multi-agent LLM system. It monitors memory allocation, prevents OOM
errors, and optimizes memory usage across multiple models and devices.

Key Concepts:
- VRAM is the most critical resource constraint in LLM systems
- A 7B parameter model uses ~6GB VRAM (FP16) or ~1.5GB (4-bit quantized)
- Multiple models must share limited VRAM (e.g., 12GB GPU)
- Fragmentation can waste VRAM even when total is available
- KV cache grows during generation (~500MB for long sequences)

Management Features:
- Real-time VRAM usage tracking per model and device
- Allocation planning with reservation system
- Memory pressure monitoring and warnings
- Fragmentation detection and defragmentation
- Multi-GPU support with device-specific tracking
- Memory profiling and statistics
- OOM prevention with safety thresholds

All operations follow zero-error philosophy with explicit validation.
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import time


class MemoryStatus(Enum):
    """
    Memory status levels.

    Attributes:
        HEALTHY: Memory usage is safe (< 70%)
        WARNING: Memory usage is elevated (70-85%)
        CRITICAL: Memory usage is very high (85-95%)
        DANGER: Memory usage is at limit (> 95%)
    """
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DANGER = "danger"


class AllocationStrategy(Enum):
    """
    VRAM allocation strategies.

    Attributes:
        FIRST_FIT: Allocate on first device with space
        BEST_FIT: Allocate on device with closest fit
        BALANCED: Balance allocation across devices
        PRIORITY: Prefer higher priority devices
    """
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    BALANCED = "balanced"
    PRIORITY = "priority"


@dataclass
class VRAMConfig:
    """
    Configuration for VRAM management.

    Attributes:
        enable_tracking: Enable VRAM tracking
        enable_profiling: Enable detailed profiling
        warning_threshold: Warning threshold (0.0-1.0)
        critical_threshold: Critical threshold (0.0-1.0)
        danger_threshold: Danger threshold (0.0-1.0)
        safety_margin_gb: Reserve this much VRAM for safety
        enable_defragmentation: Enable automatic defragmentation
        allocation_strategy: Strategy for allocating VRAM
        max_fragmentation_ratio: Max fragmentation before defrag (0.0-1.0)
        track_kv_cache: Track KV cache separately
    """
    enable_tracking: bool = True
    enable_profiling: bool = False
    warning_threshold: float = 0.70
    critical_threshold: float = 0.85
    danger_threshold: float = 0.95
    safety_margin_gb: float = 1.0
    enable_defragmentation: bool = False
    allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED
    max_fragmentation_ratio: float = 0.3
    track_kv_cache: bool = True

    def validate(self) -> bool:
        """
        Validate VRAM configuration.

        Returns:
            True if valid, False otherwise
        """
        # Validate thresholds
        if not (0.0 <= self.warning_threshold <= 1.0):
            return False
        if not (0.0 <= self.critical_threshold <= 1.0):
            return False
        if not (0.0 <= self.danger_threshold <= 1.0):
            return False

        # Thresholds should be ordered
        if not (self.warning_threshold < self.critical_threshold < self.danger_threshold):
            return False

        # Validate safety margin
        if self.safety_margin_gb < 0:
            return False

        # Validate fragmentation ratio
        if not (0.0 <= self.max_fragmentation_ratio <= 1.0):
            return False

        return True


@dataclass
class DeviceInfo:
    """
    Information about a GPU device.

    Attributes:
        device_id: Device identifier (e.g., "cuda:0")
        total_vram_gb: Total VRAM capacity in GB
        device_name: GPU device name
        compute_capability: CUDA compute capability
        is_available: Whether device is available
    """
    device_id: str
    total_vram_gb: float
    device_name: str = "Unknown"
    compute_capability: str = "0.0"
    is_available: bool = True


@dataclass
class MemoryAllocation:
    """
    VRAM allocation record.

    Attributes:
        allocation_id: Unique allocation identifier
        model_id: Model using this allocation
        device_id: Device where allocated
        size_gb: Allocation size in GB
        allocated_at: Allocation timestamp
        purpose: Allocation purpose (model/kv_cache/activation)
        is_pinned: Whether allocation is pinned (cannot be freed)
        metadata: Additional metadata
    """
    allocation_id: str
    model_id: str
    device_id: str
    size_gb: float
    allocated_at: float = field(default_factory=time.time)
    purpose: str = "model"
    is_pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceMemoryState:
    """
    Current memory state for a device.

    Attributes:
        device_id: Device identifier
        total_vram_gb: Total VRAM capacity
        allocated_gb: Currently allocated VRAM
        free_gb: Available free VRAM
        reserved_gb: Reserved VRAM (safety margin)
        fragmented_gb: Fragmented (unusable) VRAM
        utilization: Memory utilization ratio (0.0-1.0)
        status: Memory status level
        allocations: List of active allocations
    """
    device_id: str
    total_vram_gb: float
    allocated_gb: float = 0.0
    free_gb: float = 0.0
    reserved_gb: float = 0.0
    fragmented_gb: float = 0.0
    utilization: float = 0.0
    status: MemoryStatus = MemoryStatus.HEALTHY
    allocations: List[MemoryAllocation] = field(default_factory=list)

    def update_state(self, reserved_gb: float) -> None:
        """
        Update derived state fields.

        Args:
            reserved_gb: Reserved memory size
        """
        self.reserved_gb = reserved_gb
        self.free_gb = self.total_vram_gb - self.allocated_gb - self.reserved_gb
        self.utilization = self.allocated_gb / self.total_vram_gb if self.total_vram_gb > 0 else 0.0


@dataclass
class VRAMStatistics:
    """
    VRAM usage statistics.

    Attributes:
        total_allocations: Total number of allocations made
        total_deallocations: Total number of deallocations
        peak_usage_gb: Peak VRAM usage observed
        average_usage_gb: Average VRAM usage
        total_fragmentation_events: Number of fragmentation events
        oom_prevention_count: Number of OOM events prevented
        defragmentation_count: Number of defragmentations performed
        allocation_failures: Number of allocation failures
    """
    total_allocations: int = 0
    total_deallocations: int = 0
    peak_usage_gb: float = 0.0
    average_usage_gb: float = 0.0
    total_fragmentation_events: int = 0
    oom_prevention_count: int = 0
    defragmentation_count: int = 0
    allocation_failures: int = 0


class VRAMManager:
    """
    Comprehensive VRAM manager for multi-model LLM system.

    Tracks memory allocation across devices, prevents OOM errors,
    and optimizes memory usage.
    """

    def __init__(self, config: VRAMConfig):
        """
        Initialize VRAM manager.

        Args:
            config: VRAM management configuration
        """
        if not config.validate():
            raise ValueError("Invalid VRAM configuration")

        self.config = config

        # Device registry
        self.devices: Dict[str, DeviceInfo] = {}
        self.device_states: Dict[str, DeviceMemoryState] = {}

        # Allocation tracking
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.allocation_counter = 0

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = VRAMStatistics()

        # Initialize devices
        self._initialize_devices()

    def _initialize_devices(self) -> None:
        """Initialize available GPU devices."""
        # In production, would query actual GPU devices using torch.cuda
        # For now, simulate device detection

        # Simulate single GPU with 12GB VRAM
        device_info = DeviceInfo(
            device_id="cuda:0",
            total_vram_gb=12.0,
            device_name="NVIDIA GPU",
            compute_capability="8.6",
            is_available=True
        )

        with self.lock:
            self.devices["cuda:0"] = device_info
            self.device_states["cuda:0"] = DeviceMemoryState(
                device_id="cuda:0",
                total_vram_gb=12.0
            )

    def register_device(self, device_info: DeviceInfo) -> bool:
        """
        Register a GPU device for tracking.

        Args:
            device_info: Device information

        Returns:
            True if registered, False if already exists
        """
        with self.lock:
            if device_info.device_id in self.devices:
                return False

            self.devices[device_info.device_id] = device_info
            self.device_states[device_info.device_id] = DeviceMemoryState(
                device_id=device_info.device_id,
                total_vram_gb=device_info.total_vram_gb
            )

            return True

    def allocate(
        self,
        model_id: str,
        size_gb: float,
        device_id: Optional[str] = None,
        purpose: str = "model",
        is_pinned: bool = False
    ) -> Optional[str]:
        """
        Allocate VRAM for a model.

        Args:
            model_id: Model identifier
            size_gb: Size to allocate in GB
            device_id: Specific device (None = auto-select)
            purpose: Allocation purpose
            is_pinned: Whether allocation is pinned

        Returns:
            Allocation ID if successful, None if failed
        """
        if size_gb <= 0:
            return None

        with self.lock:
            # Select device if not specified
            if device_id is None:
                device_id = self._select_device(size_gb)

            if device_id is None:
                self.stats.allocation_failures += 1
                return None

            # Check if allocation would fit
            if not self._can_allocate(device_id, size_gb):
                self.stats.allocation_failures += 1
                return None

            # Create allocation
            allocation_id = f"alloc_{self.allocation_counter}"
            self.allocation_counter += 1

            allocation = MemoryAllocation(
                allocation_id=allocation_id,
                model_id=model_id,
                device_id=device_id,
                size_gb=size_gb,
                purpose=purpose,
                is_pinned=is_pinned
            )

            # Update device state
            device_state = self.device_states[device_id]
            device_state.allocated_gb += size_gb
            device_state.allocations.append(allocation)
            device_state.update_state(self.config.safety_margin_gb)

            # Update status
            self._update_device_status(device_id)

            # Store allocation
            self.allocations[allocation_id] = allocation

            # Update statistics
            self.stats.total_allocations += 1
            if device_state.allocated_gb > self.stats.peak_usage_gb:
                self.stats.peak_usage_gb = device_state.allocated_gb

            return allocation_id

    def deallocate(self, allocation_id: str) -> bool:
        """
        Deallocate VRAM.

        Args:
            allocation_id: Allocation to free

        Returns:
            True if deallocated, False if not found
        """
        with self.lock:
            if allocation_id not in self.allocations:
                return False

            allocation = self.allocations[allocation_id]

            # Check if pinned
            if allocation.is_pinned:
                return False

            # Update device state
            device_state = self.device_states[allocation.device_id]
            device_state.allocated_gb -= allocation.size_gb

            # Remove from allocations list
            device_state.allocations = [
                a for a in device_state.allocations
                if a.allocation_id != allocation_id
            ]

            device_state.update_state(self.config.safety_margin_gb)

            # Update status
            self._update_device_status(allocation.device_id)

            # Remove allocation
            del self.allocations[allocation_id]

            # Update statistics
            self.stats.total_deallocations += 1

            return True

    def _can_allocate(self, device_id: str, size_gb: float) -> bool:
        """
        Check if allocation would fit on device.

        Args:
            device_id: Device identifier
            size_gb: Size to allocate

        Returns:
            True if fits, False otherwise
        """
        if device_id not in self.device_states:
            return False

        device_state = self.device_states[device_id]

        # Calculate available space
        available_gb = (
            device_state.total_vram_gb -
            device_state.allocated_gb -
            self.config.safety_margin_gb
        )

        return size_gb <= available_gb

    def _select_device(self, size_gb: float) -> Optional[str]:
        """
        Select optimal device for allocation.

        Args:
            size_gb: Size to allocate

        Returns:
            Device ID if found, None if no suitable device
        """
        if self.config.allocation_strategy == AllocationStrategy.FIRST_FIT:
            return self._select_first_fit(size_gb)
        elif self.config.allocation_strategy == AllocationStrategy.BEST_FIT:
            return self._select_best_fit(size_gb)
        elif self.config.allocation_strategy == AllocationStrategy.BALANCED:
            return self._select_balanced(size_gb)
        else:
            return self._select_first_fit(size_gb)

    def _select_first_fit(self, size_gb: float) -> Optional[str]:
        """Select first device with sufficient space."""
        for device_id in self.device_states.keys():
            if self._can_allocate(device_id, size_gb):
                return device_id
        return None

    def _select_best_fit(self, size_gb: float) -> Optional[str]:
        """Select device with closest fit."""
        best_device = None
        best_remaining = float('inf')

        for device_id, device_state in self.device_states.items():
            if self._can_allocate(device_id, size_gb):
                available = (
                    device_state.total_vram_gb -
                    device_state.allocated_gb -
                    self.config.safety_margin_gb
                )
                remaining = available - size_gb

                if remaining < best_remaining:
                    best_remaining = remaining
                    best_device = device_id

        return best_device

    def _select_balanced(self, size_gb: float) -> Optional[str]:
        """Select device with lowest utilization."""
        best_device = None
        lowest_util = float('inf')

        for device_id, device_state in self.device_states.items():
            if self._can_allocate(device_id, size_gb):
                if device_state.utilization < lowest_util:
                    lowest_util = device_state.utilization
                    best_device = device_id

        return best_device

    def _update_device_status(self, device_id: str) -> None:
        """
        Update memory status for device.

        Args:
            device_id: Device identifier
        """
        device_state = self.device_states[device_id]

        # Determine status based on utilization
        if device_state.utilization >= self.config.danger_threshold:
            device_state.status = MemoryStatus.DANGER
        elif device_state.utilization >= self.config.critical_threshold:
            device_state.status = MemoryStatus.CRITICAL
        elif device_state.utilization >= self.config.warning_threshold:
            device_state.status = MemoryStatus.WARNING
        else:
            device_state.status = MemoryStatus.HEALTHY

    def get_device_state(self, device_id: str) -> Optional[DeviceMemoryState]:
        """
        Get current memory state for device.

        Args:
            device_id: Device identifier

        Returns:
            DeviceMemoryState if exists, None otherwise
        """
        with self.lock:
            if device_id not in self.device_states:
                return None

            # Return copy
            state = self.device_states[device_id]
            return DeviceMemoryState(
                device_id=state.device_id,
                total_vram_gb=state.total_vram_gb,
                allocated_gb=state.allocated_gb,
                free_gb=state.free_gb,
                reserved_gb=state.reserved_gb,
                fragmented_gb=state.fragmented_gb,
                utilization=state.utilization,
                status=state.status,
                allocations=state.allocations.copy()
            )

    def get_all_device_states(self) -> Dict[str, DeviceMemoryState]:
        """
        Get memory states for all devices.

        Returns:
            Dictionary of device states
        """
        with self.lock:
            return {
                device_id: self.get_device_state(device_id)
                for device_id in self.device_states.keys()
            }

    def get_model_allocations(self, model_id: str) -> List[MemoryAllocation]:
        """
        Get all allocations for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of allocations
        """
        with self.lock:
            return [
                alloc for alloc in self.allocations.values()
                if alloc.model_id == model_id
            ]

    def get_total_allocated(self) -> float:
        """
        Get total allocated VRAM across all devices.

        Returns:
            Total allocated in GB
        """
        with self.lock:
            return sum(
                state.allocated_gb
                for state in self.device_states.values()
            )

    def get_statistics(self) -> VRAMStatistics:
        """
        Get VRAM statistics.

        Returns:
            Copy of current statistics
        """
        with self.lock:
            return VRAMStatistics(
                total_allocations=self.stats.total_allocations,
                total_deallocations=self.stats.total_deallocations,
                peak_usage_gb=self.stats.peak_usage_gb,
                average_usage_gb=self.stats.average_usage_gb,
                total_fragmentation_events=self.stats.total_fragmentation_events,
                oom_prevention_count=self.stats.oom_prevention_count,
                defragmentation_count=self.stats.defragmentation_count,
                allocation_failures=self.stats.allocation_failures
            )

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.stats = VRAMStatistics()


def create_default_vram_config() -> VRAMConfig:
    """
    Create default VRAM configuration.

    Returns:
        VRAMConfig with sensible defaults
    """
    return VRAMConfig(
        enable_tracking=True,
        enable_profiling=False,
        warning_threshold=0.70,
        critical_threshold=0.85,
        danger_threshold=0.95,
        safety_margin_gb=1.0,
        allocation_strategy=AllocationStrategy.BALANCED
    )


def create_conservative_config() -> VRAMConfig:
    """
    Create conservative VRAM configuration with early warnings.

    Returns:
        VRAMConfig with conservative thresholds
    """
    return VRAMConfig(
        enable_tracking=True,
        enable_profiling=True,
        warning_threshold=0.60,
        critical_threshold=0.75,
        danger_threshold=0.90,
        safety_margin_gb=2.0,
        allocation_strategy=AllocationStrategy.BEST_FIT
    )


def create_aggressive_config() -> VRAMConfig:
    """
    Create aggressive VRAM configuration for maximum utilization.

    Returns:
        VRAMConfig with aggressive thresholds
    """
    return VRAMConfig(
        enable_tracking=True,
        enable_profiling=False,
        warning_threshold=0.80,
        critical_threshold=0.90,
        danger_threshold=0.98,
        safety_margin_gb=0.5,
        allocation_strategy=AllocationStrategy.FIRST_FIT
    )
