"""
Metrics collection implementation.

This module provides comprehensive Prometheus-style metrics collection including
counters, gauges, histograms, and summaries with label support and explicit error handling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import time
import threading


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """
    A metric value with timestamp and labels.
    
    Attributes:
        value: Metric value
        labels: Label key-value pairs
        timestamp: When value was recorded
    """
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0
    
    def label_string(self) -> str:
        """Get formatted label string."""
        if not self.labels:
            return ""
        
        pairs = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(pairs) + "}"


@dataclass
class MetricMetadata:
    """
    Metadata for a metric.
    
    Attributes:
        name: Metric name
        metric_type: Type of metric
        help_text: Help description
        unit: Unit of measurement
        label_names: Expected label names
    """
    name: str
    metric_type: MetricType
    help_text: str
    unit: str = ""
    label_names: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if metadata is valid."""
        return bool(self.name and self.help_text)


class Counter:
    """
    Counter metric that only increases.
    
    Counters are used for cumulative values like request counts,
    errors, tasks completed, etc.
    """
    
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        """
        Initialize counter.
        
        Args:
            name: Counter name
            help_text: Help description
            label_names: Optional label names
        """
        self.metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.COUNTER,
            help_text=help_text,
            label_names=label_names or []
        )
        self.values: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Increment counter.
        
        Args:
            amount: Amount to increment by
            labels: Optional labels
        
        Returns:
            Tuple of (success, message)
        """
        if amount < 0:
            return (False, "Counter can only increase")
        
        # Validate labels
        if labels:
            success, msg = self._validate_labels(labels)
            if not success:
                return (False, msg)
        
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            if label_key not in self.values:
                self.values[label_key] = 0.0
            
            self.values[label_key] += amount
        
        return (True, f"Counter incremented by {amount}")
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, float, str]:
        """
        Get counter value.
        
        Args:
            labels: Optional labels
        
        Returns:
            Tuple of (success, value, message)
        """
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            value = self.values.get(label_key, 0.0)
        
        return (True, value, "Value retrieved")
    
    def _validate_labels(self, labels: Dict[str, str]) -> Tuple[bool, str]:
        """Validate labels match expected names."""
        if not self.metadata.label_names:
            return (True, "")
        
        provided = set(labels.keys())
        expected = set(self.metadata.label_names)
        
        if provided != expected:
            return (False, f"Labels mismatch. Expected: {expected}, Got: {provided}")
        
        return (True, "")
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Get string key for labels."""
        if not labels:
            return ""
        
        pairs = [f"{k}={v}" for k, v in sorted(labels.items())]
        return ",".join(pairs)
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self.lock:
            self.values.clear()


class Gauge:
    """
    Gauge metric that can increase or decrease.
    
    Gauges are used for values that can go up and down like
    memory usage, queue size, active connections, etc.
    """
    
    def __init__(self, name: str, help_text: str, label_names: Optional[List[str]] = None):
        """
        Initialize gauge.
        
        Args:
            name: Gauge name
            help_text: Help description
            label_names: Optional label names
        """
        self.metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.GAUGE,
            help_text=help_text,
            label_names=label_names or []
        )
        self.values: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Set gauge value.
        
        Args:
            value: Value to set
            labels: Optional labels
        
        Returns:
            Tuple of (success, message)
        """
        # Validate labels
        if labels:
            success, msg = self._validate_labels(labels)
            if not success:
                return (False, msg)
        
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            self.values[label_key] = value
        
        return (True, f"Gauge set to {value}")
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Increment gauge.
        
        Args:
            amount: Amount to increment by
            labels: Optional labels
        
        Returns:
            Tuple of (success, message)
        """
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            if label_key not in self.values:
                self.values[label_key] = 0.0
            
            self.values[label_key] += amount
        
        return (True, f"Gauge incremented by {amount}")
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Decrement gauge.
        
        Args:
            amount: Amount to decrement by
            labels: Optional labels
        
        Returns:
            Tuple of (success, message)
        """
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            if label_key not in self.values:
                self.values[label_key] = 0.0
            
            self.values[label_key] -= amount
        
        return (True, f"Gauge decremented by {amount}")
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, float, str]:
        """
        Get gauge value.
        
        Args:
            labels: Optional labels
        
        Returns:
            Tuple of (success, value, message)
        """
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            value = self.values.get(label_key, 0.0)
        
        return (True, value, "Value retrieved")
    
    def _validate_labels(self, labels: Dict[str, str]) -> Tuple[bool, str]:
        """Validate labels match expected names."""
        if not self.metadata.label_names:
            return (True, "")
        
        provided = set(labels.keys())
        expected = set(self.metadata.label_names)
        
        if provided != expected:
            return (False, f"Labels mismatch. Expected: {expected}, Got: {provided}")
        
        return (True, "")
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Get string key for labels."""
        if not labels:
            return ""
        
        pairs = [f"{k}={v}" for k, v in sorted(labels.items())]
        return ",".join(pairs)
    
    def reset(self) -> None:
        """Reset gauge to zero."""
        with self.lock:
            self.values.clear()


class Histogram:
    """
    Histogram metric for tracking distributions.
    
    Histograms are used for request durations, response sizes,
    and other distributions.
    """
    
    def __init__(
        self,
        name: str,
        help_text: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None
    ):
        """
        Initialize histogram.
        
        Args:
            name: Histogram name
            help_text: Help description
            buckets: Bucket boundaries
            label_names: Optional label names
        """
        self.metadata = MetricMetadata(
            name=name,
            metric_type=MetricType.HISTOGRAM,
            help_text=help_text,
            label_names=label_names or []
        )
        
        # Default buckets if not provided
        if buckets is None:
            self.buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        else:
            self.buckets = sorted(buckets)
        
        # Storage: label_key -> {bucket -> count}
        self.bucket_counts: Dict[str, Dict[float, int]] = {}
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        """
        Observe a value.
        
        Args:
            value: Value to observe
            labels: Optional labels
        
        Returns:
            Tuple of (success, message)
        """
        if value < 0:
            return (False, "Value cannot be negative")
        
        # Validate labels
        if labels:
            success, msg = self._validate_labels(labels)
            if not success:
                return (False, msg)
        
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            # Initialize if needed
            if label_key not in self.bucket_counts:
                self.bucket_counts[label_key] = {b: 0 for b in self.buckets}
                self.bucket_counts[label_key][float('inf')] = 0
                self.sums[label_key] = 0.0
                self.counts[label_key] = 0
            
            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[label_key][bucket] += 1
            
            # Always increment +Inf bucket
            self.bucket_counts[label_key][float('inf')] += 1
            
            # Update sum and count
            self.sums[label_key] += value
            self.counts[label_key] += 1
        
        return (True, f"Observed value {value}")
    
    def get_stats(self, labels: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any], str]:
        """
        Get histogram statistics.
        
        Args:
            labels: Optional labels
        
        Returns:
            Tuple of (success, stats dict, message)
        """
        label_key = self._get_label_key(labels or {})
        
        with self.lock:
            if label_key not in self.counts:
                return (False, {}, "No observations for these labels")
            
            count = self.counts[label_key]
            total_sum = self.sums[label_key]
            avg = total_sum / count if count > 0 else 0.0
            
            stats = {
                "count": count,
                "sum": total_sum,
                "average": avg,
                "buckets": dict(self.bucket_counts[label_key])
            }
        
        return (True, stats, "Stats retrieved")
    
    def _validate_labels(self, labels: Dict[str, str]) -> Tuple[bool, str]:
        """Validate labels match expected names."""
        if not self.metadata.label_names:
            return (True, "")
        
        provided = set(labels.keys())
        expected = set(self.metadata.label_names)
        
        if provided != expected:
            return (False, f"Labels mismatch. Expected: {expected}, Got: {provided}")
        
        return (True, "")
    
    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """Get string key for labels."""
        if not labels:
            return ""
        
        pairs = [f"{k}={v}" for k, v in sorted(labels.items())]
        return ",".join(pairs)
    
    def reset(self) -> None:
        """Reset histogram."""
        with self.lock:
            self.bucket_counts.clear()
            self.sums.clear()
            self.counts.clear()


class MetricsCollector:
    """
    Central metrics collector and registry.
    
    Manages all metrics and provides registration, collection,
    and export functionality.
    """
    
    def __init__(self, namespace: str = ""):
        """
        Initialize metrics collector.
        
        Args:
            namespace: Optional namespace prefix for metrics
        """
        self.namespace = namespace
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.lock = threading.Lock()
    
    def register_counter(
        self,
        name: str,
        help_text: str,
        label_names: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Counter], str]:
        """
        Register a new counter.
        
        Args:
            name: Counter name
            help_text: Help description
            label_names: Optional label names
        
        Returns:
            Tuple of (success, counter or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if not help_text:
            return (False, None, "help_text cannot be empty")
        
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name in self.counters:
                return (False, None, f"Counter '{full_name}' already registered")
            
            counter = Counter(full_name, help_text, label_names)
            self.counters[full_name] = counter
        
        return (True, counter, f"Counter '{full_name}' registered")
    
    def register_gauge(
        self,
        name: str,
        help_text: str,
        label_names: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Gauge], str]:
        """
        Register a new gauge.
        
        Args:
            name: Gauge name
            help_text: Help description
            label_names: Optional label names
        
        Returns:
            Tuple of (success, gauge or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if not help_text:
            return (False, None, "help_text cannot be empty")
        
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name in self.gauges:
                return (False, None, f"Gauge '{full_name}' already registered")
            
            gauge = Gauge(full_name, help_text, label_names)
            self.gauges[full_name] = gauge
        
        return (True, gauge, f"Gauge '{full_name}' registered")
    
    def register_histogram(
        self,
        name: str,
        help_text: str,
        buckets: Optional[List[float]] = None,
        label_names: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[Histogram], str]:
        """
        Register a new histogram.
        
        Args:
            name: Histogram name
            help_text: Help description
            buckets: Bucket boundaries
            label_names: Optional label names
        
        Returns:
            Tuple of (success, histogram or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        if not help_text:
            return (False, None, "help_text cannot be empty")
        
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name in self.histograms:
                return (False, None, f"Histogram '{full_name}' already registered")
            
            histogram = Histogram(full_name, help_text, buckets, label_names)
            self.histograms[full_name] = histogram
        
        return (True, histogram, f"Histogram '{full_name}' registered")
    
    def get_counter(self, name: str) -> Tuple[bool, Optional[Counter], str]:
        """Get a registered counter."""
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name not in self.counters:
                return (False, None, f"Counter '{full_name}' not found")
            
            counter = self.counters[full_name]
        
        return (True, counter, "Counter retrieved")
    
    def get_gauge(self, name: str) -> Tuple[bool, Optional[Gauge], str]:
        """Get a registered gauge."""
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name not in self.gauges:
                return (False, None, f"Gauge '{full_name}' not found")
            
            gauge = self.gauges[full_name]
        
        return (True, gauge, "Gauge retrieved")
    
    def get_histogram(self, name: str) -> Tuple[bool, Optional[Histogram], str]:
        """Get a registered histogram."""
        full_name = self._get_full_name(name)
        
        with self.lock:
            if full_name not in self.histograms:
                return (False, None, f"Histogram '{full_name}' not found")
            
            histogram = self.histograms[full_name]
        
        return (True, histogram, "Histogram retrieved")
    
    def _get_full_name(self, name: str) -> str:
        """Get full metric name with namespace."""
        if self.namespace:
            return f"{self.namespace}_{name}"
        return name
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collector statistics.
        
        Returns:
            Dictionary with stats
        """
        with self.lock:
            return {
                "namespace": self.namespace,
                "total_counters": len(self.counters),
                "total_gauges": len(self.gauges),
                "total_histograms": len(self.histograms),
                "total_metrics": len(self.counters) + len(self.gauges) + len(self.histograms)
            }
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        with self.lock:
            for counter in self.counters.values():
                counter.reset()
            
            for gauge in self.gauges.values():
                gauge.reset()
            
            for histogram in self.histograms.values():
                histogram.reset()
