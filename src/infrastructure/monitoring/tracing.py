"""
Distributed tracing implementation.

This module provides distributed tracing capabilities for tracking requests
across services with span creation, context propagation, and trace export.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time
import random


class SpanKind(Enum):
    """Types of spans."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """
    Span context for propagation.
    
    Attributes:
        trace_id: Unique trace identifier
        span_id: Unique span identifier
        parent_span_id: Parent span ID
        trace_flags: Trace flags
        trace_state: Trace state
    """
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    trace_flags: int = 1
    trace_state: str = ""
    
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self.trace_id and self.span_id)


@dataclass
class Span:
    """
    A trace span.
    
    Attributes:
        context: Span context
        name: Span name
        kind: Span kind
        start_time: Start timestamp
        end_time: End timestamp
        attributes: Span attributes
        events: Span events
        status_code: Status code
        status_message: Status message
    """
    context: SpanContext
    name: str
    kind: SpanKind
    start_time: float
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status_code: int = 0
    status_message: str = ""
    
    def is_valid(self) -> bool:
        """Check if span is valid."""
        return self.context.is_valid() and bool(self.name)
    
    def is_ended(self) -> bool:
        """Check if span has ended."""
        return self.end_time > 0
    
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time == 0:
            return 0.0
        
        return (self.end_time - self.start_time) * 1000


class TracingManager:
    """
    Distributed tracing manager.
    
    Manages trace creation, span lifecycle, and trace export.
    """
    
    def __init__(self, service_name: str, sample_rate: float = 1.0):
        """
        Initialize tracing manager.
        
        Args:
            service_name: Name of this service
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.sample_rate = sample_rate
        self.traces: Dict[str, List[Span]] = {}
        self.active_spans: Dict[str, Span] = {}
    
    def validate_config(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if not self.service_name:
            return (False, "service_name cannot be empty")
        
        if not 0.0 <= self.sample_rate <= 1.0:
            return (False, "sample_rate must be between 0.0 and 1.0")
        
        return (True, "")
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent_context: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Span], str]:
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Span kind
            parent_context: Parent span context
            attributes: Initial attributes
        
        Returns:
            Tuple of (success, span or None, message)
        """
        if not name:
            return (False, None, "name cannot be empty")
        
        # Check sampling
        if random.random() > self.sample_rate:
            return (False, None, "Span not sampled")
        
        # Generate IDs
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = self._generate_trace_id()
            parent_span_id = ""
        
        span_id = self._generate_span_id()
        
        # Create context
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id
        )
        
        # Create span
        span = Span(
            context=context,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        if not span.is_valid():
            return (False, None, "Invalid span created")
        
        # Store span
        self.active_spans[span_id] = span
        
        if trace_id not in self.traces:
            self.traces[trace_id] = []
        
        return (True, span, f"Span started: {span_id}")
    
    def end_span(
        self,
        span: Span,
        status_code: int = 0,
        status_message: str = ""
    ) -> Tuple[bool, str]:
        """
        End a span.
        
        Args:
            span: Span to end
            status_code: Status code (0 = OK)
            status_message: Status message
        
        Returns:
            Tuple of (success, message)
        """
        if not span:
            return (False, "span cannot be None")
        
        if span.is_ended():
            return (False, "Span already ended")
        
        # Update span
        span.end_time = time.time()
        span.status_code = status_code
        span.status_message = status_message
        
        # Move to trace
        self.traces[span.context.trace_id].append(span)
        
        # Remove from active
        self.active_spans.pop(span.context.span_id, None)
        
        return (True, f"Span ended: {span.context.span_id}")
    
    def add_span_event(
        self,
        span: Span,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Add an event to a span.
        
        Args:
            span: Span to add event to
            name: Event name
            attributes: Event attributes
        
        Returns:
            Tuple of (success, message)
        """
        if not span:
            return (False, "span cannot be None")
        
        if not name:
            return (False, "name cannot be empty")
        
        if span.is_ended():
            return (False, "Cannot add event to ended span")
        
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        
        span.events.append(event)
        
        return (True, f"Event '{name}' added to span")
    
    def set_span_attribute(
        self,
        span: Span,
        key: str,
        value: Any
    ) -> Tuple[bool, str]:
        """
        Set a span attribute.
        
        Args:
            span: Span to set attribute on
            key: Attribute key
            value: Attribute value
        
        Returns:
            Tuple of (success, message)
        """
        if not span:
            return (False, "span cannot be None")
        
        if not key:
            return (False, "key cannot be empty")
        
        span.attributes[key] = value
        
        return (True, f"Attribute '{key}' set")
    
    def get_trace(self, trace_id: str) -> Tuple[bool, List[Span], str]:
        """
        Get all spans for a trace.
        
        Args:
            trace_id: Trace identifier
        
        Returns:
            Tuple of (success, spans list, message)
        """
        if not trace_id:
            return (False, [], "trace_id cannot be empty")
        
        if trace_id not in self.traces:
            return (False, [], f"Trace {trace_id} not found")
        
        spans = self.traces[trace_id]
        
        return (True, spans, f"Retrieved {len(spans)} spans")
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        timestamp = int(time.time() * 1000000)
        random_part = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        return f"{timestamp:016x}{random_part:016x}"
    
    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        return f"{random.randint(0, 0xFFFFFFFFFFFFFFFF):016x}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        total_spans = sum(len(spans) for spans in self.traces.values())
        
        return {
            "service_name": self.service_name,
            "sample_rate": self.sample_rate,
            "total_traces": len(self.traces),
            "total_spans": total_spans,
            "active_spans": len(self.active_spans)
        }
