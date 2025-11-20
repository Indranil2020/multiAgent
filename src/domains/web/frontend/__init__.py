"""
Frontend framework specifications and patterns.

This module provides comprehensive knowledge for frontend development
including React, Vue, Angular, Svelte and associated patterns, state
management, routing, and component architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class FrontendFramework(Enum):
    """Frontend frameworks and libraries."""
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    SVELTE = "svelte"
    SOLID = "solid"
    PREACT = "preact"
    LIT = "lit"


class ComponentPattern(Enum):
    """Component architecture patterns."""
    FUNCTIONAL = "functional"
    CLASS_BASED = "class_based"
    COMPOSITION = "composition"
    HOOKS = "hooks"
    MIXINS = "mixins"
    HIGHER_ORDER = "higher_order_component"


class StateManagement(Enum):
    """State management solutions."""
    REDUX = "redux"
    MOBX = "mobx"
    VUEX = "vuex"
    PINIA = "pinia"
    CONTEXT_API = "context_api"
    RECOIL = "recoil"
    ZUSTAND = "zustand"
    JOTAI = "jotai"
    XSTATE = "xstate"


class BuildTool(Enum):
    """Frontend build tools."""
    WEBPACK = "webpack"
    VITE = "vite"
    ROLLUP = "rollup"
    PARCEL = "parcel"
    ESBUILD = "esbuild"
    TURBOPACK = "turbopack"



@dataclass
class ComponentSpec:
    """
    Specification for a frontend component.
    
    Attributes:
        name: Component name
        type: Component type (functional/class)
        props: Component props
        state: Component state
        lifecycle_methods: Lifecycle methods needed
        hooks: React hooks used
        events: Events emitted
        children: Child components
    """
    name: str
    type: ComponentPattern
    props: List[str] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    lifecycle_methods: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if component spec is valid."""
        return bool(self.name)


@dataclass
class RoutingConfig:
    """
    Routing configuration.
    
    Attributes:
        router_type: Router type (hash/history/memory)
        routes: List of routes
        lazy_loading: Enable lazy loading
        guards: Route guards
        middleware: Route middleware
    """
    router_type: str
    routes: List[Dict[str, Any]] = field(default_factory=list)
    lazy_loading: bool = True
    guards: List[str] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)


class FrontendKnowledge:
    """
    Frontend development knowledge base.
    
    Provides patterns, best practices, and specifications for
    building modern frontend applications.
    """
    
    def __init__(self):
        """Initialize frontend knowledge."""
        self.component_patterns: Dict[str, str] = {}
        self.performance_patterns: List[str] = []
        self.accessibility_rules: List[str] = []
        self._initialize_patterns()
        self._initialize_performance()
        self._initialize_accessibility()
    
    def _initialize_patterns(self) -> None:
        """Initialize component patterns."""
        self.component_patterns = {
            "container_presentational": "Separate logic from presentation",
            "compound_components": "Components that work together",
            "render_props": "Share code using render props",
            "controlled_uncontrolled": "Form input control patterns",
            "provider_consumer": "Context-based data sharing"
        }
    
    def _initialize_performance(self) -> None:
        """Initialize performance patterns."""
        self.performance_patterns = [
            "Code splitting with dynamic imports",
            "Lazy loading of routes and components",
            "Memoization with React.memo/useMemo",
            "Virtual scrolling for large lists",
            "Image lazy loading and optimization",
            "Bundle size optimization",
            "Tree shaking unused code",
            "Debouncing and throttling events",
            "Web Workers for heavy computation",
            "Service Workers for caching"
        ]
    
    def _initialize_accessibility(self) -> None:
        """Initialize accessibility rules."""
        self.accessibility_rules = [
            "All images must have alt text",
            "Proper heading hierarchy (h1-h6)",
            "Keyboard navigation support",
            "ARIA labels for interactive elements",
            "Sufficient color contrast ratios",
            "Focus indicators visible",
            "Form labels associated with inputs",
            "Skip navigation links",
            "Screen reader friendly",
            "No keyboard traps"
        ]
    
    def validate_component_spec(
        self,
        spec: ComponentSpec
    ) -> Tuple[bool, List[str], str]:
        """
        Validate a component specification.
        
        Args:
            spec: Component specification
        
        Returns:
            Tuple of (is_valid, errors list, message)
        """
        errors = []
        
        if not spec.is_valid():
            errors.append("Invalid component specification")
        
        # Check naming convention
        if not spec.name[0].isupper():
            errors.append("Component name must start with uppercase")
        
        # Check for state in functional components without hooks
        if spec.type == ComponentPattern.FUNCTIONAL and spec.state and not spec.hooks:
            errors.append("Functional component with state must use hooks")
        
        # Check for lifecycle methods in functional components
        if spec.type == ComponentPattern.FUNCTIONAL and spec.lifecycle_methods:
            errors.append("Functional components should use hooks, not lifecycle methods")
        
        # Check for duplicate props
        if len(spec.props) != len(set(spec.props)):
            errors.append("Duplicate props found")
        
        if errors:
            return (False, errors, f"Validation failed with {len(errors)} errors")
        
        return (True, [], "Component specification valid")
    
    def generate_component_code(
        self,
        spec: ComponentSpec,
        framework: FrontendFramework
    ) -> Tuple[bool, str, str]:
        """
        Generate component code from specification.
        
        Args:
            spec: Component specification
            framework: Target framework
        
        Returns:
            Tuple of (success, code, message)
        """
        if not spec.is_valid():
            return (False, "", "Invalid component specification")
        
        if framework == FrontendFramework.REACT:
            return self._generate_react_component(spec)
        elif framework == FrontendFramework.VUE:
            return self._generate_vue_component(spec)
        elif framework == FrontendFramework.SVELTE:
            return self._generate_svelte_component(spec)
        else:
            return (False, "", f"Framework {framework.value} not supported")
    
    def _generate_react_component(
        self,
        spec: ComponentSpec
    ) -> Tuple[bool, str, str]:
        """Generate React component code."""
        if spec.type == ComponentPattern.FUNCTIONAL:
            code = f"import React from 'react';\n\n"
            
            # Add hooks if needed
            if spec.hooks:
                code += f"import {{ {', '.join(spec.hooks)} }} from 'react';\n\n"
            
            # Component definition
            props_str = "{ " + ", ".join(spec.props) + " }" if spec.props else ""
            code += f"const {spec.name} = ({props_str}) => {{\n"
            
            # Add state hooks
            for state_var in spec.state:
                code += f"  const [{state_var}, set{state_var.capitalize()}] = useState();\n"
            
            code += "\n  return (\n    <div>\n      {/* Component content */}\n    </div>\n  );\n};\n\n"
            code += f"export default {spec.name};\n"
            
            return (True, code, "React component generated")
        
        return (False, "", "Only functional components supported")
    
    def _generate_vue_component(
        self,
        spec: ComponentSpec
    ) -> Tuple[bool, str, str]:
        """Generate Vue component code."""
        code = "<template>\n  <div>\n    <!-- Component content -->\n  </div>\n</template>\n\n"
        code += "<script setup>\n"
        
        # Add imports
        if spec.state:
            code += "import { ref } from 'vue';\n\n"
        
        # Add props
        if spec.props:
            code += "const props = defineProps({\n"
            for prop in spec.props:
                code += f"  {prop}: {{ type: String, required: true }},\n"
            code += "});\n\n"
        
        # Add state
        for state_var in spec.state:
            code += f"const {state_var} = ref();\n"
        
        code += "</script>\n\n"
        code += "<style scoped>\n/* Component styles */\n</style>\n"
        
        return (True, code, "Vue component generated")
    
    def _generate_svelte_component(
        self,
        spec: ComponentSpec
    ) -> Tuple[bool, str, str]:
        """Generate Svelte component code."""
        code = "<script>\n"
        
        # Add props
        for prop in spec.props:
            code += f"  export let {prop};\n"
        
        # Add state
        for state_var in spec.state:
            code += f"  let {state_var};\n"
        
        code += "</script>\n\n"
        code += "<div>\n  <!-- Component content -->\n</div>\n\n"
        code += "<style>\n  /* Component styles */\n</style>\n"
        
        return (True, code, "Svelte component generated")
    
    def estimate_bundle_size(
        self,
        framework: FrontendFramework,
        num_components: int,
        state_management: Optional[StateManagement] = None,
        additional_libs: List[str] = None
    ) -> Tuple[bool, int, str]:
        """
        Estimate bundle size in KB.
        
        Args:
            framework: Frontend framework
            num_components: Number of components
            state_management: State management library
            additional_libs: Additional libraries
        
        Returns:
            Tuple of (success, size_kb, message)
        """
        # Base framework sizes (minified + gzipped)
        framework_sizes = {
            FrontendFramework.REACT: 45,
            FrontendFramework.VUE: 35,
            FrontendFramework.ANGULAR: 150,
            FrontendFramework.SVELTE: 2,
            FrontendFramework.PREACT: 4,
            FrontendFramework.SOLID: 7
        }
        
        size = framework_sizes.get(framework, 50)
        
        # Add component overhead (avg 2KB per component)
        size += num_components * 2
        
        # Add state management
        if state_management:
            state_sizes = {
                StateManagement.REDUX: 15,
                StateManagement.MOBX: 20,
                StateManagement.VUEX: 10,
                StateManagement.PINIA: 5,
                StateManagement.RECOIL: 25,
                StateManagement.ZUSTAND: 3,
                StateManagement.JOTAI: 5
            }
            size += state_sizes.get(state_management, 10)
        
        # Add additional libraries (estimate 10KB each)
        if additional_libs:
            size += len(additional_libs) * 10
        
        return (True, size, f"Estimated bundle size: {size}KB")
    
    def get_performance_recommendations(
        self,
        num_components: int,
        has_large_lists: bool = False,
        has_forms: bool = False
    ) -> List[str]:
        """
        Get performance recommendations.
        
        Args:
            num_components: Number of components
            has_large_lists: Has large lists/tables
            has_forms: Has forms
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if num_components > 50:
            recommendations.append("Implement code splitting")
            recommendations.append("Use lazy loading for routes")
        
        if has_large_lists:
            recommendations.append("Implement virtual scrolling")
            recommendations.append("Use pagination or infinite scroll")
            recommendations.append("Memoize list items")
        
        if has_forms:
            recommendations.append("Debounce form validation")
            recommendations.append("Use controlled components carefully")
            recommendations.append("Consider uncontrolled components for large forms")
        
        # Always recommend
        recommendations.append("Optimize images (WebP, lazy loading)")
        recommendations.append("Minimize re-renders with memoization")
        recommendations.append("Use production builds")
        
        return recommendations
    
    def validate_accessibility(
        self,
        component_html: str
    ) -> Tuple[bool, List[str], str]:
        """
        Validate accessibility of component HTML.
        
        Args:
            component_html: Component HTML string
        
        Returns:
            Tuple of (is_valid, violations list, message)
        """
        violations = []
        
        # Check for images without alt
        if "<img" in component_html and 'alt=' not in component_html:
            violations.append("Images missing alt attribute")
        
        # Check for buttons without text
        if "<button>" in component_html or "<button " in component_html:
            if not any(text in component_html for text in ["aria-label=", "aria-labelledby="]):
                # This is a simplified check
                pass
        
        # Check for form inputs without labels
        if "<input" in component_html:
            if "<label" not in component_html and "aria-label=" not in component_html:
                violations.append("Form inputs missing labels")
        
        if violations:
            return (False, violations, f"Found {len(violations)} accessibility violations")
        
        return (True, [], "Accessibility validation passed")
    
    def get_routing_config(
        self,
        framework: FrontendFramework,
        num_routes: int
    ) -> Tuple[bool, RoutingConfig, str]:
        """
        Get recommended routing configuration.
        
        Args:
            framework: Frontend framework
            num_routes: Number of routes
        
        Returns:
            Tuple of (success, config, message)
        """
        if framework == FrontendFramework.REACT:
            router_type = "history"
        elif framework == FrontendFramework.VUE:
            router_type = "history"
        elif framework == FrontendFramework.ANGULAR:
            router_type = "history"
        else:
            router_type = "hash"
        
        config = RoutingConfig(
            router_type=router_type,
            lazy_loading=(num_routes > 10),
            guards=["auth_guard"] if num_routes > 5 else []
        )
        
        return (True, config, "Routing configuration generated")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frontend knowledge statistics."""
        return {
            "total_patterns": len(self.component_patterns),
            "performance_patterns": len(self.performance_patterns),
            "accessibility_rules": len(self.accessibility_rules),
            "supported_frameworks": len(FrontendFramework),
            "state_management_options": len(StateManagement)
        }


__all__ = [
    # Enums
    "FrontendFramework",
    "ComponentPattern",
    "StateManagement",
    "BuildTool",
    # Data classes
    "ComponentSpec",
    "RoutingConfig",
    # Main class
    "FrontendKnowledge",
]
