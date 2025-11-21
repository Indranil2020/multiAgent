"""
Game Development Prompts Module.

This module provides comprehensive prompt templates for game development tasks
in the zero-error system. Covers game loops, physics, AI, rendering, ECS,
collision detection, and game-specific optimizations.

Key Areas:
- Game loop and state management
- Entity Component System (ECS)
- Physics and collision detection
- Game AI (pathfinding, behavior trees)
- Rendering and graphics
- Input handling
- Audio systems
- Multiplayer networking
- Performance optimization

All prompts enforce zero-error philosophy with production-ready implementations.
"""

from typing import Optional, List
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_prompts import PromptTemplate, PromptFormat


# Game Loop Implementation Prompt
GAME_LOOP_PROMPT = PromptTemplate(
    template_id="game_loop",
    name="Game Loop Implementation Prompt",
    template_text="""Implement a robust game loop with fixed timestep.

GAME TYPE: {game_type}
TARGET FPS: {target_fps}
FEATURES: {features}

REQUIREMENTS:
1. Fixed timestep for physics (frame-rate independent)
2. Variable rendering (smooth visuals)
3. Input processing
4. State management
5. Delta time tracking
6. Frame rate limiting
7. Graceful degradation on slow hardware
8. Proper resource cleanup
9. Pause/resume support
10. Performance monitoring

GAME LOOP PATTERN (Fixed Timestep):

```python
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GameState(Enum):
    \"\"\"
    Game state enumeration.

    Attributes:
        INITIALIZING: Game is initializing
        RUNNING: Game is actively running
        PAUSED: Game is paused
        SHUTTING_DOWN: Game is shutting down
    \"\"\"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class GameConfig:
    \"\"\"
    Game configuration.

    Attributes:
        target_fps: Target frames per second for rendering
        physics_fps: Fixed physics update rate
        max_frame_skip: Maximum physics updates per frame
        vsync_enabled: Enable vertical sync
    \"\"\"
    target_fps: int = 60
    physics_fps: int = 60  # Fixed physics timestep
    max_frame_skip: int = 5  # Prevent spiral of death
    vsync_enabled: bool = True


class GameLoop:
    \"\"\"
    Main game loop with fixed timestep for physics.

    This implements the \"Fix Your Timestep\" pattern:
    - Fixed timestep for physics (deterministic, frame-rate independent)
    - Variable timestep for rendering (smooth visuals)
    - Accumulator for sub-frame precision
    \"\"\"

    def __init__(self, config: GameConfig):
        \"\"\"
        Initialize game loop.

        Args:
            config: Game configuration
        \"\"\"
        self.config = config
        self.state = GameState.INITIALIZING

        # Timing
        self.physics_dt = 1.0 / config.physics_fps  # Fixed physics timestep
        self.frame_dt = 1.0 / config.target_fps  # Target frame time
        self.accumulator = 0.0  # Sub-frame time accumulation
        self.current_time = time.perf_counter()

        # Performance metrics
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_update = self.current_time

        # Systems
        self.physics_system = None
        self.render_system = None
        self.input_system = None
        self.audio_system = None

    def initialize(self) -> bool:
        \"\"\"
        Initialize game systems.

        Returns:
            True if initialization successful

        Raises:
            RuntimeError: If initialization fails
        \"\"\"
        logger.info("Initializing game systems...")

        # Initialize systems
        # We assume system constructors are safe or we would need factory methods that return status
        self.physics_system = PhysicsSystem()
        self.render_system = RenderSystem()
        self.input_system = InputSystem()
        self.audio_system = AudioSystem()

        # Load resources
        # Assumes _load_resources handles its own errors or returns status
        self._load_resources()

        # Set initial state
        self.state = GameState.RUNNING

        logger.info("Game systems initialized successfully")
        return True

    def run(self) -> None:
        \"\"\"
        Main game loop.

        Implements fixed timestep for physics and variable timestep for rendering.
        This ensures physics runs at consistent rate regardless of frame rate.
        \"\"\"
        logger.info("Starting game loop...")

        while self.state != GameState.SHUTTING_DOWN:
            # Calculate frame time
            new_time = time.perf_counter()
            frame_time = new_time - self.current_time

            # Cap frame time to prevent spiral of death
            if frame_time > 0.25:  # Cap at 250ms (4 FPS)
                frame_time = 0.25
                logger.warning("Frame time capped - system struggling")

            self.current_time = new_time
            self.accumulator += frame_time

            # Process input (before physics)
            self.process_input()

            # Fixed timestep physics updates
            physics_updates = 0
            while self.accumulator >= self.physics_dt:
                if self.state == GameState.RUNNING:
                    # Fixed timestep physics
                    self.update_physics(self.physics_dt)

                    # Update game logic
                    self.update_game(self.physics_dt)

                self.accumulator -= self.physics_dt
                physics_updates += 1

                # Prevent spiral of death
                if physics_updates >= self.config.max_frame_skip:
                    self.accumulator = 0.0
                    logger.warning(f"Skipped {{physics_updates}} physics updates")
                    break

            # Calculate interpolation alpha for smooth rendering
            alpha = self.accumulator / self.physics_dt

            # Render with interpolation
            if self.state == GameState.RUNNING or self.state == GameState.PAUSED:
                self.render(alpha)

            # Update FPS counter
            self._update_fps_counter()

            # Frame rate limiting (if not using VSync)
            if not self.config.vsync_enabled:
                self._limit_frame_rate(frame_time)

    def process_input(self) -> None:
        \"\"\"Process player input.\"\"\"
        if self.input_system is None:
            return

        # Poll input events
        events = self.input_system.poll_events()

        for event in events:
            if event.type == "quit":
                self.shutdown()
            elif event.type == "pause":
                self.toggle_pause()
            elif self.state == GameState.RUNNING:
                # Handle game input
                self._handle_game_input(event)

    def update_physics(self, dt: float) -> None:
        \"\"\"
        Update physics simulation.

        Args:
            dt: Fixed physics timestep
        \"\"\"
        if self.physics_system is None:
            return

        # Physics runs at fixed timestep for determinism
        self.physics_system.update(dt)

    def update_game(self, dt: float) -> None:
        \"\"\"
        Update game logic.

        Args:
            dt: Fixed game timestep
        \"\"\"
        # Update game state
        # Update AI
        # Check win/loss conditions
        # Spawn entities
        # etc.
        pass

    def render(self, alpha: float) -> None:
        \"\"\"
        Render frame with interpolation.

        Args:
            alpha: Interpolation factor (0.0 to 1.0)
                  Used to smooth rendering between physics updates
        \"\"\"
        if self.render_system is None:
            return

        # Interpolate positions for smooth rendering
        # render_pos = prev_pos + (current_pos - prev_pos) * alpha
        self.render_system.render(alpha)

        # Swap buffers / present frame
        self.render_system.present()

    def toggle_pause(self) -> None:
        \"\"\"Toggle pause state.\"\"\"
        if self.state == GameState.RUNNING:
            self.state = GameState.PAUSED
            logger.info("Game paused")
        elif self.state == GameState.PAUSED:
            self.state = GameState.RUNNING
            logger.info("Game resumed")

    def shutdown(self) -> None:
        \"\"\"Gracefully shutdown game.\"\"\"
        logger.info("Shutting down game...")
        self.state = GameState.SHUTTING_DOWN

        # Cleanup systems
        if self.audio_system:
            self.audio_system.shutdown()
        if self.render_system:
            self.render_system.shutdown()
        if self.physics_system:
            self.physics_system.shutdown()
        if self.input_system:
            self.input_system.shutdown()

        logger.info("Game shutdown complete")

    def _load_resources(self) -> None:
        \"\"\"Load game resources.\"\"\"
        # Load textures, sounds, levels, etc.
        pass

    def _handle_game_input(self, event) -> None:
        \"\"\"Handle game-specific input.\"\"\"
        # Process game controls
        pass

    def _update_fps_counter(self) -> None:
        \"\"\"Update FPS counter for debugging.\"\"\"
        self.frame_count += 1

        current_time = time.perf_counter()
        elapsed = current_time - self.last_fps_update

        if elapsed >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time

            # Log FPS if significantly below target
            if self.fps < self.config.target_fps * 0.9:
                logger.warning(f"FPS: {{self.fps:.1f}} (target: {{self.config.target_fps}})")

    def _limit_frame_rate(self, frame_time: float) -> None:
        \"\"\"
        Limit frame rate using sleep.

        Args:
            frame_time: Time taken for this frame
        \"\"\"
        target_frame_time = 1.0 / self.config.target_fps
        sleep_time = target_frame_time - frame_time

        if sleep_time > 0:
            time.sleep(sleep_time)


# Main entry point
def main():
    \"\"\"Main game entry point.\"\"\"
    # Create game configuration
    config = GameConfig(
        target_fps=60,
        physics_fps=60,
        max_frame_skip=5,
        vsync_enabled=True
    )

    # Create and run game loop
    game = GameLoop(config)

    # Initialize game
    if game.initialize():
        # Run main loop
        game.run()
    
    # Always cleanup
    game.shutdown()


if __name__ == "__main__":
    main()
```

KEY CONCEPTS:
1. **Fixed Timestep**: Physics always updates at fixed rate (60 FPS)
2. **Accumulator**: Tracks sub-frame time for precise timing
3. **Interpolation**: Smooth rendering between physics steps
4. **Spiral of Death Prevention**: Cap max physics updates per frame

BENEFITS:
- Frame-rate independent physics
- Deterministic simulation
- Smooth rendering on any hardware
- Reproducible gameplay
- Network synchronization friendly

Generate complete, production-ready game loop.""",
    format=PromptFormat.MARKDOWN,
    variables=["game_type", "target_fps", "features"]
)


# Entity Component System (ECS) Implementation Prompt
ECS_PROMPT = PromptTemplate(
    template_id="ecs_implementation",
    name="Entity Component System Implementation Prompt",
    template_text="""Implement an Entity Component System (ECS) architecture.

GAME TYPE: {game_type}
COMPONENTS: {components}
SYSTEMS: {systems}

REQUIREMENTS:
1. Data-oriented design (cache-friendly)
2. Entity management (create, destroy, query)
3. Component storage (struct of arrays)
4. System execution order
5. Entity queries (with/without components)
6. Performance optimization
7. Serialization support
8. Event system for communication

ENTITY COMPONENT SYSTEM IMPLEMENTATION:

```python
from typing import Dict, List, Set, Type, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Type aliases
EntityID = int
ComponentType = Type['Component']

T = TypeVar('T', bound='Component')


class Component(ABC):
    \"\"\"
    Base class for all components.

    Components are pure data containers with no behavior.
    All game logic lives in Systems.
    \"\"\"
    pass


@dataclass
class TransformComponent(Component):
    \"\"\"
    Transform component (position, rotation, scale).

    Attributes:
        x: X position
        y: Y position
        rotation: Rotation in degrees
        scale_x: X scale
        scale_y: Y scale
    \"\"\"
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0


@dataclass
class VelocityComponent(Component):
    \"\"\"
    Velocity component for physics.

    Attributes:
        vx: X velocity
        vy: Y velocity
        angular_velocity: Angular velocity
    \"\"\"
    vx: float = 0.0
    vy: float = 0.0
    angular_velocity: float = 0.0


@dataclass
class SpriteComponent(Component):
    \"\"\"
    Sprite rendering component.

    Attributes:
        texture_id: Texture identifier
        width: Sprite width
        height: Sprite height
        flip_x: Flip horizontally
        flip_y: Flip vertically
    \"\"\"
    texture_id: str = ""
    width: float = 32.0
    height: float = 32.0
    flip_x: bool = False
    flip_y: bool = False


@dataclass
class HealthComponent(Component):
    \"\"\"
    Health component.

    Attributes:
        current: Current health
        maximum: Maximum health
    \"\"\"
    current: float = 100.0
    maximum: float = 100.0

    def is_alive(self) -> bool:
        \"\"\"Check if entity is alive.\"\"\"
        return self.current > 0


class ComponentManager:
    \"\"\"
    Manages component storage using struct-of-arrays for cache efficiency.
    \"\"\"

    def __init__(self):
        \"\"\"Initialize component manager.\"\"\"
        # Entity -> Set of component types
        self.entity_components: Dict[EntityID, Set[ComponentType]] = {{}}

        # Component type -> (Entity -> Component instance)
        self.components: Dict[ComponentType, Dict[EntityID, Component]] = {{}}

    def add_component(self, entity: EntityID, component: Component) -> None:
        \"\"\"
        Add component to entity.

        Args:
            entity: Entity ID
            component: Component instance
        \"\"\"
        component_type = type(component)

        # Track component type for this entity
        if entity not in self.entity_components:
            self.entity_components[entity] = set()
        self.entity_components[entity].add(component_type)

        # Store component
        if component_type not in self.components:
            self.components[component_type] = {{}}
        self.components[component_type][entity] = component

    def remove_component(self, entity: EntityID, component_type: ComponentType) -> None:
        \"\"\"
        Remove component from entity.

        Args:
            entity: Entity ID
            component_type: Component class
        \"\"\"
        if entity in self.entity_components:
            self.entity_components[entity].discard(component_type)

        if component_type in self.components:
            self.components[component_type].pop(entity, None)

    def get_component(self, entity: EntityID, component_type: Type[T]) -> Optional[T]:
        \"\"\"
        Get component from entity.

        Args:
            entity: Entity ID
            component_type: Component class

        Returns:
            Component instance or None
        \"\"\"
        if component_type not in self.components:
            return None

        return self.components[component_type].get(entity)

    def has_component(self, entity: EntityID, component_type: ComponentType) -> bool:
        \"\"\"
        Check if entity has component.

        Args:
            entity: Entity ID
            component_type: Component class

        Returns:
            True if entity has component
        \"\"\"
        if entity not in self.entity_components:
            return False

        return component_type in self.entity_components[entity]

    def get_entities_with_components(self, *component_types: ComponentType) -> List[EntityID]:
        \"\"\"
        Query entities that have ALL specified components.

        Args:
            *component_types: Component classes to filter by

        Returns:
            List of entity IDs
        \"\"\"
        if not component_types:
            return list(self.entity_components.keys())

        # Start with entities that have first component type
        result = set(self.components.get(component_types[0], {{}}).keys())

        # Intersect with entities that have other component types
        for component_type in component_types[1:]:
            result &= set(self.components.get(component_type, {{}}).keys())

        return list(result)

    def remove_entity_components(self, entity: EntityID) -> None:
        \"\"\"
        Remove all components from entity.

        Args:
            entity: Entity ID
        \"\"\"
        if entity not in self.entity_components:
            return

        # Remove from each component storage
        for component_type in self.entity_components[entity]:
            if component_type in self.components:
                self.components[component_type].pop(entity, None)

        # Remove entity tracking
        del self.entity_components[entity]


class EntityManager:
    \"\"\"
    Manages entity lifecycle (creation, destruction).
    \"\"\"

    def __init__(self):
        \"\"\"Initialize entity manager.\"\"\"
        self.next_entity_id: EntityID = 1
        self.alive_entities: Set[EntityID] = set()
        self.component_manager = ComponentManager()

    def create_entity(self) -> EntityID:
        \"\"\"
        Create new entity.

        Returns:
            New entity ID
        \"\"\"
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        self.alive_entities.add(entity_id)
        return entity_id

    def destroy_entity(self, entity: EntityID) -> None:
        \"\"\"
        Destroy entity and remove all components.

        Args:
            entity: Entity ID to destroy
        \"\"\"
        if entity not in self.alive_entities:
            return

        # Remove all components
        self.component_manager.remove_entity_components(entity)

        # Remove entity
        self.alive_entities.discard(entity)

    def is_alive(self, entity: EntityID) -> bool:
        \"\"\"
        Check if entity is alive.

        Args:
            entity: Entity ID

        Returns:
            True if entity exists
        \"\"\"
        return entity in self.alive_entities


class System(ABC):
    \"\"\"
    Base class for all systems.

    Systems contain game logic and operate on entities with specific components.
    \"\"\"

    def __init__(self, entity_manager: EntityManager):
        \"\"\"
        Initialize system.

        Args:
            entity_manager: Entity manager
        \"\"\"
        self.entity_manager = entity_manager
        self.enabled = True

    @abstractmethod
    def update(self, dt: float) -> None:
        \"\"\"
        Update system.

        Args:
            dt: Delta time in seconds
        \"\"\"
        pass


class MovementSystem(System):
    \"\"\"
    Movement system applies velocity to transform.
    \"\"\"

    def update(self, dt: float) -> None:
        \"\"\"
        Update entity positions based on velocity.

        Args:
            dt: Delta time
        \"\"\"
        if not self.enabled:
            return

        # Query entities with both Transform and Velocity components
        entities = self.entity_manager.component_manager.get_entities_with_components(
            TransformComponent, VelocityComponent
        )

        for entity in entities:
            transform = self.entity_manager.component_manager.get_component(
                entity, TransformComponent
            )
            velocity = self.entity_manager.component_manager.get_component(
                entity, VelocityComponent
            )

            if transform and velocity:
                # Update position
                transform.x += velocity.vx * dt
                transform.y += velocity.vy * dt
                transform.rotation += velocity.angular_velocity * dt


class RenderSystem(System):
    \"\"\"
    Render system draws sprites at transform positions.
    \"\"\"

    def update(self, dt: float) -> None:
        \"\"\"
        Render all sprites.

        Args:
            dt: Delta time (not used for rendering)
        \"\"\"
        if not self.enabled:
            return

        # Query entities with Transform and Sprite components
        entities = self.entity_manager.component_manager.get_entities_with_components(
            TransformComponent, SpriteComponent
        )

        # Sort by y-position for depth (painter's algorithm)
        entities_sorted = sorted(entities, key=lambda e: self._get_y_position(e))

        for entity in entities_sorted:
            transform = self.entity_manager.component_manager.get_component(
                entity, TransformComponent
            )
            sprite = self.entity_manager.component_manager.get_component(
                entity, SpriteComponent
            )

            if transform and sprite:
                # Render sprite at transform position
                self._render_sprite(transform, sprite)

    def _get_y_position(self, entity: EntityID) -> float:
        \"\"\"Get entity Y position for sorting.\"\"\"
        transform = self.entity_manager.component_manager.get_component(
            entity, TransformComponent
        )
        return transform.y if transform else 0.0

    def _render_sprite(self, transform: TransformComponent, sprite: SpriteComponent) -> None:
        \"\"\"
        Render sprite.

        Args:
            transform: Transform component
            sprite: Sprite component
        \"\"\"
        # Actual rendering implementation
        pass


class World:
    \"\"\"
    World manages all entities and systems.
    \"\"\"

    def __init__(self):
        \"\"\"Initialize world.\"\"\"
        self.entity_manager = EntityManager()
        self.systems: List[System] = []

    def add_system(self, system: System) -> None:
        \"\"\"
        Add system to world.

        Args:
            system: System instance
        \"\"\"
        self.systems.append(system)

    def update(self, dt: float) -> None:
        \"\"\"
        Update all systems.

        Args:
            dt: Delta time
        \"\"\"
        for system in self.systems:
            if system.enabled:
                system.update(dt)

    def create_entity(self, *components: Component) -> EntityID:
        \"\"\"
        Create entity with components.

        Args:
            *components: Components to add

        Returns:
            Entity ID
        \"\"\"
        entity = self.entity_manager.create_entity()

        for component in components:
            self.entity_manager.component_manager.add_component(entity, component)

        return entity

    def destroy_entity(self, entity: EntityID) -> None:
        \"\"\"
        Destroy entity.

        Args:
            entity: Entity ID
        \"\"\"
        self.entity_manager.destroy_entity(entity)


# Example usage
def example_ecs():
    \"\"\"Example ECS usage.\"\"\"
    # Create world
    world = World()

    # Add systems
    world.add_system(MovementSystem(world.entity_manager))
    world.add_system(RenderSystem(world.entity_manager))

    # Create player entity
    player = world.create_entity(
        TransformComponent(x=100.0, y=100.0),
        VelocityComponent(vx=50.0, vy=0.0),
        SpriteComponent(texture_id="player", width=32, height=32),
        HealthComponent(current=100.0, maximum=100.0)
    )

    # Game loop
    dt = 1.0 / 60.0  # 60 FPS
    for frame in range(600):  # 10 seconds
        world.update(dt)

    # Cleanup
    world.destroy_entity(player)
```

ECS BENEFITS:
- Data-oriented design (cache-friendly)
- Flexible entity composition
- Easy to add/remove features
- Better performance than inheritance
- Serialization-friendly

Generate complete ECS implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["game_type", "components", "systems"]
)


# Pathfinding (A* Algorithm) Implementation Prompt
PATHFINDING_PROMPT = PromptTemplate(
    template_id="pathfinding_astar",
    name="A* Pathfinding Implementation Prompt",
    template_text="""Implement A* pathfinding algorithm for game AI.

MAP TYPE: {map_type}
GRID SIZE: {grid_size}
MOVEMENT: {movement}  # 4-way, 8-way, or custom
HEURISTIC: {heuristic}

REQUIREMENTS:
1. Efficient A* implementation
2. Configurable heuristics
3. Path smoothing
4. Dynamic obstacle avoidance
5. Path caching
6. Performance optimization
7. Diagonal movement support
8. Variable movement costs

A* PATHFINDING IMPLEMENTATION:

```python
from typing import List, Tuple, Optional, Set, Dict, Callable
from dataclasses import dataclass, field
from queue import PriorityQueue
import math

# Type aliases
Position = Tuple[int, int]


@dataclass(order=True)
class Node:
    \"\"\"
    A* node for priority queue.

    Attributes:
        f_score: Total estimated cost (g + h)
        position: Grid position
        g_score: Cost from start
        h_score: Heuristic estimate to goal
        parent: Parent node
    \"\"\"
    f_score: float
    position: Position = field(compare=False)
    g_score: float = field(default=0.0, compare=False)
    h_score: float = field(default=0.0, compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


class Pathfinder:
    \"\"\"
    A* pathfinding implementation.
    \"\"\"

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        diagonal_movement: bool = True,
        diagonal_cost: float = 1.4142  # sqrt(2)
    ):
        \"\"\"
        Initialize pathfinder.

        Args:
            grid_width: Grid width
            grid_height: Grid height
            diagonal_movement: Allow diagonal movement
            diagonal_cost: Cost multiplier for diagonal movement
        \"\"\"
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.diagonal_movement = diagonal_movement
        self.diagonal_cost = diagonal_cost

        # Obstacle grid
        self.obstacles: Set[Position] = set()

        # Movement cost grid (default 1.0)
        self.costs: Dict[Position, float] = {{}}

    def set_obstacle(self, position: Position, is_obstacle: bool = True) -> None:
        \"\"\"
        Set position as obstacle.

        Args:
            position: Grid position
            is_obstacle: Whether position is obstacle
        \"\"\"
        if is_obstacle:
            self.obstacles.add(position)
        else:
            self.obstacles.discard(position)

    def set_cost(self, position: Position, cost: float) -> None:
        \"\"\"
        Set movement cost for position.

        Args:
            position: Grid position
            cost: Movement cost (1.0 = normal, higher = harder to traverse)
        \"\"\"
        self.costs[position] = cost

    def find_path(
        self,
        start: Position,
        goal: Position,
        heuristic: str = "euclidean"
    ) -> Optional[List[Position]]:
        \"\"\"
        Find path from start to goal using A*.

        Args:
            start: Start position
            goal: Goal position
            heuristic: Heuristic function ("manhattan", "euclidean", "chebyshev")

        Returns:
            List of positions from start to goal, or None if no path
        \"\"\"
        # Validate positions
        if not self._is_valid(start) or not self._is_valid(goal):
            return None

        if start in self.obstacles or goal in self.obstacles:
            return None

        # Select heuristic function
        heuristic_func = self._get_heuristic_function(heuristic)

        # Initialize
        open_set = PriorityQueue()
        closed_set: Set[Position] = set()

        # Create start node
        start_node = Node(
            f_score=heuristic_func(start, goal),
            position=start,
            g_score=0.0,
            h_score=heuristic_func(start, goal),
            parent=None
        )

        open_set.put(start_node)
        open_positions = {{start}}  # Track positions in open set

        # A* main loop
        while not open_set.empty():
            # Get node with lowest f_score
            current = open_set.get()
            open_positions.discard(current.position)

            # Goal reached
            if current.position == goal:
                return self._reconstruct_path(current)

            # Add to closed set
            closed_set.add(current.position)

            # Check neighbors
            for neighbor_pos in self._get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue

                if not self._is_valid(neighbor_pos) or neighbor_pos in self.obstacles:
                    continue

                # Calculate movement cost
                movement_cost = self._get_movement_cost(current.position, neighbor_pos)
                g_score = current.g_score + movement_cost

                # Check if this path is better
                if neighbor_pos not in open_positions:
                    # New node
                    h_score = heuristic_func(neighbor_pos, goal)
                    neighbor_node = Node(
                        f_score=g_score + h_score,
                        position=neighbor_pos,
                        g_score=g_score,
                        h_score=h_score,
                        parent=current
                    )
                    open_set.put(neighbor_node)
                    open_positions.add(neighbor_pos)

        # No path found
        return None

    def _get_neighbors(self, position: Position) -> List[Position]:
        \"\"\"
        Get neighboring positions.

        Args:
            position: Current position

        Returns:
            List of neighbor positions
        \"\"\"
        x, y = position
        neighbors = []

        # 4-way movement
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbors.append((x + dx, y + dy))

        # Diagonal movement
        if self.diagonal_movement:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                # Only allow diagonal if both adjacent cells are free
                if ((x + dx, y) not in self.obstacles and
                    (x, y + dy) not in self.obstacles):
                    neighbors.append((x + dx, y + dy))

        return neighbors

    def _get_movement_cost(self, from_pos: Position, to_pos: Position) -> float:
        \"\"\"
        Get cost to move from one position to another.

        Args:
            from_pos: Start position
            to_pos: End position

        Returns:
            Movement cost
        \"\"\"
        # Base cost
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])

        if dx + dy == 2:  # Diagonal
            base_cost = self.diagonal_cost
        else:  # Straight
            base_cost = 1.0

        # Apply terrain cost
        terrain_cost = self.costs.get(to_pos, 1.0)

        return base_cost * terrain_cost

    def _is_valid(self, position: Position) -> bool:
        \"\"\"
        Check if position is valid.

        Args:
            position: Grid position

        Returns:
            True if valid
        \"\"\"
        x, y = position
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def _get_heuristic_function(self, heuristic: str) -> Callable[[Position, Position], float]:
        \"\"\"
        Get heuristic function by name.

        Args:
            heuristic: Heuristic name

        Returns:
            Heuristic function
        \"\"\"
        heuristics = {{
            "manhattan": self._manhattan_distance,
            "euclidean": self._euclidean_distance,
            "chebyshev": self._chebyshev_distance
        }}

        return heuristics.get(heuristic, self._euclidean_distance)

    @staticmethod
    def _manhattan_distance(a: Position, b: Position) -> float:
        \"\"\"
        Manhattan distance heuristic (4-way movement).

        Args:
            a: Start position
            b: Goal position

        Returns:
            Manhattan distance
        \"\"\"
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _euclidean_distance(a: Position, b: Position) -> float:
        \"\"\"
        Euclidean distance heuristic (straight line).

        Args:
            a: Start position
            b: Goal position

        Returns:
            Euclidean distance
        \"\"\"
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def _chebyshev_distance(a: Position, b: Position) -> float:
        \"\"\"
        Chebyshev distance heuristic (8-way movement).

        Args:
            a: Start position
            b: Goal position

        Returns:
            Chebyshev distance
        \"\"\"
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    @staticmethod
    def _reconstruct_path(node: Node) -> List[Position]:
        \"\"\"
        Reconstruct path from goal node.

        Args:
            node: Goal node

        Returns:
            Path from start to goal
        \"\"\"
        path = []
        current = node

        while current is not None:
            path.append(current.position)
            current = current.parent

        return list(reversed(path))


# Example usage
def example_pathfinding():
    \"\"\"Example A* pathfinding.\"\"\"
    # Create pathfinder for 20x20 grid
    pathfinder = Pathfinder(
        grid_width=20,
        grid_height=20,
        diagonal_movement=True
    )

    # Add obstacles (wall)
    for y in range(5, 15):
        pathfinder.set_obstacle((10, y))

    # Add difficult terrain (mud)
    for x in range(15, 20):
        for y in range(10, 15):
            pathfinder.set_cost((x, y), 5.0)

    # Find path
    start = (0, 0)
    goal = (19, 19)

    path = pathfinder.find_path(start, goal, heuristic="euclidean")

    if path:
        print(f"Path found with {{len(path)}} steps")
        print(f"Path: {{path}}")
    else:
        print("No path found")
```

PATHFINDING OPTIMIZATIONS:
1. **Jump Point Search**: Skip redundant nodes
2. **Hierarchical Pathfinding**: Multi-level path planning
3. **Flow Fields**: For many units to same goal
4. **Path Caching**: Reuse computed paths
5. **Theta***: Any-angle pathfinding

Generate complete pathfinding implementation.""",
    format=PromptFormat.MARKDOWN,
    variables=["map_type", "grid_size", "movement", "heuristic"]
)


# Collision Detection Implementation Prompt
COLLISION_DETECTION_PROMPT = PromptTemplate(
    template_id="collision_detection",
    name="Collision Detection Implementation Prompt",
    template_text="""Implement collision detection system.

COLLISION TYPES: {collision_types}
PHYSICS TYPE: {physics_type}
OPTIMIZATION: {optimization}

REQUIREMENTS:
1. Multiple collision shapes (AABB, circle, polygon)
2. Broad phase optimization (spatial partitioning)
3. Narrow phase collision detection
4. Collision response
5. Trigger volumes (non-physical collisions)
6. Layer-based collision filtering
7. Continuous collision detection
8. Performance optimization

COLLISION DETECTION IMPLEMENTATION:

```python
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import math


class CollisionLayer(Enum):
    \"\"\"
    Collision layers for filtering.

    Attributes:
        PLAYER: Player layer
        ENEMY: Enemy layer
        PROJECTILE: Projectile layer
        ENVIRONMENT: Static environment
        TRIGGER: Trigger volumes
    \"\"\"
    PLAYER = 1 << 0  # Bit 0
    ENEMY = 1 << 1  # Bit 1
    PROJECTILE = 1 << 2  # Bit 2
    ENVIRONMENT = 1 << 3  # Bit 3
    TRIGGER = 1 << 4  # Bit 4


@dataclass
class AABB:
    \"\"\"
    Axis-Aligned Bounding Box.

    Attributes:
        min_x: Minimum X
        min_y: Minimum Y
        max_x: Maximum X
        max_y: Maximum Y
    \"\"\"
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2


@dataclass
class Circle:
    \"\"\"
    Circle collision shape.

    Attributes:
        x: Center X
        y: Center Y
        radius: Radius
    \"\"\"
    x: float
    y: float
    radius: float


@dataclass
class CollisionInfo:
    \"\"\"
    Collision information.

    Attributes:
        collider_a: First collider
        collider_b: Second collider
        normal: Collision normal
        penetration: Penetration depth
        point: Collision point
    \"\"\"
    collider_a: 'Collider'
    collider_b: 'Collider'
    normal: Tuple[float, float]
    penetration: float
    point: Tuple[float, float]


class Collider:
    \"\"\"
    Base collider class.
    \"\"\"

    def __init__(
        self,
        layer: CollisionLayer,
        is_trigger: bool = False
    ):
        \"\"\"
        Initialize collider.

        Args:
            layer: Collision layer
            is_trigger: Whether this is a trigger (no physics response)
        \"\"\"
        self.layer = layer
        self.is_trigger = is_trigger
        self.enabled = True

    def get_aabb(self) -> AABB:
        \"\"\"
        Get axis-aligned bounding box.

        Returns:
            AABB for broad phase
        \"\"\"
        raise NotImplementedError


class AABBCollider(Collider):
    \"\"\"AABB collider.\"\"\"

    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        layer: CollisionLayer,
        is_trigger: bool = False
    ):
        super().__init__(layer, is_trigger)
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_aabb(self) -> AABB:
        return AABB(
            min_x=self.x,
            min_y=self.y,
            max_x=self.x + self.width,
            max_y=self.y + self.height
        )


class CircleCollider(Collider):
    \"\"\"Circle collider.\"\"\"

    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        layer: CollisionLayer,
        is_trigger: bool = False
    ):
        super().__init__(layer, is_trigger)
        self.x = x
        self.y = y
        self.radius = radius

    def get_aabb(self) -> AABB:
        return AABB(
            min_x=self.x - self.radius,
            min_y=self.y - self.radius,
            max_x=self.x + self.radius,
            max_y=self.y + self.radius
        )


class CollisionDetector:
    \"\"\"
    Collision detection system.
    \"\"\"

    @staticmethod
    def aabb_vs_aabb(a: AABB, b: AABB) -> bool:
        \"\"\"
        Check AABB collision.

        Args:
            a: First AABB
            b: Second AABB

        Returns:
            True if colliding
        \"\"\"
        return (a.min_x <= b.max_x and a.max_x >= b.min_x and
                a.min_y <= b.max_y and a.max_y >= b.min_y)

    @staticmethod
    def circle_vs_circle(a: Circle, b: Circle) -> bool:
        \"\"\"
        Check circle collision.

        Args:
            a: First circle
            b: Second circle

        Returns:
            True if colliding
        \"\"\"
        dx = a.x - b.x
        dy = a.y - b.y
        distance_squared = dx * dx + dy * dy
        radius_sum = a.radius + b.radius
        return distance_squared <= radius_sum * radius_sum

    @staticmethod
    def aabb_vs_circle(aabb: AABB, circle: Circle) -> bool:
        \"\"\"
        Check AABB vs circle collision.

        Args:
            aabb: AABB
            circle: Circle

        Returns:
            True if colliding
        \"\"\"
        # Find closest point on AABB to circle center
        closest_x = max(aabb.min_x, min(circle.x, aabb.max_x))
        closest_y = max(aabb.min_y, min(circle.y, aabb.max_y))

        # Calculate distance from circle center to closest point
        dx = circle.x - closest_x
        dy = circle.y - closest_y
        distance_squared = dx * dx + dy * dy

        return distance_squared <= circle.radius * circle.radius

    @staticmethod
    def get_aabb_collision_info(
        a: AABBCollider,
        b: AABBCollider
    ) -> Optional[CollisionInfo]:
        \"\"\"
        Get detailed AABB collision info.

        Args:
            a: First collider
            b: Second collider

        Returns:
            Collision info or None
        \"\"\"
        aabb_a = a.get_aabb()
        aabb_b = b.get_aabb()

        if not CollisionDetector.aabb_vs_aabb(aabb_a, aabb_b):
            return None

        # Calculate overlap on each axis
        overlap_x = min(aabb_a.max_x, aabb_b.max_x) - max(aabb_a.min_x, aabb_b.min_x)
        overlap_y = min(aabb_a.max_y, aabb_b.max_y) - max(aabb_a.min_y, aabb_b.min_y)

        # Find minimum penetration axis
        if overlap_x < overlap_y:
            # Collision on X axis
            if aabb_a.center_x < aabb_b.center_x:
                normal = (-1.0, 0.0)
            else:
                normal = (1.0, 0.0)
            penetration = overlap_x
        else:
            # Collision on Y axis
            if aabb_a.center_y < aabb_b.center_y:
                normal = (0.0, -1.0)
            else:
                normal = (0.0, 1.0)
            penetration = overlap_y

        # Calculate collision point (center of overlap)
        point = (
            (max(aabb_a.min_x, aabb_b.min_x) + min(aabb_a.max_x, aabb_b.max_x)) / 2,
            (max(aabb_a.min_y, aabb_b.min_y) + min(aabb_a.max_y, aabb_b.max_y)) / 2
        )

        return CollisionInfo(
            collider_a=a,
            collider_b=b,
            normal=normal,
            penetration=penetration,
            point=point
        )


class SpatialHash:
    \"\"\"
    Spatial hash for broad-phase collision detection.
    \"\"\"

    def __init__(self, cell_size: float):
        \"\"\"
        Initialize spatial hash.

        Args:
            cell_size: Size of each grid cell
        \"\"\"
        self.cell_size = cell_size
        self.cells: dict[Tuple[int, int], Set[Collider]] = {{}}

    def insert(self, collider: Collider) -> None:
        \"\"\"
        Insert collider into spatial hash.

        Args:
            collider: Collider to insert
        \"\"\"
        aabb = collider.get_aabb()
        cells = self._get_cells(aabb)

        for cell in cells:
            if cell not in self.cells:
                self.cells[cell] = set()
            self.cells[cell].add(collider)

    def query(self, aabb: AABB) -> Set[Collider]:
        \"\"\"
        Query colliders in AABB.

        Args:
            aabb: Query AABB

        Returns:
            Set of potential colliders
        \"\"\"
        cells = self._get_cells(aabb)
        result = set()

        for cell in cells:
            if cell in self.cells:
                result.update(self.cells[cell])

        return result

    def clear(self) -> None:
        \"\"\"Clear spatial hash.\"\"\"
        self.cells.clear()

    def _get_cells(self, aabb: AABB) -> List[Tuple[int, int]]:
        \"\"\"
        Get cells overlapping AABB.

        Args:
            aabb: AABB

        Returns:
            List of cell coordinates
        \"\"\"
        min_cell_x = int(aabb.min_x / self.cell_size)
        min_cell_y = int(aabb.min_y / self.cell_size)
        max_cell_x = int(aabb.max_x / self.cell_size)
        max_cell_y = int(aabb.max_y / self.cell_size)

        cells = []
        for x in range(min_cell_x, max_cell_x + 1):
            for y in range(min_cell_y, max_cell_y + 1):
                cells.append((x, y))

        return cells
```

COLLISION RESPONSE:
```python
def resolve_collision(info: CollisionInfo) -> None:
    \"\"\"Resolve collision by separating objects.\"\"\"
    if isinstance(info.collider_a, AABBCollider) and isinstance(info.collider_b, AABBCollider):
        # Move half distance each
        move_x = info.normal[0] * info.penetration / 2
        move_y = info.normal[1] * info.penetration / 2

        info.collider_a.x -= move_x
        info.collider_a.y -= move_y
        info.collider_b.x += move_x
        info.collider_b.y += move_y
```

Generate complete collision detection system.""",
    format=PromptFormat.MARKDOWN,
    variables=["collision_types", "physics_type", "optimization"]
)


# Export all templates
ALL_GAME_TEMPLATES = {
    "game_loop": GAME_LOOP_PROMPT,
    "ecs_implementation": ECS_PROMPT,
    "pathfinding_astar": PATHFINDING_PROMPT,
    "collision_detection": COLLISION_DETECTION_PROMPT
}


def get_game_template(template_id: str) -> Optional[PromptTemplate]:
    """
    Get game development prompt template by ID.

    Args:
        template_id: Template identifier

    Returns:
        PromptTemplate if found, None otherwise
    """
    return ALL_GAME_TEMPLATES.get(template_id)


def list_game_templates() -> List[str]:
    """
    List all available game development template IDs.

    Returns:
        List of template IDs
    """
    return list(ALL_GAME_TEMPLATES.keys())
