from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Any, Dict, List, Optional, Tuple

from settings import (
    DEFAULT_AGENT_IDS,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    DEFAULT_INITIAL_FOOD,
    DEFAULT_INITIAL_HEALTH,
    DEFAULT_INITIAL_TRAPS,
    DEFAULT_MAX_STEPS,
    FOOD_REWARD,
    STEP_PENALTY,
    TRAP_DAMAGE,
)


Position = Tuple[int, int]


class CellType(Enum):
    """Types of cells in the grid."""

    EMPTY = 0
    FOOD = 1
    TRAP = 2


class Action(Enum):
    """Actions an agent can take each step."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"


@dataclass
class Agent:
    """Basic agent representation used by the environment."""

    id: str
    x: int
    y: int
    health: int = 100
    score: int = 0

    @property
    def position(self) -> Position:
        """Return the current `(x, y)` position."""
        return (self.x, self.y)

    @position.setter
    def position(self, value: Position) -> None:
        """Update the current `(x, y)` position."""
        self.x, self.y = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state for environment output."""
        return {
            "id": self.id,
            "position": {"x": self.x, "y": self.y},
            "health": self.health,
            "score": self.score,
        }


class GridWorld:
    """Grid-world environment for movement, food collection, and traps."""

    FOOD_REWARD = FOOD_REWARD
    TRAP_DAMAGE = TRAP_DAMAGE
    STEP_PENALTY = STEP_PENALTY

    def __init__(
        self,
        width: int = DEFAULT_GRID_WIDTH,
        height: int = DEFAULT_GRID_HEIGHT,
        max_steps: int = DEFAULT_MAX_STEPS,
        agent_ids: Optional[List[str]] = None,
        initial_food: int = DEFAULT_INITIAL_FOOD,
        initial_traps: int = DEFAULT_INITIAL_TRAPS,
        initial_health: int = DEFAULT_INITIAL_HEALTH,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize environment configuration and create initial state."""
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.initial_food = initial_food
        self.initial_traps = initial_traps
        self.initial_health = initial_health
        self._rng = random.Random(seed)

        self.agent_ids = agent_ids if agent_ids else list(DEFAULT_AGENT_IDS)
        self.grid: List[List[CellType]] = []
        self.agents: Dict[str, Agent] = {}
        self.agent_positions: List[Position] = []
        self.step_count = 0

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset step count, clear grid, place agents, and spawn food/traps."""
        self.grid = [
            [CellType.EMPTY for _ in range(self.width)] for _ in range(self.height)
        ]
        self.agents = {}
        self.step_count = 0

        for agent_id in self.agent_ids:
            x, y = self.random_empty_cell()
            self.agents[agent_id] = Agent(
                id=agent_id,
                x=x,
                y=y,
                health=self.initial_health,
                score=0,
            )

        self._refresh_agent_positions()

        for _ in range(self.initial_food):
            self.spawn_food()
        for _ in range(self.initial_traps):
            self.spawn_trap()

        return self.get_state()

    def step(self, actions: Dict[str, Action | str]) -> Dict[str, Any]:
        """Advance the simulation by one step using `agent_id -> action` inputs."""
        self.step_count += 1
        events: List[Dict[str, Any]] = []

        current_positions = {
            agent_id: agent.position for agent_id, agent in self.agents.items()
        }
        proposed_moves = self._propose_moves(actions, current_positions)
        resolved_positions = self._resolve_moves(current_positions, proposed_moves)

        for agent_id, new_pos in resolved_positions.items():
            self.agents[agent_id].position = new_pos

        self._refresh_agent_positions()

        for agent_id, agent in self.agents.items():
            agent.score += self.STEP_PENALTY
            x, y = agent.position
            cell = self.get_cell(x, y)

            if cell == CellType.FOOD:
                agent.score += self.FOOD_REWARD
                self.set_cell(x, y, CellType.EMPTY)
                self.spawn_food()
                events.append(
                    {
                        "agent_id": agent_id,
                        "event": "food_collected",
                        "position": {"x": x, "y": y},
                        "score_change": self.FOOD_REWARD,
                    }
                )

            elif cell == CellType.TRAP:
                agent.health -= self.TRAP_DAMAGE
                self.set_cell(x, y, CellType.EMPTY)
                self.spawn_trap()
                events.append(
                    {
                        "agent_id": agent_id,
                        "event": "trap_triggered",
                        "position": {"x": x, "y": y},
                        "health_change": -self.TRAP_DAMAGE,
                    }
                )

        done = self.step_count >= self.max_steps

        return self.get_state(events=events, done=done)

    def spawn_food(self) -> bool:
        """Spawn one food item on a random empty cell."""
        pos = self._safe_random_empty_cell()
        if pos is None:
            return False
        x, y = pos
        self.set_cell(x, y, CellType.FOOD)
        return True

    def spawn_trap(self) -> bool:
        """Spawn one trap item on a random empty cell."""
        pos = self._safe_random_empty_cell()
        if pos is None:
            return False
        x, y = pos
        self.set_cell(x, y, CellType.TRAP)
        return True

    def is_valid_position(self, x: int, y: int) -> bool:
        """Return whether `(x, y)` is inside grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> CellType:
        """Get the cell type at `(x, y)`."""
        if not self.is_valid_position(x, y):
            raise ValueError(f"Invalid position: ({x}, {y})")
        return self.grid[y][x]

    def set_cell(self, x: int, y: int, cell_type: CellType) -> None:
        """Set the cell type at `(x, y)` to `cell_type`."""
        if not self.is_valid_position(x, y):
            raise ValueError(f"Invalid position: ({x}, {y})")
        self.grid[y][x] = cell_type

    def random_empty_cell(self) -> Position:
        """Return a random empty cell not occupied by any agent."""
        empty_positions: List[Position] = []
        occupied = set(self.agent_positions)

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in occupied:
                    continue
                if self.grid[y][x] == CellType.EMPTY:
                    empty_positions.append((x, y))

        if not empty_positions:
            raise RuntimeError("No empty cells available.")

        return self._rng.choice(empty_positions)

    @staticmethod
    def manhattan_distance(a: Position, b: Position) -> int:
        """Compute Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def render_grid(self) -> str:
        """Return an ASCII grid with coordinates and legend for debugging."""
        agent_by_pos = {agent.position: agent.id for agent in self.agents.values()}

        def cell_symbol(x: int, y: int) -> str:
            pos = (x, y)
            if pos in agent_by_pos:
                return "A"

            cell = self.grid[y][x]
            if cell == CellType.EMPTY:
                return "."
            if cell == CellType.FOOD:
                return "F"
            return "T"

        cell_width = 2
        header_cells = " ".join(f"{x:^{cell_width}}" for x in range(self.width)).rstrip()
        header = "     " + header_cells
        sample_row = " ".join(f"{'.':^{cell_width}}" for _ in range(self.width)).rstrip()
        border = "   +" + "-" * (len(sample_row) + 2) + "+"
        lines: List[str] = [header, border]

        for y in range(self.height - 1, -1, -1):
            row_cells = " ".join(
                f"{cell_symbol(x, y):^{cell_width}}" for x in range(self.width)
            ).rstrip()
            lines.append(f"{y:>2} | {row_cells} |")

        lines.append(border)

        agent_entries = ", ".join(
            f"{agent.id}@({agent.x},{agent.y}) H={agent.health} S={agent.score}"
            for agent in self.agents.values()
        )
        lines.append("Legend: A=Agent F=Food T=Trap .=Empty")
        lines.append(f"Agents: {agent_entries if agent_entries else 'None'}")
        return "\n".join(lines)

    def print_grid(self) -> None:
        """Print a text-based grid for quick debugging."""
        print(self.render_grid())

    def get_state(
        self,
        *,
        events: Optional[List[Dict[str, Any]]] = None,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Return environment state including positions, scores, and health."""
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "done": done,
            "agents": {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            "agent_positions": [
                {"id": agent_id, "x": agent.x, "y": agent.y}
                for agent_id, agent in self.agents.items()
            ],
            "events": events or [],
        }

    def _safe_random_empty_cell(self) -> Optional[Position]:
        """Return random empty cell or `None` when no empty cell exists."""
        try:
            return self.random_empty_cell()
        except RuntimeError:
            return None

    def _refresh_agent_positions(self) -> None:
        """Update cached list of agent positions."""
        self.agent_positions = [agent.position for agent in self.agents.values()]

    def _action_to_delta(self, action: Action | str) -> Position:
        """Map an action to `(dx, dy)` movement delta."""
        normalized = action
        if isinstance(action, str):
            normalized = Action[action.upper()]

        if normalized == Action.UP:
            return (0, 1)
        if normalized == Action.DOWN:
            return (0, -1)
        if normalized == Action.LEFT:
            return (-1, 0)
        if normalized == Action.RIGHT:
            return (1, 0)
        return (0, 0)

    def _propose_moves(
        self,
        actions: Dict[str, Action | str],
        current_positions: Dict[str, Position],
    ) -> Dict[str, Position]:
        """Build proposed target positions for each agent."""
        proposed: Dict[str, Position] = {}

        for agent_id, current in current_positions.items():
            action = actions.get(agent_id, Action.STAY)
            dx, dy = self._action_to_delta(action)
            target = (current[0] + dx, current[1] + dy)

            if not self.is_valid_position(target[0], target[1]):
                proposed[agent_id] = current
            else:
                proposed[agent_id] = target

        return proposed

    def _resolve_moves(
        self,
        current_positions: Dict[str, Position],
        proposed_moves: Dict[str, Position],
    ) -> Dict[str, Position]:
        """Resolve conflicts and return final agent positions."""
        targets: Dict[Position, List[str]] = {}
        for agent_id, target in proposed_moves.items():
            targets.setdefault(target, []).append(agent_id)

        winners: Dict[str, Position] = {}
        blocked: set[str] = set()

        for target, ids in targets.items():
            if len(ids) == 1:
                winners[ids[0]] = target
            else:
                winner = self._rng.choice(ids)
                winners[winner] = target
                for agent_id in ids:
                    if agent_id != winner:
                        blocked.add(agent_id)

        tentative_moves = {
            agent_id: target
            for agent_id, target in winners.items()
            if target != current_positions[agent_id]
        }

        changed = True
        while changed:
            changed = False
            non_movers = set(current_positions.keys()) - set(tentative_moves.keys())
            to_block: List[str] = []

            for moving_id, target in tentative_moves.items():
                for still_id in non_movers:
                    if current_positions[still_id] == target:
                        to_block.append(moving_id)
                        break

            if to_block:
                changed = True
                for moving_id in to_block:
                    tentative_moves.pop(moving_id, None)
                    blocked.add(moving_id)

        final_positions = dict(current_positions)
        for moving_id, target in tentative_moves.items():
            final_positions[moving_id] = target

        return final_positions


if __name__ == "__main__":
    env = GridWorld(width=10, height=10, max_steps=20, agent_ids=["a1", "a2"], seed=7)
    print("Initial state:")
    env.print_grid()
    print(env.get_state())

    demo_actions: List[Dict[str, Action]] = [
        {"a1": Action.RIGHT, "a2": Action.UP},
        {"a1": Action.UP, "a2": Action.RIGHT},
        {"a1": Action.UP, "a2": Action.LEFT},
        {"a1": Action.STAY, "a2": Action.DOWN},
    ]

    for i, actions in enumerate(demo_actions, start=1):
        print(f"\nStep {i} actions: {actions}")
        state = env.step(actions)
        env.print_grid()
        print(state)
