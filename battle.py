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


# what each cell in the grid contains
class CellType(Enum):
    EMPTY = 0
    FOOD = 1
    TRAP = 2


# the available actions for agents to take
class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STAY = "STAY"


# the agent state we track in the environment
@dataclass
class Agent:
    id: str
    x: int
    y: int
    health: int = 100
    score: int = 0

    @property
    def position(self) -> Position:
        return (self.x, self.y)

    @position.setter
    def position(self, value: Position) -> None:
        self.x, self.y = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": {"x": self.x, "y": self.y},
            "health": self.health,
            "score": self.score,
        }


# the main environment class that simulates the grid world and agent interactions
class GridWorld:
    FOOD_REWARD = FOOD_REWARD
    TRAP_DAMAGE = TRAP_DAMAGE
    STEP_PENALTY = STEP_PENALTY

    def __init__(
        self,
        width: int = DEFAULT_GRID_WIDTH,
        height: int = DEFAULT_GRID_HEIGHT,
        max_steps: int = DEFAULT_MAX_STEPS,
        agent_ids: Optional[List[str]] = None,
        agent_display_chars: Optional[Dict[str, str]] = None,
        initial_food: int = DEFAULT_INITIAL_FOOD,
        initial_traps: int = DEFAULT_INITIAL_TRAPS,
        initial_health: int = DEFAULT_INITIAL_HEALTH,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.initial_food = initial_food
        self.initial_traps = initial_traps
        self.initial_health = initial_health
        self._rng = random.Random(seed)

        self.agent_ids = agent_ids if agent_ids else list(DEFAULT_AGENT_IDS)
        self.agent_display_chars = agent_display_chars or {}
        self.grid: List[List[CellType]] = []
        self.agents: Dict[str, Agent] = {}
        self.agent_positions: List[Position] = []
        self.step_count = 0

        self.reset()

    # the main method to reset the environment to a fresh state
    def reset(self) -> Dict[str, Any]:
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

        self.refresh_agent_positions()

        for _ in range(self.initial_food):
            self.spawn_food()
        for _ in range(self.initial_traps):
            self.spawn_trap()

        return self.get_state()

    # run one turn of the environment
    def step(self, actions: Dict[str, Action | str]) -> Dict[str, Any]:
        self.step_count += 1
        events: List[Dict[str, Any]] = []
        rewards: Dict[str, int] = {agent_id: 0 for agent_id in self.agents.keys()}

        current_positions = {
            agent_id: agent.position for agent_id, agent in self.agents.items()
        }
        proposed_moves = self.propose_moves(actions, current_positions)
        resolved_positions = self.resolve_moves(current_positions, proposed_moves)

        for agent_id, new_pos in resolved_positions.items():
            self.agents[agent_id].position = new_pos

        self.refresh_agent_positions()

        for agent_id, agent in self.agents.items():
            # every move costs a little bit
            agent.score += self.STEP_PENALTY
            rewards[agent_id] += self.STEP_PENALTY

            x, y = agent.position
            cell = self.get_cell(x, y)

            if cell == CellType.FOOD:
                agent.score += self.FOOD_REWARD
                rewards[agent_id] += self.FOOD_REWARD
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
                rewards[agent_id] -= self.TRAP_DAMAGE
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

        # end if max steps reached or somebody dies
        done = self.step_count >= self.max_steps or any(
            agent.health <= 0 for agent in self.agents.values()
        )

        return self.get_state(events=events, done=done, rewards=rewards)

    # helper method to spawn food
    def spawn_food(self) -> bool:
        pos = self.safe_random_empty_cell()
        if pos is None:
            return False
        x, y = pos
        self.set_cell(x, y, CellType.FOOD)
        return True

    # helper method to spawn traps
    def spawn_trap(self) -> bool:
        pos = self.safe_random_empty_cell()
        if pos is None:
            return False
        x, y = pos
        self.set_cell(x, y, CellType.TRAP)
        return True

    # helper method to check if a position is within the grid bounds
    def is_valid_position(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    # helper methods to get cell
    def get_cell(self, x: int, y: int) -> CellType:
        if not self.is_valid_position(x, y):
            raise ValueError(f"Invalid position: ({x}, {y})")
        return self.grid[y][x]

    # helper method to set cell
    def set_cell(self, x: int, y: int, cell_type: CellType) -> None:
        if not self.is_valid_position(x, y):
            raise ValueError(f"Invalid position: ({x}, {y})")
        self.grid[y][x] = cell_type

    # helper method to find a random empty cell for spawning food/traps or placing agents
    def random_empty_cell(self) -> Position:
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

    # helper method to calculate Manhattan distance between two positions
    @staticmethod
    def manhattan_distance(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # helper method to render the grid as text for debugging
    def render_grid(self) -> str:
        agent_by_pos = {agent.position: agent.id for agent in self.agents.values()}

        def cell_symbol(x: int, y: int) -> str:
            pos = (x, y)
            if pos in agent_by_pos:
                agent_id = agent_by_pos[pos]
                return self.agent_display_chars.get(agent_id, "A")

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
            f"{self.agent_display_chars.get(agent.id, 'A')}:{agent.id}@({agent.x},{agent.y}) "
            f"H={agent.health} S={agent.score}"
            for agent in self.agents.values()
        )
        lines.append("Legend: [agent-char]=Agent F=Food T=Trap .=Empty")
        lines.append(f"Agents: {agent_entries if agent_entries else 'None'}")
        return "\n".join(lines)

    # helper method to print the grid to the terminal
    def print_grid(self) -> None:
        print(self.render_grid())

    # method to get the full state of the environment, including agent states and positions of food/traps
    def get_state(
        self,
        *,
        events: Optional[List[Dict[str, Any]]] = None,
        done: bool = False,
        rewards: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "done": done,
            "agents": {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
            "agent_positions": [
                {"id": agent_id, "x": agent.x, "y": agent.y}
                for agent_id, agent in self.agents.items()
            ],
            "food_positions": self.get_positions_by_cell_type(CellType.FOOD),
            "trap_positions": self.get_positions_by_cell_type(CellType.TRAP),
            "events": events or [],
            "rewards": rewards or {},
        }

    # method to give one agent a simplified view of the world for decision-making (e.g. for Q-learning)
    def get_agent_view(self, agent_id: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if state is None:
            state = self.get_state()

        me = self.agents[agent_id]
        my_pos = (me.x, me.y)

        enemy_positions = [
            [agent.x, agent.y]
            for aid, agent in self.agents.items()
            if aid != agent_id
        ]
        nearest_enemy_pos, nearest_enemy_distance = self.get_nearest_target(
            my_pos,
            enemy_positions,
        )

        nearest_food_pos, nearest_food_distance = self.get_nearest_target(
            my_pos,
            state["food_positions"],
        )
        nearest_trap_pos, nearest_trap_distance = self.get_nearest_target(
            my_pos,
            state["trap_positions"],
        )

        return {
            "self_pos": [me.x, me.y],
            "self_health": me.health,
            "self_score": me.score,
            "nearest_enemy_direction": self.direction_to_target(my_pos, nearest_enemy_pos),
            "nearest_enemy_distance": nearest_enemy_distance,
            "nearest_food_direction": self.direction_to_target(my_pos, nearest_food_pos),
            "nearest_food_distance": nearest_food_distance,
            "nearest_trap_direction": self.direction_to_target(my_pos, nearest_trap_pos),
            "nearest_trap_distance": nearest_trap_distance,
            "step": state["step"],
            "done": state["done"],
        }

    # function to find nearest target
    def get_nearest_target(
        self,
        origin: Position,
        positions: List[List[int]],
    ) -> Tuple[Optional[Position], Optional[int]]:
        if not positions:
            return None, None

        targets = [(pos[0], pos[1]) for pos in positions]
        nearest = min(
            targets,
            key=lambda target: (self.manhattan_distance(origin, target), target[1], target[0]),
        )
        return nearest, self.manhattan_distance(origin, nearest)

    # function to get direction to target
    def direction_to_target(
        self,
        origin: Position,
        target: Optional[Position],
    ) -> Optional[str]:
        if target is None:
            return None

        dx = target[0] - origin[0]
        dy = target[1] - origin[1]

        if dx == 0 and dy == 0:
            return Action.STAY.name

        # Ties break horizontally for deterministic behavior.
        if abs(dx) >= abs(dy):
            return Action.RIGHT.name if dx > 0 else Action.LEFT.name

        return Action.UP.name if dy > 0 else Action.DOWN.name

    # function to get positions by cell type
    def get_positions_by_cell_type(self, cell_type: CellType) -> List[List[int]]:
        positions: List[List[int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == cell_type:
                    positions.append([x, y])
        return positions

    # safe version of random_empty_cell that returns None instead of raising if no empty cells are available
    def safe_random_empty_cell(self) -> Optional[Position]:
        try:
            return self.random_empty_cell()
        except RuntimeError:
            return None

    # helper method to refresh the list of agent positions after they move
    def refresh_agent_positions(self) -> None:
        self.agent_positions = [agent.position for agent in self.agents.values()]

    def _action_to_delta(self, action: Action | str) -> Position:
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

    # function to propose moves for agents
    def propose_moves(
        self,
        actions: Dict[str, Action | str],
        current_positions: Dict[str, Position],
    ) -> Dict[str, Position]:
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

    # function to resolve moves for agents
    def resolve_moves(
        self,
        current_positions: Dict[str, Position],
        proposed_moves: Dict[str, Position],
    ) -> Dict[str, Position]:
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
