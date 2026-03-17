# Grid Battle Environment (MVP)

Minimal grid-world environment for agent simulation. This project currently implements only the environment layer (no reinforcement learning, no A* pathfinding).

## Assumptions

- Grid size defaults are defined in `settings.py`.
- Multi-agent simulation is supported.
- Each class in `agents/` represents one agent policy instance.
- Multiple in-game agents are created by selecting multiple agent instances in `main.py`.
- Agents are tracked separately from grid cells.
- Cell types are: `EMPTY`, `FOOD`, `TRAP`.
- Agent actions are: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.
- Initial values:
  - Step penalty: `-1` score per step
  - Food reward: `+10` score
  - Trap effect: `-10` health
  - Agent health starts at `100`

## Core Rules

1. Agents cannot move outside grid bounds.
2. Agents cannot occupy the same cell.
3. If multiple agents target the same cell in one step, one random winner moves and others are blocked.
4. Food and traps spawn only on `EMPTY` cells not occupied by agents.
5. When food is collected, that food is removed and a new food is spawned.
6. When a trap is triggered, that trap is removed and a new trap is spawned.
7. Every step applies the step penalty to each agent.

## Step Resolution (per turn)

For each step:

1. Read `agent_id -> action` input.
2. Compute proposed positions.
3. Resolve movement conflicts and invalid moves.
4. Apply movement.
5. Apply cell effects (`FOOD`/`TRAP`) and respawn consumed/triggered items.
6. Return environment state.

## State Output

Each step returns a state dictionary containing:

- `step`, `max_steps`, `done`
- `agents` (position, health, score)
- `agent_positions`
- `events` (e.g., `food_collected`, `trap_triggered`)

## Files

- `battle.py`: environment implementation (`GridWorld`, `Agent`, enums, utilities)
- `settings.py`: centralized simulation constants and defaults
- `agents/agent_random.py`: single-agent random policy logic (uniform action selection)
- `main.py`: simulation runner/orchestrator with selectable agent-instance array

### Agent Selection

`main.py` uses `SELECTED_AGENTS` to build in-game agent instances. Each entry supports:

- `policy`: policy key in `AGENT_REGISTRY`
- `agent_id`: unique in-environment ID
- `display_char`: single character used on the rendered grid

## Run

```bash
python main.py
```
