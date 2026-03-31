# Grid Battle Environment (MVP)

A grid-based environment for simulating and comparing intelligent agent behaviour. The system includes multiple agent policies, including a reinforcement learning agent and a rule-based heuristic agent, operating within a shared environment.

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

1. Agents cannot move outside grid bounds
2. Two agents cannot occupy the same cell
3. If multiple agents target the same cell, one winner is chosen randomly and others are blocked
4. Food and traps only spawn on empty, unoccupied cells
5. Collecting food removes that food and respawns a new one
6. Triggering a trap removes that trap and respawns a new one
7. Each step applies a score penalty
8. An episode ends when max steps are reached or any agent health reaches zero

## Project Structure

- `main.py`: Orchestrates episodes, mode behavior, rendering, and optional Q updates
- `battle.py`: Environment engine (`GridWorld`), movement resolution, rewards/events, and state output
- `settings.py`: All simulation constants, runtime mode, rendering flags, and mode presets
- `alpha_sweep.py`: Compares multiple alpha values for Q-learning experiments
- `agents/agent_random.py`: Uniform random action policy for testing
- `agents/agent_greedy.py`: Moves toward nearest food while avoiding immediate trap risk
- `agents/agent_q_learning.py`: Tabular Q-learning policy with epsilon-greedy exploration

## Agent Configuration

`main.py` uses `DEFAULT_SELECTED_AGENTS` from `settings.py` to create in-game agents

Each config entry includes:

- `policy`: key from `AGENT_REGISTRY` (`random`, `greedy`, `q_learning`)
- `agent_id`: unique ID in the world
- `display_char`: single-character marker in terminal rendering

## State And Agent View

Each environment step returns a state dictionary containing:

- `step`, `max_steps`, `done`
- `agents` (position, health, score)
- `agent_positions`
- `food_positions`
- `trap_positions`
- `events` (e.g., `food_collected`, `trap_triggered`)

### Agent View (policy input)

When `main.py` queries `get_agent_view(agent_id, state)`, each agent receives a simplified view of the environment:

- `self_pos`, `self_health`, `self_score`, `step`, `done`
- `nearest_enemy_direction` and `nearest_enemy_distance`
- `nearest_food_direction` and `nearest_food_distance`
- `nearest_trap_direction` and `nearest_trap_distance`

If no enemy exists, `nearest_enemy_direction` and `nearest_enemy_distance` are `None`.

Direction selection uses Manhattan guidance:

- Move along the larger axis delta toward the target.
- If the target is perfectly diagonal (`|dx| == |dy|`), tie-break horizontally for deterministic behavior.

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

Default setup runs one `q_learning` agent and one `greedy` agent together.
Q-learning updates each step using environment rewards and saves learned values to `q_table.json` at the end of the run.

## Runtime Modes

The system supports two modes of operation. Modes are configured in `settings.py`:

- `train`: enables learning, exploration, and Q-table updates
- `eval`: disables learning and minimizes exploration to evaluate learned behaviour


## Run

```bash
python main.py
```
