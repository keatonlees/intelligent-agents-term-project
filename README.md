# Grid Battle Environment

A lightweight multi-agent grid world for experimenting with policy behavior and Q-learning

The project supports:

- Turn-based movement on a 2D grid
- Food and trap interactions
- Multiple policy types (greedy, Q-learning)
- Evaluation mode with terminal rendering

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Run the game (eval mode)

Set in `settings.py`:

```python
RUN_MODE = "eval"
```

Then run:

```bash
python main.py
```

What you get:

- A rendered terminal grid each step
- Agent positions, health, scores, and events
- No Q-table updates during the run

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
- `food_positions`, `trap_positions`
- `events` (such as `food_collected`, `trap_triggered`)
- `rewards` (per-agent step reward)

`GridWorld.get_agent_view` returns a compact per-agent observation used by policies:

- self stats: position, health, score
- nearest enemy direction and distance
- nearest food direction and distance
- nearest trap direction and distance
- step and done flag
