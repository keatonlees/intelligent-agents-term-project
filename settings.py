"""Centralized configuration constants for the grid simulation."""

DEFAULT_GRID_WIDTH = 10
DEFAULT_GRID_HEIGHT = 10
DEFAULT_MAX_STEPS = 200
DEFAULT_INITIAL_FOOD = 3
DEFAULT_INITIAL_TRAPS = 3
DEFAULT_INITIAL_HEALTH = 100
DEFAULT_AGENT_IDS = ("agent_0",)

FOOD_REWARD = 10
TRAP_DAMAGE = 10
STEP_PENALTY = -1

# Simulation run configuration (used by main.py)
DEMO_STEPS = 200
DEFAULT_SELECTED_AGENTS = (
	{"policy": "q_learning", "agent_id": "q1", "display_char": "Q"},
	{"policy": "greedy", "agent_id": "g1", "display_char": "G"},
)
