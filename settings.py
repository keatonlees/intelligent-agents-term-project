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
DEMO_STEPS = 100
DEFAULT_SELECTED_AGENTS = (
	{"policy": "q_learning", "agent_id": "q1", "display_char": "Q"},
	{"policy": "greedy", "agent_id": "g1", "display_char": "G"},
)

# Terminal animation configuration
ANIMATE_TERMINAL = True
FRAME_DELAY_SECONDS = 0.15
CLEAR_SCREEN_EACH_FRAME = True

# Runtime mode configuration
# Change only this variable for normal usage.
RUN_MODE = "eval"  # valid values: "train", "eval"

# Presets that fully define behavior for each mode.
MODE_PRESETS = {
	"train": {
		"episodes": 1000,
		"show_frames": False,
		"animate_terminal": False,
		"q_epsilon_start": 0.35,
		"q_epsilon_min": 0.05,
		"q_epsilon_decay": 0.995,
		"enable_q_updates": True,
		"save_q_table": True,
		"enable_q_reward_shaping": True,
	},
	"eval": {
		"episodes": 1,
		"show_frames": True,
		"animate_terminal": True,
		"q_epsilon_start": 0.01,
		"q_epsilon_min": 0.01,
		"q_epsilon_decay": 1.0,
		"enable_q_updates": False,
		"save_q_table": False,
		"enable_q_reward_shaping": False,
	},
}

# Distance-based reward shaping for Q-learning
Q_FOOD_DISTANCE_REWARD_SCALE = 0.4
Q_TRAP_DISTANCE_REWARD_SCALE = 0.2
