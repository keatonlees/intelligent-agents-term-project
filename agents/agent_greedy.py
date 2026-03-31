import random
from typing import Any, Dict

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class GreedyAgent:
    policy_name = "agent_greedy"

    def __init__(self, agent_id: str, name: str = "greedy_agent"):
        self.agent_id = agent_id
        self.name = name

    # choose an action based on the current state, ignoring the step index
    def action_for_step(self, step_index: int, state: Dict[str, Any]) -> str:
        del step_index
        return self.choose_action(state)

    # chooses the action that moves toward the nearest food, while avoiding traps when possible
    def choose_action(self, state):
        food_direction = state.get("nearest_food_direction")
        food_distance = state.get("nearest_food_distance")
        trap_direction = state.get("nearest_trap_direction")
        trap_distance = state.get("nearest_trap_distance")

        # If no food exists, just move randomly.
        if food_direction is None or food_distance is None:
            return random.choice(ACTIONS)

        # If already on food, stay.
        if food_distance == 0 or food_direction == "STAY":
            return "STAY"

        # If a trap is one step away in the same direction, choose a safer move.
        if trap_distance == 1 and trap_direction == food_direction:
            alternatives = [a for a in ACTIONS if a not in (food_direction, "STAY")]
            return random.choice(alternatives) if alternatives else "STAY"

        return food_direction
