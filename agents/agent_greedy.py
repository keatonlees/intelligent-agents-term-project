import random

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class GreedyAgent:
    def __init__(self, name="greedy_agent"):
        self.name = name

    def choose_action(self, state):
        """
        This agent is simple:
        - move toward the nearest food direction
        - avoid stepping toward a trap when it's adjacent
        """

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
