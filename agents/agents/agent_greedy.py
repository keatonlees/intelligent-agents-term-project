import random

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class GreedyAgent:
    def __init__(self, name="greedy_agent"):
        self.name = name

    def choose_action(self, state):
        """
        This agent is simple:
        - move toward the closest food
        - avoid stepping on traps if possible
        """

        my_x, my_y = state["self_pos"]
        food_positions = state.get("food_positions", [])
        trap_positions = state.get("trap_positions", [])

        # If no food exists, just move randomly
        if not food_positions:
            return random.choice(ACTIONS)

        # Find the closest food using simple distance
        closest_food = min(
            food_positions,
            key=lambda f: abs(f[0] - my_x) + abs(f[1] - my_y)
        )

        food_x, food_y = closest_food

        # Decide preferred direction toward food
        possible_moves = []

        if food_x > my_x:
            possible_moves.append("RIGHT")
        elif food_x < my_x:
            possible_moves.append("LEFT")

        if food_y > my_y:
            possible_moves.append("UP")
        elif food_y < my_y:
            possible_moves.append("DOWN")

        # If already on food, just stay
        if not possible_moves:
            return "STAY"

        # Avoid traps: filter out moves that lead to trap
        safe_moves = []

        for move in possible_moves:
            next_pos = self.get_next_position(my_x, my_y, move)

            if next_pos not in trap_positions:
                safe_moves.append(move)

        # If we found safe moves, use them
        if safe_moves:
            return random.choice(safe_moves)

        # If all moves are risky, just pick one anyway
        return random.choice(possible_moves)

    def get_next_position(self, x, y, action):
        """
        Given a move, where would we end up?
        """

        if action == "UP":
            return [x, y + 1]
        if action == "DOWN":
            return [x, y - 1]
        if action == "LEFT":
            return [x - 1, y]
        if action == "RIGHT":
            return [x + 1, y]

        return [x, y]
