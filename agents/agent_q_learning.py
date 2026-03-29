import json
import random
import os
from typing import Any, Dict

# simple actions list
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class QLearningAgent:
    policy_name = "agent_q_learning"

    def __init__(
        self,
        agent_id: str,
        name: str = "q_agent",
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.agent_id = agent_id
        self.name = name

        # learning settings
        self.alpha = alpha      # how fast it learns
        self.gamma = gamma      # future reward importance
        self.epsilon = epsilon  # randomness (exploration)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # where we store what it learns
        self.q_table = {}

        # load saved learning if it exists
        if os.path.exists("q_table.json"):
            with open("q_table.json", "r") as f:
                self.q_table = json.load(f)

    def action_for_step(self, step_index: int, state: Dict[str, Any]) -> str:
        del step_index
        return self.choose_action(state)

    # turn the game state into something simple the agent can remember
    def get_state_key(self, state):
        my_pos = tuple(state["self_pos"])
        enemy_dir = state.get("nearest_enemy_direction")
        enemy_dist = state.get("nearest_enemy_distance")

        food_dir = state.get("nearest_food_direction")
        food_dist = state.get("nearest_food_distance")
        trap_dir = state.get("nearest_trap_direction")
        trap_dist = state.get("nearest_trap_distance")

        # include compact nearest-object features so learning does not depend on raw lists
        return str((my_pos, enemy_dir, enemy_dist, food_dir, food_dist, trap_dir, trap_dist))

    # get Q value (or 0 if we haven't seen this state yet)
    def get_q(self, state_key, action):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in ACTIONS}
        return self.q_table[state_key][action]

    # choose action (sometimes random, sometimes best known)
    def choose_action(self, state):
        state_key = self.get_state_key(state)

        # explore
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        # exploit (pick best known move)
        q_values = self.q_table.get(state_key, {a: 0 for a in ACTIONS})
        max_q = max(q_values.values())
        best_actions = [action for action, value in q_values.items() if value == max_q]
        return random.choice(best_actions)

    # update learning after each move
    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        # make sure states exist
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in ACTIONS}
        if next_key not in self.q_table:
            self.q_table[next_key] = {a: 0 for a in ACTIONS}

        # current value
        current_q = self.q_table[state_key][action]

        # best future value
        max_future_q = max(self.q_table[next_key].values())

        # Q learning formula (don’t overthink this)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        self.q_table[state_key][action] = new_q
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # save learning to file
    def save(self):
        with open("q_table.json", "w") as f:
            json.dump(self.q_table, f)
