import json
import random
import os

# simple actions list
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


class QLearningAgent:
    def __init__(self, name="q_agent", alpha=0.1, gamma=0.9, epsilon=0.1):
        self.name = name

        # learning settings
        self.alpha = alpha      # how fast it learns
        self.gamma = gamma      # future reward importance
        self.epsilon = epsilon  # randomness (exploration)

        # where we store what it learns
        self.q_table = {}

        # load saved learning if it exists
        if os.path.exists("q_table.json"):
            with open("q_table.json", "r") as f:
                self.q_table = json.load(f)

    # turn the game state into something simple the agent can remember
    def get_state_key(self, state):
        my_pos = tuple(state["self_pos"])
        enemy_pos = tuple(state["enemy_pos"])

        # just use positions for now (keep it simple)
        return str((my_pos, enemy_pos))

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
        return max(q_values, key=q_values.get)

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

    # save learning to file
    def save(self):
        with open("q_table.json", "w") as f:
            json.dump(self.q_table, f)
