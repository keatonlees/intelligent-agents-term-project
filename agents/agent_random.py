from __future__ import annotations

import random
from typing import Any, Dict

from battle import Action


# a simple agent that selects actions uniformly at random
class RandomAgent:
    policy_name = "agent_random"

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._actions = list(Action)

    def action_for_step(self, step_index: int, state: Dict[str, Any]) -> Action:
        del step_index
        del state
        return random.choice(self._actions)
