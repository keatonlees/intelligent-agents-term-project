from __future__ import annotations

import random
from typing import Any, Dict

from battle import Action


class AgentRandom:
    """Random policy that samples uniformly from all available actions."""

    policy_name = "agent_random"

    def __init__(self, agent_id: str) -> None:
        """Initialize one random-policy agent instance."""
        self.agent_id = agent_id
        self._actions = list(Action)

    def action_for_step(self, step_index: int, state: Dict[str, Any]) -> Action:
        """Return a uniformly random action for this step."""
        del step_index
        del state
        return random.choice(self._actions)
