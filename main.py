import time
from typing import Any, Dict, List, Type

from battle import Action, GridWorld
from agents.agent_greedy import GreedyAgent
from agents.agent_q_learning import QLearningAgent
from agents.agent_random import AgentRandom
from settings import (
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
)


AGENT_REGISTRY: Dict[str, Type[Any]] = {
    "random": AgentRandom,
    "greedy": GreedyAgent,
    "q_learning": QLearningAgent,
}

SELECTED_AGENTS = [
    {"policy": "q_learning", "agent_id": "q1", "display_char": "Q"},
    {"policy": "greedy", "agent_id": "g1", "display_char": "G"},
]
DEMO_STEPS = 100


def _action_name(action: Action | str) -> str:
    if isinstance(action, Action):
        return action.name
    return str(action).upper()


def _format_actions(actions: dict[str, Action | str]) -> str:
    """Return a compact action string like `a1:RIGHT | a2:UP`."""
    return " | ".join(
        f"{agent_id}:{_action_name(action)}" for agent_id, action in actions.items()
    )


def _print_step_header(title: str) -> None:
    """Print a visual section header for cleaner step separation."""
    width = 56
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def _print_state_summary(state: dict) -> None:
    """Print a concise and readable summary of environment state."""
    print(f"Step: {state['step']}/{state['max_steps']} | Done: {state['done']}")
    print("Agents:")
    print("  ID   Pos      Health  Score")
    for agent_id, agent in state["agents"].items():
        position = agent["position"]
        print(
            f"  {agent_id:<4} ({position['x']},{position['y']})"
            f"   {agent['health']:<6}  {agent['score']}"
        )

    events = state.get("events", [])
    if events:
        print("Events:")
        for event in events:
            event_type = event["event"]
            position = event["position"]
            details = []
            if "score_change" in event:
                details.append(f"score {event['score_change']:+d}")
            if "health_change" in event:
                details.append(f"health {event['health_change']:+d}")
            detail_text = ", ".join(details) if details else "no delta"
            print(
                f"  - {event['agent_id']} {event_type} at "
                f"({position['x']},{position['y']}), {detail_text}"
            )
    else:
        print("Events: none")


def run_agents(selected_agents: List[Dict[str, str]]) -> None:
    """Run one environment using the selected agent instances."""
    if not selected_agents:
        raise ValueError("No agents selected.")

    agent_instances = []
    display_chars: Dict[str, str] = {}
    for selection in selected_agents:
        policy_name = selection["policy"]
        agent_id = selection["agent_id"]
        display_char = selection.get("display_char", "A")
        if len(display_char) != 1:
            raise ValueError(
                f"display_char for agent '{agent_id}' must be a single character."
            )
        if policy_name not in AGENT_REGISTRY:
            raise ValueError(
                f"Unknown policy '{policy_name}'. Available: {list(AGENT_REGISTRY)}"
            )
        policy_cls = AGENT_REGISTRY[policy_name]
        agent_instances.append(policy_cls(agent_id=agent_id))
        display_chars[agent_id] = display_char

    agent_ids = [agent.agent_id for agent in agent_instances]
    env = GridWorld(
        width=DEFAULT_GRID_WIDTH,
        height=DEFAULT_GRID_HEIGHT,
        max_steps=DEMO_STEPS,
        agent_ids=agent_ids,
        agent_display_chars=display_chars,
        seed=time.time_ns(),
    )

    _print_step_header("AGENTS: " + ", ".join(agent_ids))
    _print_step_header("INITIAL STATE")
    env.print_grid()
    state = env.get_state()
    _print_state_summary(state)

    for step_index in range(DEMO_STEPS):
        views_before_step = {
            agent.agent_id: env.get_agent_view(agent.agent_id, state)
            for agent in agent_instances
        }
        actions = {
            agent.agent_id: agent.action_for_step(
                step_index,
                views_before_step[agent.agent_id],
            )
            for agent in agent_instances
        }

        _print_step_header(f"STEP {step_index + 1} | Actions: {_format_actions(actions)}")
        next_state = env.step(actions)

        for agent in agent_instances:
            if not hasattr(agent, "update"):
                continue

            reward = next_state.get("rewards", {}).get(agent.agent_id, 0)
            previous_view = views_before_step[agent.agent_id]
            next_view = env.get_agent_view(agent.agent_id, next_state)
            action_name = _action_name(actions[agent.agent_id])
            agent.update(previous_view, action_name, reward, next_view)

        state = next_state
        env.print_grid()
        _print_state_summary(state)
        if state["done"]:
            break

    for agent in agent_instances:
        if hasattr(agent, "save"):
            agent.save()


def main() -> None:
    """Run the configured set of agent instances."""
    run_agents(SELECTED_AGENTS)


if __name__ == "__main__":
    main()
