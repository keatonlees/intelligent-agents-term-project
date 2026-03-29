import time
from typing import Any, Dict, List, Type

from battle import Action, GridWorld
from agents.agent_greedy import GreedyAgent
from agents.agent_q_learning import QLearningAgent
from agents.agent_random import AgentRandom
from settings import (
    ANIMATE_TERMINAL,
    CLEAR_SCREEN_EACH_FRAME,
    DEFAULT_SELECTED_AGENTS,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_GRID_WIDTH,
    DEMO_STEPS,
    FRAME_DELAY_SECONDS,
    MODE_PRESETS,
    Q_FOOD_DISTANCE_REWARD_SCALE,
    Q_TRAP_DISTANCE_REWARD_SCALE,
    RUN_MODE,
)


AGENT_REGISTRY: Dict[str, Type[Any]] = {
    "random": AgentRandom,
    "greedy": GreedyAgent,
    "q_learning": QLearningAgent,
}

SELECTED_AGENTS = list(DEFAULT_SELECTED_AGENTS)


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


def _clear_terminal() -> None:
    """Clear the terminal and reset cursor to top-left."""
    print("\033[2J\033[H", end="")


def _render_frame(
    title: str,
    env: GridWorld,
    state: dict,
    *,
    show_frames: bool,
    animate: bool,
) -> None:
    """Render one frame of the simulation output."""
    if not show_frames:
        return

    if animate and CLEAR_SCREEN_EACH_FRAME:
        _clear_terminal()

    _print_step_header(title)
    env.print_grid()
    _print_state_summary(state)

    if animate:
        time.sleep(FRAME_DELAY_SECONDS)


def _distance_shaping_reward(previous_view: dict, next_view: dict) -> float:
    """Reward moving toward food and away from traps using distance deltas."""

    def _delta(key: str) -> int | None:
        prev = previous_view.get(key)
        nxt = next_view.get(key)
        if prev is None or nxt is None:
            return None
        return prev - nxt

    shaping = 0.0

    food_delta = _delta("nearest_food_distance")
    if food_delta is not None:
        shaping += food_delta * Q_FOOD_DISTANCE_REWARD_SCALE

    trap_delta = _delta("nearest_trap_distance")
    if trap_delta is not None:
        # Moving away from traps should be positive; toward traps should be negative.
        shaping += (-trap_delta) * Q_TRAP_DISTANCE_REWARD_SCALE

    return shaping


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


def _build_agent_instances(
    selected_agents: List[Dict[str, Any]],
    *,
    q_epsilon_start: float,
    q_epsilon_min: float,
    q_epsilon_decay: float,
) -> tuple[List[Any], Dict[str, str]]:
    """Build and return policy instances and display character map."""
    if not selected_agents:
        raise ValueError("No agents selected.")

    agent_instances: List[Any] = []
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
        policy_kwargs: Dict[str, Any] = {}
        if policy_name == "q_learning":
            policy_kwargs = {
                "epsilon": q_epsilon_start,
                "epsilon_min": q_epsilon_min,
                "epsilon_decay": q_epsilon_decay,
            }
        agent_instances.append(policy_cls(agent_id=agent_id, **policy_kwargs))
        display_chars[agent_id] = display_char

    return agent_instances, display_chars


def _run_episode(
    *,
    agent_instances: List[Any],
    display_chars: Dict[str, str],
    episode_index: int,
    total_episodes: int,
    enable_q_updates: bool,
    enable_q_reward_shaping: bool,
    show_frames: bool,
    animate: bool,
) -> Dict[str, Any]:
    """Run one episode and return final environment state."""

    agent_ids = [agent.agent_id for agent in agent_instances]
    env = GridWorld(
        width=DEFAULT_GRID_WIDTH,
        height=DEFAULT_GRID_HEIGHT,
        max_steps=DEMO_STEPS,
        agent_ids=agent_ids,
        agent_display_chars=display_chars,
        seed=time.time_ns(),
    )

    state = env.get_state()
    prefix = f"EP {episode_index}/{total_episodes} | "
    _render_frame(
        prefix + "AGENTS: " + ", ".join(agent_ids),
        env,
        state,
        show_frames=show_frames,
        animate=animate,
    )
    _render_frame(
        prefix + "INITIAL STATE",
        env,
        state,
        show_frames=show_frames,
        animate=animate,
    )

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

        frame_title = f"STEP {step_index + 1} | Actions: {_format_actions(actions)}"
        next_state = env.step(actions)

        if enable_q_updates:
            for agent in agent_instances:
                if not hasattr(agent, "update"):
                    continue

                previous_view = views_before_step[agent.agent_id]
                next_view = env.get_agent_view(agent.agent_id, next_state)
                reward = float(next_state.get("rewards", {}).get(agent.agent_id, 0))
                if (
                    enable_q_reward_shaping
                    and getattr(agent, "policy_name", "") == "agent_q_learning"
                ):
                    reward += _distance_shaping_reward(previous_view, next_view)
                action_name = _action_name(actions[agent.agent_id])
                agent.update(previous_view, action_name, reward, next_view)

        state = next_state
        _render_frame(
            prefix + frame_title,
            env,
            state,
            show_frames=show_frames,
            animate=animate,
        )
        if state["done"]:
            break

    scores = {
        agent_id: data["score"] for agent_id, data in state.get("agents", {}).items()
    }
    print(
        f"Episode {episode_index}/{total_episodes} finished | "
        f"step={state['step']} done={state['done']} scores={scores}"
    )

    return state


def run_agents(selected_agents: List[Dict[str, Any]]) -> None:
    """Run train/eval workflow using the configured mode."""
    mode = RUN_MODE.strip().lower()
    if mode not in MODE_PRESETS:
        raise ValueError(f"RUN_MODE must be one of: {sorted(MODE_PRESETS)}")

    mode_config = MODE_PRESETS[mode]
    episodes = int(mode_config["episodes"])
    show_frames = bool(mode_config["show_frames"])
    animate = ANIMATE_TERMINAL and bool(mode_config["animate_terminal"]) and show_frames
    enable_q_updates = bool(mode_config["enable_q_updates"])
    enable_q_reward_shaping = bool(mode_config["enable_q_reward_shaping"])

    agent_instances, display_chars = _build_agent_instances(
        selected_agents,
        q_epsilon_start=float(mode_config["q_epsilon_start"]),
        q_epsilon_min=float(mode_config["q_epsilon_min"]),
        q_epsilon_decay=float(mode_config["q_epsilon_decay"]),
    )

    print(
        f"Mode={mode} episodes={episodes} steps_per_episode={DEMO_STEPS} "
        f"show_frames={show_frames} animate={animate}"
    )

    for episode_index in range(1, episodes + 1):
        _run_episode(
            agent_instances=agent_instances,
            display_chars=display_chars,
            episode_index=episode_index,
            total_episodes=episodes,
            enable_q_updates=enable_q_updates,
            enable_q_reward_shaping=enable_q_reward_shaping,
            show_frames=show_frames,
            animate=animate,
        )

    if bool(mode_config["save_q_table"]):
        for agent in agent_instances:
            if hasattr(agent, "save"):
                agent.save()


def main() -> None:
    """Run the configured set of agent instances."""
    run_agents(SELECTED_AGENTS)


if __name__ == "__main__":
    main()
