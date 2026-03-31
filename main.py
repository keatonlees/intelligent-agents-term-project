import time
from typing import Any, Dict, List, Type

from battle import Action, GridWorld
from agents.agent_greedy import GreedyAgent
from agents.agent_q_learning import QLearningAgent
from agents.agent_random import RandomAgent
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
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "q_learning": QLearningAgent,
}

SELECTED_AGENTS = list(DEFAULT_SELECTED_AGENTS)


# utility function returns an action into a printable name
def get_action_name(action: Action | str) -> str:
    if isinstance(action, Action):
        return action.name
    return str(action).upper()


# formats actions into a concise string for display 
def format_actions(actions: dict[str, Action | str]) -> str:
    return " | ".join(
        f"{agent_id}:{get_action_name(action)}" for agent_id, action in actions.items()
    )


# prints a formatted header for each step/frame of the simulation
def print_step_header(title: str) -> None:
    width = 56
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


# clears terminal for next frame
def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def render_frame(
    title: str,
    env: GridWorld,
    state: dict,
    *,
    show_frames: bool,
    animate: bool,
) -> None:
    if not show_frames:
        return

    if animate and CLEAR_SCREEN_EACH_FRAME:
        clear_terminal()

    print_step_header(title)
    env.print_grid()
    print_state_summary(state)

    if animate:
        time.sleep(FRAME_DELAY_SECONDS)


# reward shaping function that provides additional reward based on changes in distance to food and traps
def distance_shaping_reward(previous_view: dict, next_view: dict) -> float:
    # calculates difference in distance to food and traps and applies scaling factors
    def delta(key: str) -> int | None:
        prev = previous_view.get(key)
        nxt = next_view.get(key)
        if prev is None or nxt is None:
            return None
        return prev - nxt

    shaping = 0.0

    food_delta = delta("nearest_food_distance")
    if food_delta is not None:
        shaping += food_delta * Q_FOOD_DISTANCE_REWARD_SCALE

    trap_delta = delta("nearest_trap_distance")
    if trap_delta is not None:
        # Moving away from traps should be positive; toward traps should be negative.
        shaping += (-trap_delta) * Q_TRAP_DISTANCE_REWARD_SCALE

    return shaping


# concise output of current state summary
def print_state_summary(state: dict) -> None:
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


# function to initialize agent instances based on selected agents
def build_agent_instances(
    selected_agents: List[Dict[str, Any]],
    *,
    q_epsilon_start: float,
    q_epsilon_min: float,
    q_epsilon_decay: float,
    q_alpha: float | None = None,
    q_table_path: str = "q_table_iteration_4.json",
    q_load_existing: bool = True,
) -> tuple[List[Any], Dict[str, str]]:
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
                "q_table_path": q_table_path,
                "load_existing": q_load_existing,
            }
            if q_alpha is not None:
                policy_kwargs["alpha"] = q_alpha
        agent_instances.append(policy_cls(agent_id=agent_id, **policy_kwargs))
        display_chars[agent_id] = display_char

    return agent_instances, display_chars


# function to run one episode of the environment with the given agents and return the final state
def simulate_episode(
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
    render_frame(
        prefix + "AGENTS: " + ", ".join(agent_ids),
        env,
        state,
        show_frames=show_frames,
        animate=animate,
    )
    render_frame(
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

        frame_title = f"STEP {step_index + 1} | Actions: {format_actions(actions)}"
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
                    reward += distance_shaping_reward(previous_view, next_view)
                action_name = action_name(actions[agent.agent_id])
                agent.update(previous_view, action_name, reward, next_view)

        state = next_state
        render_frame(
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


# main function to run the full training/eval
def run_agents(
    selected_agents: List[Dict[str, Any]],
    *,
    mode: str | None = None,
    mode_overrides: Dict[str, Any] | None = None,
    q_alpha: float | None = None,
    q_table_path: str = "q_table_iteration_4.json",
    q_load_existing: bool = True,
) -> List[Dict[str, int]]:
    selected_mode = (mode or RUN_MODE).strip().lower()
    if selected_mode not in MODE_PRESETS:
        raise ValueError(f"RUN_MODE must be one of: {sorted(MODE_PRESETS)}")

    mode_config = dict(MODE_PRESETS[selected_mode])
    if mode_overrides:
        mode_config.update(mode_overrides)

    episodes = int(mode_config["episodes"])
    show_frames = bool(mode_config["show_frames"])
    animate = ANIMATE_TERMINAL and bool(mode_config["animate_terminal"]) and show_frames
    enable_q_updates = bool(mode_config["enable_q_updates"])
    enable_q_reward_shaping = bool(mode_config["enable_q_reward_shaping"])

    agent_instances, display_chars = build_agent_instances(
        selected_agents,
        q_epsilon_start=float(mode_config["q_epsilon_start"]),
        q_epsilon_min=float(mode_config["q_epsilon_min"]),
        q_epsilon_decay=float(mode_config["q_epsilon_decay"]),
        q_alpha=q_alpha,
        q_table_path=q_table_path,
        q_load_existing=q_load_existing,
    )

    print(
        f"Mode={selected_mode} episodes={episodes} steps_per_episode={DEMO_STEPS} "
        f"show_frames={show_frames} animate={animate}"
    )

    episode_scores: List[Dict[str, int]] = []
    for episode_index in range(1, episodes + 1):
        final_state = simulate_episode(
            agent_instances=agent_instances,
            display_chars=display_chars,
            episode_index=episode_index,
            total_episodes=episodes,
            enable_q_updates=enable_q_updates,
            enable_q_reward_shaping=enable_q_reward_shaping,
            show_frames=show_frames,
            animate=animate,
        )
        episode_scores.append(
            {
                agent_id: agent_data["score"]
                for agent_id, agent_data in final_state.get("agents", {}).items()
            }
        )

    if bool(mode_config["save_q_table"]):
        for agent in agent_instances:
            if hasattr(agent, "save"):
                agent.save()

    return episode_scores


def main() -> None:
    run_agents(SELECTED_AGENTS)


if __name__ == "__main__":
    main()
