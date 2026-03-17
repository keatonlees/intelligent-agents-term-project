from battle import Action, GridWorld
from settings import DEFAULT_GRID_HEIGHT, DEFAULT_GRID_WIDTH


def _format_actions(actions: dict[str, Action]) -> str:
    """Return a compact action string like `a1:RIGHT | a2:UP`."""
    return " | ".join(f"{agent_id}:{action.name}" for agent_id, action in actions.items())


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


def run_testing_agent_demo() -> None:
    """Run a short deterministic test agent demo against GridWorld."""
    env = GridWorld(
        width=DEFAULT_GRID_WIDTH,
        height=DEFAULT_GRID_HEIGHT,
        max_steps=10,
        agent_ids=["a1", "a2"],
        seed=7,
    )

    _print_step_header("INITIAL STATE")
    env.print_grid()
    _print_state_summary(env.get_state())

    demo_actions = [
        {"a1": Action.RIGHT, "a2": Action.UP},
        {"a1": Action.UP, "a2": Action.RIGHT},
        {"a1": Action.LEFT, "a2": Action.LEFT},
        {"a1": Action.STAY, "a2": Action.DOWN},
    ]

    for step_index, actions in enumerate(demo_actions, start=1):
        _print_step_header(f"STEP {step_index} | Actions: {_format_actions(actions)}")
        state = env.step(actions)
        env.print_grid()
        _print_state_summary(state)
