from battle import Action, GridWorld


def run_demo() -> None:
	"""Run a short deterministic demo of the GridWorld environment."""
	env = GridWorld(width=10, height=10, max_steps=10, agent_ids=["a1", "a2"], seed=7)

	print("Initial grid:")
	env.print_grid()
	print(env.get_state())

	demo_actions = [
		{"a1": Action.RIGHT, "a2": Action.UP},
		{"a1": Action.UP, "a2": Action.RIGHT},
		{"a1": Action.LEFT, "a2": Action.LEFT},
		{"a1": Action.STAY, "a2": Action.DOWN},
	]

	for step_index, actions in enumerate(demo_actions, start=1):
		print(f"\nStep {step_index} actions: {actions}")
		state = env.step(actions)
		env.print_grid()
		print(state)


if __name__ == "__main__":
	run_demo()
