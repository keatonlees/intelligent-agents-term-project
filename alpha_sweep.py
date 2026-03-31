from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from main import SELECTED_AGENTS, run_agents

ALPHA_VALUES = [0.05, 0.1, 0.2, 0.3]
TRIALS_PER_ALPHA = 3
TRAIN_EPISODES = 1500
TAIL_WINDOW = 200


# get the agent IDs of all q_learning agents in the selected agents list
def q_agent_ids(selected_agents: List[Dict[str, Any]]) -> List[str]:
    return [item["agent_id"] for item in selected_agents if item.get("policy") == "q_learning"]


# compute the mean score for the specified agent over the last N episodes, where N is the tail_window
def mean_tail_score(episode_scores: List[Dict[str, int]], agent_id: str, tail_window: int) -> float:
    if not episode_scores:
        return 0.0
    tail = episode_scores[-min(tail_window, len(episode_scores)) :]
    values = [scores.get(agent_id, 0) for scores in tail]
    return float(mean(values)) if values else 0.0


# run a sweep over different alpha values for the RL agent
def run_alpha_sweep() -> None:
    q_ids = q_agent_ids(list(SELECTED_AGENTS))
    if not q_ids:
        raise ValueError("No q_learning agents are configured in SELECTED_AGENTS.")

    q_id = q_ids[0]
    results: List[Dict[str, float]] = []

    for alpha in ALPHA_VALUES:
        trial_scores: List[float] = []
        print(f"\n--- alpha={alpha} ---")

        for trial_index in range(1, TRIALS_PER_ALPHA + 1):
            scores = run_agents(
                list(SELECTED_AGENTS),
                mode="train",
                mode_overrides={
                    "episodes": TRAIN_EPISODES,
                    "show_frames": False,
                    "animate_terminal": False,
                    "save_q_table": False,
                    "enable_q_updates": True,
                    "enable_q_reward_shaping": True,
                },
                q_alpha=alpha,
                q_load_existing=False,
            )

            tail_mean = mean_tail_score(scores, q_id, TAIL_WINDOW)
            trial_scores.append(tail_mean)
            print(
                f"trial {trial_index}/{TRIALS_PER_ALPHA}: "
                f"mean {q_id} score over last {TAIL_WINDOW} episodes = {tail_mean:.2f}"
            )

        alpha_mean = float(mean(trial_scores)) if trial_scores else 0.0
        results.append({"alpha": alpha, "score": alpha_mean})
        print(f"alpha={alpha} average tail score across trials = {alpha_mean:.2f}")

    ranked = sorted(results, key=lambda row: row["score"], reverse=True)

    print("\n=== Alpha Sweep Summary (higher is better) ===")
    for row in ranked:
        print(f"alpha={row['alpha']:<4} avg_tail_score={row['score']:.2f}")

    best = ranked[0]
    print(f"\nBest alpha in this run: {best['alpha']} (score={best['score']:.2f})")


if __name__ == "__main__":
    run_alpha_sweep()
