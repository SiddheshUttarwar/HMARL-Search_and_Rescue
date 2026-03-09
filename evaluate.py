"""
Evaluation script for Hierarchical MARL.
Runs trained policies on held-out test episodes (different seed).
"""
import numpy as np
import torch

from config import Config
from environment import SAREnvironment
from agents import HierarchicalMARLAgent


def evaluate(cfg: Config, agent: HierarchicalMARLAgent | None = None,
             checkpoint_path: str | None = None):
    """
    Evaluate on D_test (seed-separated from D_train).
    Computes: entropy reduction rate, time-to-detection, coverage overlap,
              switching frequency.
    """
    # Load checkpoint if agent not passed directly
    if agent is None:
        agent = HierarchicalMARLAgent(cfg)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu",
                              weights_only=True)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.termination.load_state_dict(ckpt["termination"])
            agent.manager.load_state_dict(ckpt["manager"])
            print(f"Loaded checkpoint: {checkpoint_path}")

    agent.actor.eval()
    agent.critic.eval()
    agent.termination.eval()
    agent.manager.eval()

    env = SAREnvironment(cfg, seed=cfg.eval_seed)

    print("=" * 60)
    print("  Evaluation")
    print(f"  Episodes: {cfg.eval_episodes}  |  Seed: {cfg.eval_seed}")
    print("=" * 60)

    metrics = {
        "entropy_reduction_rate": [],
        "time_to_detection": [],
        "coverage_overlap": [],
        "switching_frequency": [],
        "victims_found": [],
        "episode_length": [],
    }

    for ep in range(1, cfg.eval_episodes + 1):
        obs = env.reset()
        done = False
        steps = 0
        option_switches = 0
        initial_entropy = env.belief.entropy()
        visited_cells = {}  # cell → set of agent ids

        # Select initial options
        agent.select_options(obs, step=0)
        detection_time = None

        while not done:
            positions = env.get_agent_positions()

            # Track cell visits for coverage overlap
            for i in range(cfg.n_agents):
                cell = (int(positions[i, 0]), int(positions[i, 1]))
                if cell not in visited_cells:
                    visited_cells[cell] = set()
                visited_cells[cell].add(i)

            with torch.no_grad():
                actions, _, _, _, _, _ = agent.select_actions(obs, positions)

            obs, _, done, info = env.step(actions)
            steps += 1

            # Time to first detection
            if detection_time is None and info["victims_found"] > 0:
                detection_time = steps

            # Check termination
            terminate = agent.check_termination(
                obs, env.get_agent_positions(), steps)
            if np.any(terminate) and not done:
                agent.select_options(obs, steps)
                option_switches += 1

        final_entropy = env.belief.entropy()
        entropy_reduction = initial_entropy - final_entropy
        ent_rate = entropy_reduction / max(steps, 1)

        # Coverage overlap
        total_visited = len(visited_cells)
        overlap_cells = sum(1 for agents in visited_cells.values()
                            if len(agents) > 1)
        coverage_overlap = overlap_cells / max(total_visited, 1)

        switch_freq = option_switches / max(steps, 1)

        metrics["entropy_reduction_rate"].append(ent_rate)
        metrics["time_to_detection"].append(
            detection_time if detection_time else steps)
        metrics["coverage_overlap"].append(coverage_overlap)
        metrics["switching_frequency"].append(switch_freq)
        metrics["victims_found"].append(info["victims_found"])
        metrics["episode_length"].append(steps)

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("  Evaluation Results")
    print("-" * 50)
    print(f"  {'Metric':<30s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 50)
    for name, vals in metrics.items():
        arr = np.array(vals)
        print(f"  {name:<30s} {arr.mean():8.3f} {arr.std():8.3f}")
    print("-" * 50)

    return metrics
