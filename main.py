"""
Entry point for Hierarchical MARL SAR framework.
Usage:
    python main.py --mode train --episodes 500
    python main.py --mode eval  --episodes 50 --checkpoint checkpoint_ep500.pt
"""
import sys
sys.path.insert(0, r"D:\torch_pkg")

import argparse
from config import Config
from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical MARL for Search-and-Rescue")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="Run mode")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of episodes")
    parser.add_argument("--agents", type=int, default=None,
                        help="Override number of agents")
    parser.add_argument("--grid", type=int, default=None,
                        help="Override grid size (square)")
    parser.add_argument("--victims", type=int, default=None,
                        help="Override number of victims")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for evaluation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    args = parser.parse_args()

    cfg = Config()

    # Apply overrides
    if args.episodes:
        if args.mode == "train":
            cfg.train_episodes = args.episodes
        else:
            cfg.eval_episodes = args.episodes
    if args.agents:
        cfg.n_agents = args.agents
    if args.grid:
        cfg.grid_width = args.grid
        cfg.grid_height = args.grid
    if args.victims:
        cfg.n_victims = args.victims
    if args.seed:
        cfg.seed = args.seed
    # Recompute derived dims
    cfg.__post_init__()

    print(f"\nConfig: {cfg.n_agents} agents, {cfg.grid_width}x{cfg.grid_height} grid, "
          f"{cfg.n_victims} victims, obs_dim={cfg.obs_dim}, "
          f"node_feat_dim={cfg.node_feat_dim}\n")

    if args.mode == "train":
        agent = train(cfg)
        print("\nRunning post-training evaluation...")
        evaluate(cfg, agent=agent)
    else:
        evaluate(cfg, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
