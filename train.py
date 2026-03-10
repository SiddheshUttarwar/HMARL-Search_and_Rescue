"""
Training loop for Hierarchical MARL — implements Algorithm 1.
"""
import numpy as np
import torch
import time

from config import Config
from environment import SAREnvironment
from agents import HierarchicalMARLAgent
from graph import get_node_features


def train(cfg: Config):
    """Main training loop."""
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = SAREnvironment(cfg, seed=cfg.seed)
    agent = HierarchicalMARLAgent(cfg)

    print("=" * 60)
    print("  Hierarchical MARL Training (Actor-Critic)")
    print(f"  Agents: {cfg.n_agents}  |  Grid: {cfg.grid_width}x{cfg.grid_height}  |  Obstacles: {cfg.n_obstacles}")
    print(f"  Victims: {cfg.n_victims}  |  Episodes: {cfg.train_episodes}")
    print("=" * 60)

    all_rewards = []
    all_entropy = []

    for ep in range(1, cfg.train_episodes + 1):
        ep_start = time.time()
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        option_switches = 0

        # Initialise options
        hl_obs = obs.copy()
        options, hl_lp, _ = agent.select_options(obs, step=0)

        old_potential = env.belief.potential()

        with torch.no_grad():
            positions = env.get_agent_positions()
            nf = get_node_features(obs, agent.current_options, cfg.n_options)
            from graph import build_graph
            ei, ef, _ = build_graph(positions, cfg.proximity_radius)
            hl_value = agent.critic(nf, ei, ef).item()

        while not done:
            positions = env.get_agent_positions()

            # Select actions (low-level)
            actions, lp, value, nf, ei, ef = agent.select_actions(obs, positions)

            # Step environment
            next_obs, rewards, done, info = env.step(actions)
            ep_reward += rewards.sum()
            ep_steps += 1

            # Store low-level transition
            agent.buffer.add_step(
                obs=obs, node_feat=nf, edge_index=ei, edge_feat=ef,
                actions=actions, log_probs=lp, rewards=rewards,
                value=value, done=done, options=agent.current_options.copy()
            )

            # Check termination
            terminate = agent.check_termination(next_obs, env.get_agent_positions(),
                                                ep_steps)

            if np.any(terminate) or done:
                new_potential = env.belief.potential()
                dt = ep_steps - int(agent.option_start_step.mean())
                rh = agent.compute_high_level_reward(old_potential, new_potential, dt)

                # Store high-level transition
                agent.buffer.add_hl_transition(
                    obs=hl_obs, option=agent.current_options.copy(),
                    log_prob=hl_lp, reward=rh, value=hl_value, done=done
                )

                if not done:
                    # Select new options
                    hl_obs = next_obs.copy()
                    options, hl_lp, _ = agent.select_options(next_obs, ep_steps)
                    option_switches += 1
                    old_potential = new_potential

                    with torch.no_grad():
                        pos = env.get_agent_positions()
                        nf_hl = get_node_features(next_obs, agent.current_options,
                                                   cfg.n_options)
                        ei_hl, ef_hl, _ = build_graph(pos, cfg.proximity_radius)
                        hl_value = agent.critic(nf_hl, ei_hl, ef_hl).item()

            obs = next_obs

            # Periodic PPO update
            if len(agent.buffer.obs) >= cfg.rollout_steps:
                losses = agent.update()

        # End-of-episode update
        if len(agent.buffer.obs) > 0:
            losses = agent.update()
        else:
            losses = {}

        ep_time = time.time() - ep_start
        ep_entropy = env.belief.entropy()
        all_rewards.append(ep_reward)
        all_entropy.append(ep_entropy)

        if ep % cfg.log_interval == 0:
            avg_r = np.mean(all_rewards[-cfg.log_interval:])
            avg_e = np.mean(all_entropy[-cfg.log_interval:])
            print(f"Ep {ep:4d} | R={avg_r:7.2f} | H={avg_e:6.2f} | "
                  f"Steps={ep_steps:3d} | Switches={option_switches:2d} | "
                  f"Found={info.get('victims_found', 0)}/{cfg.n_victims} | "
                  f"Loss A={losses.get('actor_loss', 0):.4f} "
                  f"C={losses.get('critic_loss', 0):.4f} | "
                  f"Time={ep_time:.1f}s")

        if ep % cfg.save_interval == 0:
            _save_checkpoint(agent, ep, cfg)

    print("\nTraining complete.")
    _save_checkpoint(agent, cfg.train_episodes, cfg)
    return agent


def _save_checkpoint(agent, ep, cfg):
    path = f"checkpoint_ep{ep}.pt"
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "termination": agent.termination.state_dict(),
        "manager": agent.manager.state_dict(),
        "episode": ep,
    }, path)
    print(f"  Checkpoint saved -> {path}")
