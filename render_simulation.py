"""
Render a trained Hierarchical MARL agent simulation using Matplotlib.
Usage: python render_simulation.py --checkpoint checkpoints/checkpoint_ep250.pt
"""
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import torch

from config import Config
from environment import SAREnvironment
from agents import HierarchicalMARLAgent

def main():
    parser = argparse.ArgumentParser(description="Render SAR Simulation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for environment")
    args = parser.parse_args()

    cfg = Config()
    cfg.seed = args.seed
    cfg.__post_init__()

    # Initialize environment and agent
    env = SAREnvironment(cfg, seed=cfg.seed)
    agent = HierarchicalMARLAgent(cfg)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.termination.load_state_dict(ckpt["termination"])
    agent.manager.load_state_dict(ckpt["manager"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    agent.actor.eval()
    agent.critic.eval()
    agent.termination.eval()
    agent.manager.eval()

    # Pre-record entire episode trajectory
    print("Simulating episode...")
    obs = env.reset()
    done = False
    opt_switches = 0
    step = 0
    
    frames_data = []
    
    # Select initial options
    options, _, _ = agent.select_options(obs, step=0)
    
    # Initial state
    frames_data.append({
        "step": step,
        "agents": env.get_agent_positions().copy(),
        "victims": env.victim_pos.copy(),
        "found": env.victims_found.copy(),
        "obstacles": env.obstacles.copy(),
        "belief": env.belief.grid.copy(),
        "options": options.copy()
    })

    while not done:
        positions = env.get_agent_positions()
        
        with torch.no_grad():
            actions, _, _, _, _, _ = agent.select_actions(obs, positions)
            
        obs, _, done, info = env.step(actions)
        step += 1
        
        terminate = agent.check_termination(obs, env.get_agent_positions(), step)
        if np.any(terminate) and not done:
            options, _, _ = agent.select_options(obs, step)
            opt_switches += 1
            
        frames_data.append({
            "step": step,
            "agents": env.get_agent_positions().copy(),
            "victims": env.victim_pos.copy(),
            "found": env.victims_found.copy(),
            "obstacles": env.obstacles.copy(),
            "belief": env.belief.grid.copy(),
            "options": options.copy()
        })

    print(f"Simulation finished in {step} steps. Found {info['victims_found']}/{cfg.n_victims} victims.")
    print("Rendering animation...")

    # === Matplotlib Animation Setup ===
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Search and Rescue - Hierarchical MARL')
    
    # Custom colormap for belief (dark blue to yellow)
    cdict = {'red':   [[0.0,  0.05, 0.05], [1.0,  0.9, 0.9]],
             'green': [[0.0,  0.05, 0.05], [1.0,  0.9, 0.9]],
             'blue':  [[0.0,  0.2, 0.2], [1.0,  0.1, 0.1]]}
    belief_cmap = LinearSegmentedColormap('BeliefMap', cdict)
    
    # Store dynamic plot elements
    img = ax.imshow(frames_data[0]["belief"], cmap=belief_cmap, vmin=0, vmax=1, origin='lower')
    
    # Plot obstacles as black squares
    obs_y, obs_x = np.where(frames_data[0]["obstacles"])
    ax.scatter(obs_x, obs_y, c='black', marker='s', s=100, label='Obstacle')
    
    agent_scatter = ax.scatter([], [], c=[], s=150, edgecolors='white', zorder=5, label='Agent')
    victim_scatter = ax.scatter([], [], c=[], marker='*', s=200, edgecolors='white', zorder=4, label='Victim')
    
    title = ax.set_title('')
    ax.set_xlim(-0.5, cfg.grid_width - 0.5)
    ax.set_ylim(-0.5, cfg.grid_height - 0.5)
    ax.set_xticks(np.arange(cfg.grid_width))
    ax.set_yticks(np.arange(cfg.grid_height))
    ax.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Colors for options: 0: Explore (Blue), 1: Navigate (Green), 2: Form (Purple)
    option_colors = ['#3b82f6', '#10b981', '#8b5cf6']
    option_names = ['Explore', 'Navigate', 'Form']
    
    def update(frame_idx):
        frame = frames_data[frame_idx]
        
        # Update Belief Background
        img.set_data(frame["belief"])
        
        # Update Agents
        agent_scatter.set_offsets(frame["agents"])
        a_colors = [option_colors[opt] for opt in frame["options"]]
        agent_scatter.set_color(a_colors)
        agent_scatter.set_edgecolor('white')
        
        # Update Victims (Red if undiscovered, Grey if found)
        victim_scatter.set_offsets(frame["victims"])
        v_colors = ['#64748b' if f else '#ef4444' for f in frame["found"]]
        victim_scatter.set_color(v_colors)
        victim_scatter.set_edgecolor('white')
        
        found = sum(frame["found"])
        title.set_text(f"Step: {frame['step']} | Victims Found: {found}/{cfg.n_victims}\n"
                       f"Blue=Explore, Green=Navigate, Purple=Form")
        return [img, agent_scatter, victim_scatter, title]

    print("Generating frames...")
    anim = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=200, blit=True)
    
    plt.tight_layout()
    
    # Save as GIF
    save_path = "sar_simulation_ep250.gif"
    print(f"Saving animation to {save_path}...")
    writer = animation.PillowWriter(fps=10)
    anim.save(save_path, writer=writer)
    print("Done!")

if __name__ == "__main__":
    main()
