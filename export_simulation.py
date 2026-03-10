"""
Export a trained Hierarchical MARL agent simulation trajectory to a JSON file.
This allows playback of the exact PyTorch decisions inside simulation.html.
Usage: python export_simulation.py --checkpoint checkpoints/checkpoint_ep250.pt
"""
import argparse
import json
import numpy as np
import torch

from config import Config
from environment import SAREnvironment
from agents import HierarchicalMARLAgent

def main():
    parser = argparse.ArgumentParser(description="Export SAR Simulation to JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for environment")
    parser.add_argument("--output", type=str, default="sar_playback.html", help="Output HTML path")
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

    print("Simulating episode...")
    obs = env.reset()
    done = False
    opt_switches = 0
    step = 0
    
    # Base configuration to export
    export_data = {
        "config": {
            "gridW": cfg.grid_width,
            "gridH": cfg.grid_height,
            "nAgents": cfg.n_agents,
            "nVictims": cfg.n_victims,
            "nObstacles": cfg.n_obstacles
        },
        "obstacles": env.obstacles.tolist(),
        "victims": [{"x": int(v[0]), "y": int(v[1]), "found": False, "foundStep": -1} for v in env.victim_pos],
        "frames": []
    }
    
    options, _, _ = agent.select_options(obs, step=0)
    
    # Save step 0
    def capture_frame():
        return {
            "step": step,
            "agents": [{"x": int(env.agent_pos[i, 0]), "y": int(env.agent_pos[i, 1]), "option": int(options[i])} for i in range(cfg.n_agents)],
            "victims_found": env.victims_found.tolist(),
            "belief": env.belief.grid.tolist(),
            "entropy": float(env.belief.entropy()),
            "potential": float(env.belief.potential()),
            "switches": opt_switches
        }
        
    export_data["frames"].append(capture_frame())

    while not done:
        positions = env.get_agent_positions()
        
        with torch.no_grad():
            actions, _, _, _, _, _ = agent.select_actions(obs, positions)
            
        obs, _, done, info = env.step(actions)
        step += 1
        
        # In a real environment, victims change state here
        for i, found in enumerate(env.victims_found):
            if found and not export_data["victims"][i]["found"]:
                export_data["victims"][i]["found"] = True
                export_data["victims"][i]["foundStep"] = step
        
        terminate = agent.check_termination(obs, env.get_agent_positions(), step)
        if np.any(terminate) and not done:
            options, _, _ = agent.select_options(obs, step)
            opt_switches += 1
            
        export_data["frames"].append(capture_frame())

    print(f"Simulation finished in {step} steps. Found {info['victims_found']}/{cfg.n_victims} victims.")
    
    # Read the base simulation HTML
    try:
        with open("simulation.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        print("Error: simulation.html not found. Make sure you run this script from the project root.")
        return

    # Inject the JSON payload directly into the HTML JS variables
    json_str = json.dumps(export_data)
    target_str = "let playbackData = window.playbackData || null;\n    let isPlayback = !!playbackData;"
    replacement_str = f"let playbackData = {json_str};\n    let isPlayback = true;"
    
    if target_str not in html_content:
        print("Warning: Could not find the playback injection target in simulation.html")
    else:
        html_content = html_content.replace(target_str, replacement_str)
        
    with open(args.output, 'w', encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Exported standalone HTML playback successfully to {args.output}")

if __name__ == "__main__":
    main()
