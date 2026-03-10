import sys
sys.path.insert(0, r"D:\torch_pkg")

import asyncio
import json
import logging
import websockets
import torch
import numpy as np

from config import Config
from environment import SAREnvironment
from agents import HierarchicalMARLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Initialize global instances matching the training setting
cfg = Config(seed=42)  # Use seed 42 to get the EXACT SAME wall configuration as training
agent = HierarchicalMARLAgent(cfg)
ckpt = torch.load("checkpoint_ep750.pt", map_location="cpu", weights_only=True)
agent.actor.load_state_dict(ckpt["actor"])
agent.critic.load_state_dict(ckpt["critic"])
agent.termination.load_state_dict(ckpt["termination"])
agent.manager.load_state_dict(ckpt["manager"])

agent.actor.eval()
agent.critic.eval()
agent.termination.eval()
agent.manager.eval()

# Single environment instance. `reset()` re-randomizes victims but keeps the obstacles from `seed=42`
env = SAREnvironment(cfg, seed=cfg.seed)

async def handler(websocket):
    logger.info("Client connected")
    steps = 0
    obs = None
    done = False
    
    async for message in websocket:
        data = json.loads(message)
        cmd = data.get("type")
        
        if cmd == "reset":
            obs = env.reset()
            # Select initial options via Manager network
            agent.select_options(obs, step=0)
            steps = 0
            done = False
            
            resp = {
                "type": "reset_state",
                "gridW": cfg.grid_width,
                "gridH": cfg.grid_height,
                "obstacles": env.obstacles.tolist(),
                "victims": [
                    {
                        "x": int(env.victim_pos[i][0]),
                        "y": int(env.victim_pos[i][1]),
                        "found": bool(env.victims_found[i])
                    }
                    for i in range(cfg.n_victims)
                ],
                "agents": [
                    {
                        "id": i,
                        "x": int(env.agent_pos[i][0]),
                        "y": int(env.agent_pos[i][1]),
                        "option": int(agent.current_options[i].item())
                    }
                    for i in range(cfg.n_agents)
                ],
                "belief": env.belief.grid.tolist()
            }
            await websocket.send(json.dumps(resp))
            logger.info("Sent reset state")
            
        elif cmd == "step":
            if done:
                await websocket.send(json.dumps({"type": "done"}))
                continue
                
            positions = env.get_agent_positions()
            
            # 1. Forward pass actor to select low-level navigation actions
            with torch.no_grad():
                actions, _, _, _, _, _ = agent.select_actions(obs, positions)
            
            # 2. Step environment
            obs, _, curr_done, info = env.step(actions)
            steps += 1
            
            # 3. Check for high-level option termination
            terminate = agent.check_termination(obs, env.get_agent_positions(), steps)
            if np.any(terminate) and not curr_done:
                # 4. For agents that terminated, sample new options from Manager
                agent.select_options(obs, steps)
                
            done = curr_done
            
            resp = {
                "type": "step_state",
                "step": steps,
                "done": done,
                "victims": [
                    {
                        "x": int(env.victim_pos[i][0]),
                        "y": int(env.victim_pos[i][1]),
                        "found": bool(env.victims_found[i])
                    }
                    for i in range(cfg.n_victims)
                ],
                "agents": [
                    {
                        "id": i,
                        "x": int(env.agent_pos[i][0]),
                        "y": int(env.agent_pos[i][1]),
                        "option": int(agent.current_options[i].item())
                    }
                    for i in range(cfg.n_agents)
                ],
                "belief": env.belief.grid.tolist()
            }
            await websocket.send(json.dumps(resp))

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        logger.info("WebSocket server listening on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
