"""
Configuration for Hierarchical MARL SAR framework.
All hyperparameters in one place.
"""
import sys
sys.path.insert(0, r"D:\torch_pkg")

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # -- Environment ----------------------------------------------------------
    grid_width: int = 20              # grid columns
    grid_height: int = 20             # grid rows
    n_agents: int = 5                 # number of agents
    n_victims: int = 3                # number of victims
    n_obstacles: int = 40             # number of obstacle cells (fixed per env)
    obs_radius: int = 3               # local observation radius
    detection_radius: int = 1         # victim detection radius
    max_episode_steps: int = 200      # episode length cap

    # ── Hierarchical ─────────────────────────────────────────────
    n_options: int = 3                # {Explore=0, Navigate=1, Form=2}
    k_min: int = 5                    # min steps before option can terminate
    k_max: int = 30                   # max steps before forced termination

    # ── Graph ────────────────────────────────────────────────────
    proximity_radius: float = 5.0     # edge if ||p_i - p_j|| <= r
    gnn_layers: int = 2               # message-passing layers

    # ── Network dims ─────────────────────────────────────────────
    obs_dim: int = 0                  # set dynamically after env init
    node_feat_dim: int = 0            # obs_dim + n_options (one-hot)
    edge_feat_dim: int = 3            # [dx, dy, dist]
    hidden_dim: int = 128
    action_dim: int = 5               # {up, down, left, right, stay}

    # ── Actor-Critic ─────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_manager: float = 3e-4
    lr_termination: float = 3e-4
    mini_batch_size: int = 64

    # ── Reward shaping ───────────────────────────────────────────
    lambda_time: float = 0.01        # time penalty weight
    lambda_switch: float = 0.1      # switching penalty weight
    w_explore: float = 1.0           # exploration potential weight
    w_navigate: float = 0.5          # navigation potential weight
    w_formation: float = 0.3         # formation potential weight

    # ── Training ─────────────────────────────────────────────────
    train_episodes: int = 500
    eval_episodes: int = 50
    rollout_steps: int = 128          # steps per rollout before update
    log_interval: int = 10
    save_interval: int = 50
    seed: int = 42
    eval_seed: int = 12345

    def __post_init__(self):
        """Compute derived dimensions."""
        # obs = local belief grid (flattened) + local obstacle grid (flattened)
        #       + own position (2)
        view_size = (2 * self.obs_radius + 1) ** 2
        self.obs_dim = view_size * 2 + 2   # belief channel + obstacle channel + pos
        self.node_feat_dim = self.obs_dim + self.n_options
