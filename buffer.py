"""
Rollout buffer for PPO training.
Stores transitions and computes GAE advantages.
"""
import torch
import numpy as np


class RolloutBuffer:
    """
    Stores rollout data for one update cycle.
    Handles both low-level (per-step) and high-level (per-option) data.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        # Low-level data (per timestep)
        self.obs = []               # (N, obs_dim)
        self.node_feats = []        # (N, node_feat_dim)
        self.edge_indices = []      # (2, E)
        self.edge_feats = []        # (E, 3)
        self.actions = []           # (N,)
        self.log_probs = []         # (N,)
        self.rewards = []           # (N,)
        self.values = []            # scalar
        self.dones = []             # bool
        self.options = []           # (N,)

        # High-level data (per option boundary)
        self.hl_obs = []            # obs at option start
        self.hl_options = []        # chosen option
        self.hl_log_probs = []      # log_prob of option
        self.hl_rewards = []        # R^H_k
        self.hl_values = []         # V(s_{t_k})
        self.hl_dones = []          # done at option end

    def add_step(self, obs, node_feat, edge_index, edge_feat,
                 actions, log_probs, rewards, value, done, options):
        self.obs.append(obs)
        self.node_feats.append(node_feat)
        self.edge_indices.append(edge_index)
        self.edge_feats.append(edge_feat)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(value)
        self.dones.append(done)
        self.options.append(options)

    def add_hl_transition(self, obs, option, log_prob, reward, value, done):
        self.hl_obs.append(obs)
        self.hl_options.append(option)
        self.hl_log_probs.append(log_prob)
        self.hl_rewards.append(reward)
        self.hl_values.append(value)
        self.hl_dones.append(done)

    # ── GAE for low-level ────────────────────────────────────────
    def compute_low_level_gae(self, last_value: float, gamma: float,
                              gae_lambda: float):
        """
        Compute per-agent returns and advantages using GAE.
        Since the critic produces a single shared value, advantages are
        broadcast to all agents.
        """
        n_steps = len(self.rewards)
        if n_steps == 0:
            return [], []

        n_agents = self.rewards[0].shape[0]
        # Mean reward across agents for global advantage
        rewards = np.array([r.mean() for r in self.rewards])
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones, dtype=np.float32)

        advantages = np.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    # ── GAE for high-level ───────────────────────────────────────
    def compute_high_level_gae(self, last_value: float, gamma: float,
                               gae_lambda: float):
        n_steps = len(self.hl_rewards)
        if n_steps == 0:
            return [], []

        rewards = np.array(self.hl_rewards)
        values = np.array(self.hl_values + [last_value])
        dones = np.array(self.hl_dones, dtype=np.float32)

        advantages = np.zeros(n_steps)
        gae = 0.0
        for t in reversed(range(n_steps)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns
