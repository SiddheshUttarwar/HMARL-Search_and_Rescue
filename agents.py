"""
Hierarchical MARL Agent.
Orchestrates: Manager (HighLevelPolicy) → Worker (GNNActor)
              + GraphCritic + TerminationNetwork.
Runs PPO updates for all four networks.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config
from models import GNNActor, GraphCritic, TerminationNetwork, HighLevelPolicy
from graph import build_graph, get_node_features
from buffer import RolloutBuffer


class HierarchicalMARLAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        # ── Networks ─────────────────────────────────────────────
        self.actor = GNNActor(cfg.node_feat_dim, cfg.edge_feat_dim,
                              cfg.hidden_dim, cfg.action_dim, cfg.gnn_layers)
        self.critic = GraphCritic(cfg.node_feat_dim, cfg.edge_feat_dim,
                                  cfg.hidden_dim, cfg.gnn_layers)
        self.termination = TerminationNetwork(cfg.node_feat_dim, cfg.edge_feat_dim,
                                              cfg.hidden_dim, cfg.gnn_layers)
        self.manager = HighLevelPolicy(cfg.obs_dim, cfg.hidden_dim, cfg.n_options)

        # ── Optimisers ───────────────────────────────────────────
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.opt_term = optim.Adam(self.termination.parameters(),
                                   lr=cfg.lr_termination)
        self.opt_manager = optim.Adam(self.manager.parameters(), lr=cfg.lr_manager)

        # ── State ────────────────────────────────────────────────
        self.current_options = np.zeros(cfg.n_agents, dtype=np.int64)
        self.option_start_step = np.zeros(cfg.n_agents, dtype=np.int64)
        self.prev_options = np.zeros(cfg.n_agents, dtype=np.int64)
        self.buffer = RolloutBuffer()

    # ── Option selection ─────────────────────────────────────────
    @torch.no_grad()
    def select_options(self, obs: np.ndarray, step: int):
        """Select new options for all agents via the manager."""
        obs_t = torch.tensor(obs, dtype=torch.float32)
        options, log_probs, entropy = self.manager.get_option(obs_t)
        self.prev_options = self.current_options.copy()
        self.current_options = options.numpy()
        self.option_start_step[:] = step
        return options.numpy(), log_probs.numpy(), entropy.numpy()

    # ── Action selection ─────────────────────────────────────────
    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, positions: np.ndarray):
        """
        Build graph, compute node features, run GNNActor.
        Returns actions, log_probs, value, node_feat, edge_index, edge_feat.
        """
        cfg = self.cfg
        # Build graph
        edge_index, edge_feat, _ = build_graph(positions, cfg.proximity_radius)
        node_feat = get_node_features(obs, self.current_options, cfg.n_options)

        # Actor
        actions, log_probs, entropy = self.actor.get_action(
            node_feat, edge_index, edge_feat)

        # Critic
        value = self.critic(node_feat, edge_index, edge_feat).item()

        return (actions.numpy(), log_probs.numpy(), value,
                node_feat, edge_index, edge_feat)

    # ── Termination check ────────────────────────────────────────
    @torch.no_grad()
    def check_termination(self, obs: np.ndarray, positions: np.ndarray,
                          step: int):
        """
        Returns boolean array indicating which agents terminate their option.
        Respects k_min / k_max constraints.
        """
        cfg = self.cfg
        edge_index, edge_feat, _ = build_graph(positions, cfg.proximity_radius)
        node_feat = get_node_features(obs, self.current_options, cfg.n_options)

        beta = self.termination(node_feat, edge_index, edge_feat)  # (N,)
        beta_np = beta.numpy()

        duration = step - self.option_start_step
        terminate = np.zeros(cfg.n_agents, dtype=bool)
        for i in range(cfg.n_agents):
            if duration[i] >= cfg.k_max:
                terminate[i] = True
            elif duration[i] >= cfg.k_min:
                terminate[i] = np.random.rand() < beta_np[i]
        return terminate

    # ── High-level reward R^H_k ──────────────────────────────────
    def compute_high_level_reward(self, old_potential: float,
                                  new_potential: float, dt: int):
        cfg = self.cfg
        switch_penalty = cfg.lambda_switch * np.sum(
            self.current_options != self.prev_options)
        rh = (new_potential - old_potential
              - cfg.lambda_time * dt
              - switch_penalty)
        return rh

    # ══════════════════════════════════════════════════════════════
    # Actor-Critic Updates
    # ══════════════════════════════════════════════════════════════
    def update(self):
        """Run A2C updates for actor, critic, manager, and termination."""
        cfg = self.cfg
        buf = self.buffer

        # ── Low-level advantages ─────────────────────────────────
        # Get last value
        if len(buf.node_feats) > 0:
            last_nf = buf.node_feats[-1]
            last_ei = buf.edge_indices[-1]
            last_ef = buf.edge_feats[-1]
            with torch.no_grad():
                last_val = self.critic(last_nf, last_ei, last_ef).item()
        else:
            last_val = 0.0

        ll_adv, ll_ret = buf.compute_low_level_gae(
            last_val, cfg.gamma, cfg.gae_lambda)

        if len(ll_adv) == 0:
            self.buffer.clear()
            return {}

        ll_adv = torch.tensor(ll_adv, dtype=torch.float32)
        ll_ret = torch.tensor(ll_ret, dtype=torch.float32)
        # Normalise advantages
        if ll_adv.numel() > 1:
            ll_adv = (ll_adv - ll_adv.mean()) / (ll_adv.std() + 1e-8)

        # ── Low-level Actor-Critic ───────────────────────────────────
        actor_losses, critic_losses = [], []
        # Single epoch over collected rollout data (A2C)
        for t in range(len(buf.obs)):
            nf = buf.node_feats[t]
            ei = buf.edge_indices[t]
            ef = buf.edge_feats[t]
            old_actions = torch.tensor(buf.actions[t], dtype=torch.long)
            # old_lp not needed for A2C surrogate directly (just compute new_lp and multiply by constant advantage)

            # Actor
            new_lp, ent = self.actor.evaluate(nf, ei, ef, old_actions)
            adv = ll_adv[t]
            actor_loss = -(new_lp * adv).mean() - cfg.entropy_coef * ent.mean()

            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                                     cfg.max_grad_norm)
            self.opt_actor.step()
            actor_losses.append(actor_loss.item())

            # Critic
            value = self.critic(nf, ei, ef)
            critic_loss = cfg.value_loss_coef * (value - ll_ret[t]) ** 2
            critic_loss = critic_loss.mean()

            self.opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     cfg.max_grad_norm)
            self.opt_critic.step()
            critic_losses.append(critic_loss.item())

        # ── High-level Actor-Critic (manager) ────────────────────────
        hl_adv, hl_ret = buf.compute_high_level_gae(
            0.0, cfg.gamma, cfg.gae_lambda)

        manager_losses = []
        if len(hl_adv) > 0:
            hl_adv_t = torch.tensor(hl_adv, dtype=torch.float32)
            if hl_adv_t.numel() > 1:
                hl_adv_t = (hl_adv_t - hl_adv_t.mean()) / (hl_adv_t.std() + 1e-8)

            # Single epoch for A2C
            for t in range(len(buf.hl_obs)):
                obs_t = torch.tensor(buf.hl_obs[t], dtype=torch.float32)
                opt_t = torch.tensor(buf.hl_options[t], dtype=torch.long)

                new_lp, ent = self.manager.evaluate(obs_t, opt_t)
                adv = hl_adv_t[t]

                loss = -(new_lp * adv).mean() - cfg.entropy_coef * ent.mean()

                self.opt_manager.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.manager.parameters(),
                                         cfg.max_grad_norm)
                self.opt_manager.step()
                manager_losses.append(loss.item())

        # ── Termination loss: L_β = E[β · A^H] ──────────────────
        # Use the high-level advantage at each termination point
        term_losses = []
        if len(hl_adv) > 0:
            for t in range(len(buf.hl_obs)):
                obs_t = torch.tensor(buf.hl_obs[t], dtype=torch.float32)
                # Rebuild graph for termination net
                # Use actor node features for convenience
                opt_idx = buf.hl_options[t]
                nf = get_node_features(buf.hl_obs[t],
                                       np.full(len(buf.hl_obs[t]),
                                               opt_idx if np.isscalar(opt_idx)
                                               else opt_idx[0]),
                                       cfg.n_options)
                # Dummy graph (no edges for termination update simplicity)
                ei = torch.zeros((2, 0), dtype=torch.long)
                ef = torch.zeros((0, 3), dtype=torch.float32)

                beta = self.termination(nf, ei, ef)   # (N,)
                adv = hl_adv_t[t] if len(hl_adv) > 0 else torch.tensor(0.0)
                t_loss = (beta * adv).mean()

                self.opt_term.zero_grad()
                t_loss.backward()
                nn.utils.clip_grad_norm_(self.termination.parameters(),
                                         cfg.max_grad_norm)
                self.opt_term.step()
                term_losses.append(t_loss.item())

        self.buffer.clear()
        return {
            "actor_loss": np.mean(actor_losses) if actor_losses else 0.0,
            "critic_loss": np.mean(critic_losses) if critic_losses else 0.0,
            "manager_loss": np.mean(manager_losses) if manager_losses else 0.0,
            "term_loss": np.mean(term_losses) if term_losses else 0.0,
        }
