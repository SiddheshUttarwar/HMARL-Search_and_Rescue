"""
Neural-network models for Hierarchical MARL.
All GNN layers are implemented in pure PyTorch (no torch_geometric).
  • MessagePassingLayer  — m_ij = ϕ(h_i, h_j, e_ij); h_i' = ψ(h_i, Σ m_ij)
  • GNNActor             — decentralised low-level policy
  • GraphCritic          — centralised CTDE value function
  • TerminationNetwork   — graph-conditioned β_t^i
  • HighLevelPolicy      — option manager π_H
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ═══════════════════════════════════════════════════════════════════
# Message-Passing GNN Layer
# ═══════════════════════════════════════════════════════════════════
class MessagePassingLayer(nn.Module):
    """
    m_ij = ϕ(h_i ‖ h_j ‖ e_ij)          (edge MLP)
    m_i  = Σ_{j∈N(i)} m_ij              (aggregation)
    h_i' = ψ(h_i ‖ m_i)                 (update MLP)
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """
        h:          (N, node_dim)
        edge_index: (2, E)
        edge_feat:  (E, edge_dim)
        Returns:    (N, hidden_dim)
        """
        n = h.size(0)
        hidden_dim = self.update_mlp[-1].out_features

        if edge_index.size(1) == 0:
            # No edges: aggregate is zero
            agg = torch.zeros(n, hidden_dim, device=h.device)
        else:
            src, tgt = edge_index[0], edge_index[1]
            h_src = h[src]       # (E, node_dim)
            h_tgt = h[tgt]       # (E, node_dim)
            inp = torch.cat([h_src, h_tgt, edge_feat], dim=-1)
            messages = self.edge_mlp(inp)            # (E, hidden)

            # Sum aggregation per target node
            agg = torch.zeros(n, hidden_dim, device=h.device)
            agg.scatter_add_(0, tgt.unsqueeze(1).expand_as(messages), messages)

        out = self.update_mlp(torch.cat([h, agg], dim=-1))
        return out


# ═══════════════════════════════════════════════════════════════════
# GNN backbone (stack of MP layers)
# ═══════════════════════════════════════════════════════════════════
class GNNBackbone(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int,
                 n_layers: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_feat)   # residual
        return h    # (N, hidden_dim)


# ═══════════════════════════════════════════════════════════════════
# GNN Actor — decentralised low-level policy  π_L(a | h_i')
# ═══════════════════════════════════════════════════════════════════
class GNNActor(nn.Module):
    def __init__(self, node_feat_dim: int, edge_dim: int,
                 hidden_dim: int, action_dim: int, n_layers: int):
        super().__init__()
        self.gnn = GNNBackbone(node_feat_dim, edge_dim, hidden_dim, n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        h = self.gnn(node_feat, edge_index, edge_feat)   # (N, H)
        logits = self.head(h)                             # (N, A)
        return logits

    def get_action(self, node_feat, edge_index, edge_feat):
        logits = self.forward(node_feat, edge_index, edge_feat)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, node_feat, edge_index, edge_feat, actions):
        logits = self.forward(node_feat, edge_index, edge_feat)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


# ═══════════════════════════════════════════════════════════════════
# Graph Critic — centralised value  V_ψ(s)
# ═══════════════════════════════════════════════════════════════════
class GraphCritic(nn.Module):
    def __init__(self, node_feat_dim: int, edge_dim: int,
                 hidden_dim: int, n_layers: int):
        super().__init__()
        self.gnn = GNNBackbone(node_feat_dim, edge_dim, hidden_dim, n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        h = self.gnn(node_feat, edge_index, edge_feat)   # (N, H)
        h_global = h.mean(dim=0, keepdim=True)            # (1, H) mean pool
        value = self.head(h_global).squeeze(-1)           # (1,)
        return value


# ═══════════════════════════════════════════════════════════════════
# Termination Network — β_t^i  = σ(MLP(GNN_β(o_i, neighbours)))
# ═══════════════════════════════════════════════════════════════════
class TerminationNetwork(nn.Module):
    def __init__(self, node_feat_dim: int, edge_dim: int,
                 hidden_dim: int, n_layers: int):
        super().__init__()
        self.gnn = GNNBackbone(node_feat_dim, edge_dim, hidden_dim, n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        h = self.gnn(node_feat, edge_index, edge_feat)   # (N, H)
        beta = torch.sigmoid(self.head(h)).squeeze(-1)    # (N,)
        return beta


# ═══════════════════════════════════════════════════════════════════
# High-Level Policy (Manager)  π_H(z | h_H)
# ═══════════════════════════════════════════════════════════════════
class HighLevelPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, n_options: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_options),
        )

    def forward(self, obs):
        return self.net(obs)   # (N, n_options)

    def get_option(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        option = dist.sample()
        return option, dist.log_prob(option), dist.entropy()

    def evaluate(self, obs, options):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(options), dist.entropy()
