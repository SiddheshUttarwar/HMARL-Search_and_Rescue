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
class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT v2 style).
    Computes attention scores α_ij for edges:
      e_ij = LeakyReLU( MLP_attn(h_i ‖ h_j ‖ e_ij) )
      α_ij = softmax_j(e_ij)
    Aggregates messages:
      m_i = Σ_{j∈N(i)} α_ij * (W_v h_j)
    Updates node:
      h_i' = MLP_out(h_i ‖ m_i)
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        # Attention scoring MLP
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        # Value projection
        self.value_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        
        # Update MLP
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
            
            # 1) Compute raw attention scores e_ij
            inp = torch.cat([h_tgt, h_src, edge_feat], dim=-1) # (E, 2*node_dim + edge_dim)
            e_ij = self.attn_mlp(inp).squeeze(-1)              # (E,)

            # 2) Softmax over neighborhoods
            # We must subtract max for numerical stability per target
            e_max = torch.zeros(n, device=h.device).scatter_reduce(
                0, tgt, e_ij, reduce="amax", include_self=False
            )
            e_ij_stable = e_ij - e_max[tgt]
            num = torch.exp(e_ij_stable)
            den = torch.zeros(n, device=h.device).scatter_add_(0, tgt, num)
            alpha_ij = num / (den[tgt] + 1e-8)                 # (E,)

            # 3) Value projection & Attention-weighted sum
            v_src = self.value_proj(h_src)                     # (E, hidden_dim)
            messages = alpha_ij.unsqueeze(-1) * v_src          # (E, hidden_dim)
            
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
            GraphAttentionLayer(hidden_dim, edge_dim, hidden_dim)
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
        
        # Global Attention Pooling components
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feat, edge_index, edge_feat):
        h = self.gnn(node_feat, edge_index, edge_feat)   # (N, H)
        
        # Global Attention Pooling
        attn_weights = self.pool_attn(h)                 # (N, 1)
        attn_scores = torch.softmax(attn_weights, dim=0) # (N, 1)
        h_global = torch.sum(attn_scores * h, dim=0, keepdim=True) # (1, H)
        
        value = self.head(h_global).squeeze(-1)          # (1,)
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
