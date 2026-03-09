"""
Interaction graph construction for GNN Actor / Critic / Termination.
"""
import numpy as np
import torch


def build_graph(positions: np.ndarray, radius: float):
    """
    Build an interaction graph from agent positions.

    Args:
        positions: (N, 2) array of agent positions.
        radius:    proximity threshold for edges.

    Returns:
        edge_index: (2, E) LongTensor — source/target node indices.
        edge_feat:  (E, 3) FloatTensor — [dx, dy, dist].
        adj_list:   dict mapping node → list of neighbor nodes.
    """
    n = len(positions)
    sources, targets = [], []
    edge_feats = []
    adj_list = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = positions[j] - positions[i]
            dist = np.linalg.norm(diff)
            if dist <= radius:
                sources.append(i)
                targets.append(j)
                edge_feats.append([diff[0], diff[1], dist])
                adj_list[i].append(j)

    if len(sources) == 0:
        # No edges — return empty tensors
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feat = torch.zeros((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_feat = torch.tensor(edge_feats, dtype=torch.float32)

    return edge_index, edge_feat, adj_list


def get_node_features(observations: np.ndarray, options: np.ndarray,
                      n_options: int) -> torch.Tensor:
    """
    Concatenate observations with one-hot option encoding.

    Args:
        observations: (N, obs_dim) array.
        options: (N,) int array of current option indices.
        n_options: total number of options.

    Returns:
        node_feat: (N, obs_dim + n_options) FloatTensor.
    """
    n = len(observations)
    one_hot = np.zeros((n, n_options), dtype=np.float32)
    for i in range(n):
        one_hot[i, int(options[i])] = 1.0
    feat = np.concatenate([observations, one_hot], axis=1)
    return torch.tensor(feat, dtype=torch.float32)
