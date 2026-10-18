import torch
from torch import nn

STATES = ('omega', 'omega_bar')


class WorldModel(nn.Module):
    """F_omega: pooled edge encodings and the set context c_{u,Delta} -> flip logit.

    omega (rollout scoring) and omega_bar (offline-adapted, ranking) are two parameter states of this one class.
    """

    def __init__(self, edge_in: int, set_dim: int, hidden: int) -> None:
        super().__init__()
        self.edge_in, self.set_dim, self.hidden = int(edge_in), int(set_dim), int(hidden)
        self.edge_enc = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.ctx_proj = nn.Sequential(nn.Linear(set_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head_flip = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode_edges(self, edge_rows: torch.Tensor) -> torch.Tensor:
        return self.edge_enc(edge_rows)

    def forward(self, edge_embeddings: torch.Tensor, set_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        avg = (edge_embeddings * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        maximum = (edge_embeddings + (~mask)[..., None] * -1000000000.0).amax(1)
        return self.head_flip(torch.cat([avg, maximum, self.ctx_proj(set_features)], 1)).squeeze(-1)


def world_model_states(edge_in: int, set_dim: int, hidden: int) -> nn.ModuleDict:
    """omega and omega_bar: the same architecture, two trained parameter sets."""
    return nn.ModuleDict({state: WorldModel(edge_in, set_dim, hidden) for state in STATES})
