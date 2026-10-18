import numpy as np
import torch
from torch import nn

MAX_SET_SIZE = 10


def pooled_set_descriptors(rows, mask, max_set_size=MAX_SET_SIZE):
    """psi(u, Delta): mean and max of the center-relative edge rows plus |Delta| / B."""
    weights = mask[..., None].to(rows.dtype)
    mean = (rows * weights).sum(1) / weights.sum(1).clamp_min(1)
    maximum = (rows + (~mask)[..., None] * -1000000000.0).amax(1)
    size = mask.sum(1, keepdim=True).to(rows.dtype) / float(max_set_size)
    return torch.cat([mean, maximum, size], 1)


class RankingResidual(nn.Module):
    """w_r^T psi(u, Delta) + b_r."""

    def __init__(self, edge_in: int, max_set_size: int = MAX_SET_SIZE) -> None:
        super().__init__()
        self.edge_in, self.max_set_size = int(edge_in), int(max_set_size)
        self.linear = nn.Linear(2 * self.edge_in + 1, 1)

    def forward(self, edge_rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.linear(pooled_set_descriptors(edge_rows, mask, self.max_set_size)).squeeze(-1)


class CandidateRanker:
    """S(u, Delta) = z_omega_bar(u, Delta) + w_r^T psi(u, Delta) + b_r for every candidate deletion set."""

    def __init__(self, scorer, chunk_size=4096):
        self.scorer = scorer
        self.chunk_size = chunk_size

    @torch.no_grad()
    def ranking_logits(self, indices, mask):
        return self.scorer.state_logits('omega_bar', indices, mask) + self.scorer.residual_logits(indices, mask)

    @torch.no_grad()
    def __call__(self, candidates):
        original = [tuple(sorted(candidate)) for candidate in candidates]
        unique = list(dict.fromkeys(original))
        if not unique:
            return np.empty(0, np.float64)
        sets = self.scorer.index_sets(unique)
        self.scorer.signatures.validate([self.scorer.keys[i] for row in sets for i in row])
        values = torch.cat([self.ranking_logits(*self.scorer.sets.pack(sets[begin:begin + self.chunk_size]))
                            for begin in range(0, len(sets), self.chunk_size)])
        scores = self.scorer.as_array(values, len(sets))
        position = {candidate: i for i, candidate in enumerate(unique)}
        return scores[[position[candidate] for candidate in original]]

    def contract(self):
        return dict(score='z_omega_bar(u,D)+w_r^T psi(u,D)+b_r', pointwise=True,
                    pool_scope='complete_supplied_pool_before_compute_chunks')
