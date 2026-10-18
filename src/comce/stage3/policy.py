from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def edge_key(edge: dict[str, Any]) -> str:
    return str(edge['canonical_key'])


def selected_community_membership(record: dict[str, Any], selected_community_ids: list[str]) -> dict[str, list[str]]:
    edge_keys = [str(key) for key in record['edge_keys']]
    selected = {str(rid) for rid in selected_community_ids}
    membership: dict[str, list[str]] = {}
    for community in record.get('communities', []):
        community_id = str(community.get('community_id'))
        if community_id not in selected:
            continue
        for member_index in community.get('member_indices', []):
            idx = int(member_index)
            if 0 <= idx < len(edge_keys):
                membership.setdefault(edge_keys[idx], []).append(community_id)
    return membership


class Policy(nn.Module):
    """Planner pi_xi: ADD(e) logits for every legal edge plus one STOP logit."""

    def __init__(self, edge_in_dim: int, hidden: int, center_state_dim: int, ctx_extra: int, budget: int) -> None:
        super().__init__()
        self.edge_in_dim = int(edge_in_dim)
        self.hidden = int(hidden)
        self.center_state_dim = int(center_state_dim)
        self.ctx_extra = int(ctx_extra)
        self.budget = int(budget)
        self.ctx_dim = self.center_state_dim + 2 * self.hidden + self.ctx_extra
        self.edge_encoder = nn.Sequential(nn.Linear(self.edge_in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.add_head = nn.Sequential(nn.Linear(self.ctx_dim + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.stop_head = nn.Sequential(nn.Linear(self.ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, edge_inputs: torch.Tensor, center_state: torch.Tensor, selected_mask: torch.Tensor,
                community_score_ctx: torch.Tensor) -> torch.Tensor:
        m = int(edge_inputs.shape[0])
        h = self.edge_encoder(edge_inputs) if m > 0 else torch.zeros(0, self.hidden, device=center_state.device)
        if m > 0 and bool(selected_mask.any()):
            selected_h = h[selected_mask]
            subset_mean = selected_h.mean(dim=0)
            subset_max = selected_h.max(dim=0).values
        else:
            subset_mean = torch.zeros(self.hidden, device=center_state.device)
            subset_max = torch.zeros(self.hidden, device=center_state.device)
        n_selected = float(selected_mask.sum().item())
        extras = torch.cat([torch.as_tensor([n_selected / max(1, self.budget)], dtype=torch.float32, device=center_state.device),
                            torch.as_tensor([float(np.log1p(m))], dtype=torch.float32, device=center_state.device),
                            community_score_ctx.to(dtype=torch.float32, device=center_state.device).reshape(-1)[:self.ctx_extra - 2]])
        ctx = torch.cat([center_state.to(dtype=torch.float32), subset_mean, subset_max, extras])
        if m > 0:
            add_logits = self.add_head(torch.cat([h, ctx.unsqueeze(0).expand(m, -1)], dim=-1)).squeeze(-1)
            add_logits = add_logits.masked_fill(selected_mask, float('-inf'))
        else:
            add_logits = torch.zeros(0, dtype=torch.float32, device=center_state.device)
        stop_logit = self.stop_head(ctx).squeeze(-1).unsqueeze(0)
        return torch.cat([add_logits, stop_logit])


def prepare_policy_inputs(record: dict[str, Any], domain: dict[str, Any]) -> dict:
    domain_edges = sorted((dict(edge) for edge in domain['macro_domain_edges']), key=edge_key)
    domain_keys = [edge_key(edge) for edge in domain_edges]
    position = {str(key): idx for idx, key in enumerate(record['edge_keys'])}
    rows = np.asarray(record['edge_rows'], dtype=np.float32)
    selected_community_ids = [str(rid) for rid in domain.get('selected_community_ids', [])]
    selection_score_by_id = {str(community.get('community_id')): float(community.get('selection_score', 0.0))
                             for community in domain.get('selected_communities', [])}
    membership = selected_community_membership(record, selected_community_ids)
    n_containing = np.zeros(len(domain_keys), dtype=np.float32)
    max_score = np.zeros(len(domain_keys), dtype=np.float32)
    for pos, key in enumerate(domain_keys):
        community_ids = membership.get(key, [])
        n_containing[pos] = float(len(community_ids))
        if community_ids:
            max_score[pos] = max(selection_score_by_id.get(rid, 0.0) for rid in community_ids)
    edge_features = (rows[np.asarray([position[key] for key in domain_keys], dtype=np.int64)] if domain_keys
                     else np.zeros((0, rows.shape[1]), dtype=np.float32))
    edge_inputs = np.concatenate([edge_features, n_containing[:, None], max_score[:, None]], axis=1).astype(np.float32)
    center_state = np.asarray(record['center_state'], dtype=np.float32).reshape(-1)
    selected_scores = [selection_score_by_id[rid] for rid in selected_community_ids if rid in selection_score_by_id]
    mean_score = float(np.mean(selected_scores)) if selected_scores else 0.0
    community_score_ctx = np.asarray([len(selected_community_ids) / 10.0, mean_score], dtype=np.float32)
    witness_keys = [str(key) for key in domain.get('witness_edge_keys', record.get('witness_edge_keys', []))]
    return {'edge_inputs': edge_inputs, 'edge_keys': domain_keys, 'edges': domain_edges, 'center_state': center_state,
            'community_score_ctx': community_score_ctx, 'witness_keys': witness_keys}


class PolicyLogits:
    """Frozen-policy logits for a fixed domain, cached per selected prefix."""
    # Trade-off: edge encodings are computed once per domain and logits are memoized per prefix; memory is bounded by
    # stage3.rollouts x stage3.explanation_budget distinct prefixes.

    @torch.no_grad()
    def __init__(self, policy, inputs):
        self.policy = policy
        self.keys = list(inputs['edge_keys'])
        self.index = {key: i for i, key in enumerate(self.keys)}
        device = next(policy.parameters()).device
        self.edge = torch.as_tensor(inputs['edge_inputs'], dtype=torch.float32, device=device)
        self.center = torch.as_tensor(inputs['center_state'], dtype=torch.float32, device=device)
        self.community = torch.as_tensor(inputs['community_score_ctx'], dtype=torch.float32, device=device)
        self.embedding = policy.edge_encoder(self.edge)
        self._logits_cache = {}
        self.policy_calls = 0
        self.policy_cache_hits = 0

    @torch.no_grad()
    def __call__(self, selected):
        self.policy_calls += 1
        key = tuple(sorted(selected))
        cached = self._logits_cache.get(key)
        if cached is not None:
            self.policy_cache_hits += 1
            return cached.copy()
        policy, h = self.policy, self.embedding
        m = len(self.keys)
        mask = torch.zeros(m, dtype=torch.bool, device=h.device)
        mask[[self.index[k] for k in selected]] = True
        if selected:
            mean, maximum = h[mask].mean(0), h[mask].max(0).values
        else:
            mean, maximum = torch.zeros(policy.hidden, device=h.device), torch.zeros(policy.hidden, device=h.device)
        extras = torch.cat([
            torch.as_tensor([len(selected) / max(1, policy.budget), float(np.log1p(m))], dtype=torch.float32, device=h.device),
            self.community.reshape(-1)[:policy.ctx_extra - 2]])
        context = torch.cat([self.center, mean, maximum, extras])
        adds = policy.add_head(torch.cat([h, context[None].expand(m, -1)], -1)).squeeze(-1)
        adds = adds.masked_fill(mask, -torch.inf)
        stop = policy.stop_head(context).reshape(1)
        logits = torch.cat([adds, stop]).detach().double().cpu().numpy()
        self._logits_cache[key] = logits.copy()
        return logits
