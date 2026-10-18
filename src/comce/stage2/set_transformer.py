from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _coerce_center_id(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


class SignatureEmbedding(nn.Module):

    def __init__(self, signature_k: int, signature_m: int, embed_dim: int) -> None:
        super().__init__()
        self.signature_k = int(signature_k)
        self.signature_m = int(signature_m)
        self.embed_dim = int(embed_dim)
        self.emb = nn.Embedding(self.signature_k * self.signature_m, self.embed_dim)

    def forward(self, signatures: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.signature_k, device=signatures.device).view(*[1] * (signatures.ndim - 1), self.signature_k)
        return self.emb(signatures + offsets * self.signature_m).flatten(start_dim=-2)


class CommunityEdgeReadout(nn.Module):

    def __init__(self, *, edge_feature_dim: int, readout_dim: int, d_model: int, locality_lambda_init: float,
                 receptive_radius: float) -> None:
        super().__init__()
        self.edge_feature_dim = int(edge_feature_dim)
        self.readout_dim = int(readout_dim)
        self.d_model = int(d_model)
        self.receptive_radius = float(receptive_radius)
        self.edge_mlp = nn.Sequential(nn.Linear(self.edge_feature_dim, self.readout_dim), nn.ReLU(),
                                      nn.Linear(self.readout_dim, self.readout_dim), nn.ReLU())
        self.out_proj = nn.Linear(3 * self.readout_dim, self.d_model)
        lam0 = max(1e-08, float(locality_lambda_init))
        self.locality_lambda_raw = nn.Parameter(torch.tensor(math.log(math.expm1(lam0))))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, member_feats: torch.Tensor, member_hops: torch.Tensor, segment_ids: torch.Tensor,
                n_communities: int) -> torch.Tensor:
        n_communities = int(n_communities)
        if member_feats.shape[0] == 0:
            return member_feats.new_zeros((n_communities, self.d_model))
        h = self.edge_mlp(member_feats)
        d = h.shape[1]
        seg = segment_ids.long()
        lam = F.softplus(self.locality_lambda_raw)
        hop_normalized = member_hops.to(h.dtype) / max(1e-06, self.receptive_radius)
        omega = torch.exp(-lam * hop_normalized)
        ones = torch.ones_like(omega)
        count = torch.zeros(n_communities, dtype=h.dtype, device=h.device).index_add_(0, seg, ones)
        mean = torch.zeros(n_communities, d, dtype=h.dtype, device=h.device).index_add_(0, seg, h)
        mean = mean / count.clamp_min(1.0).unsqueeze(-1)
        idx = seg.unsqueeze(-1).expand_as(h)
        maxp = h.new_full((n_communities, d), float('-inf'))
        maxp = maxp.scatter_reduce(0, idx, h, reduce='amax', include_self=True)
        maxp = torch.where(count.unsqueeze(-1) > 0, maxp, torch.zeros_like(maxp))
        wsum = torch.zeros(n_communities, dtype=h.dtype, device=h.device).index_add_(0, seg, omega)
        wmean = torch.zeros(n_communities, d, dtype=h.dtype, device=h.device).index_add_(0, seg, h * omega.unsqueeze(-1))
        wmean = wmean / wsum.clamp_min(1e-08).unsqueeze(-1)
        return self.out_proj(torch.cat([mean, maxp, wmean], dim=-1))


class SetTransformer(nn.Module):
    """Community selector F_Theta: K query slots attend over community tokens."""

    def __init__(self, *, signature_k: int, signature_m: int, signature_embed_dim: int, center_state_dim: int,
                 community_feature_dim: int, num_pattern_slots: int, d_model: int, macro_hidden_dim: int, num_heads: int,
                 dropout: float, edge_feature_dim: int, readout_dim: int, locality_lambda_init: float,
                 receptive_radius: float) -> None:
        super().__init__()
        self.signature_k = int(signature_k)
        self.signature_m = int(signature_m)
        self.num_pattern_slots = int(num_pattern_slots)
        self.d_model = int(d_model)
        self.community_feature_dim = int(community_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        sig_dim = int(signature_k * signature_embed_dim)
        self.center_signature_emb = SignatureEmbedding(signature_k, signature_m, signature_embed_dim)
        self.macro_signature_emb = SignatureEmbedding(signature_k, signature_m, signature_embed_dim)
        self.center_encoder = nn.Sequential(nn.Linear(sig_dim + center_state_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
                                            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.macro_encoder = nn.Sequential(nn.Linear(sig_dim + community_feature_dim, macro_hidden_dim),
                                           nn.LayerNorm(macro_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                                           nn.Linear(macro_hidden_dim, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.macro_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.macro_norm = nn.LayerNorm(d_model)
        self.slot_embeddings = nn.Parameter(torch.randn(num_pattern_slots, d_model) * 0.02)
        self.center_to_slots = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pattern_norm = nn.LayerNorm(d_model)
        self.pattern_score_head = nn.Linear(d_model, 1)
        self.membership_key = nn.Linear(d_model, d_model)
        self.membership_query = nn.Linear(d_model, d_model)
        self.macro_score_bias = nn.Linear(d_model, 1)
        self.edge_readout = CommunityEdgeReadout(edge_feature_dim=edge_feature_dim, readout_dim=readout_dim, d_model=d_model,
                                                 locality_lambda_init=locality_lambda_init, receptive_radius=receptive_radius)

    def forward(self, center_signature, center_state, community_signatures, community_features, community_mask,
                member_feats, member_hops, segment_ids) -> dict[str, Any]:
        center_sig = self.center_signature_emb(center_signature)
        macro_sig = self.macro_signature_emb(community_signatures)
        center_hidden = self.center_encoder(torch.cat([center_sig, center_state], dim=-1))
        macro_hidden = self.macro_encoder(torch.cat([macro_sig, community_features], dim=-1))
        batch, rmax = (int(community_signatures.shape[0]), int(community_signatures.shape[1]))
        h_edge = self.edge_readout(member_feats, member_hops, segment_ids, batch * rmax)
        h_edge = h_edge.view(batch, rmax, self.d_model)
        h_edge = h_edge * community_mask.unsqueeze(-1).to(h_edge.dtype)
        macro_hidden = macro_hidden + self.edge_readout.gamma * h_edge
        self_attended, _ = self.macro_self_attn(macro_hidden, macro_hidden, macro_hidden, key_padding_mask=~community_mask,
                                                need_weights=False)
        macro_hidden = self.macro_norm(macro_hidden + self_attended)
        slot_base = self.slot_embeddings.unsqueeze(0).expand(center_hidden.shape[0], -1, -1)
        slot_cond = self.center_to_slots(center_hidden).unsqueeze(1)
        Q = self.q_proj(slot_base + slot_cond)
        K = self.k_proj(macro_hidden)
        V = self.v_proj(macro_hidden)
        scale = math.sqrt(float(self.d_model))
        attention_logits = torch.einsum('bsd,btd->bst', Q, K) / max(scale, 1e-06)
        attention_logits = attention_logits.masked_fill(~community_mask.unsqueeze(1), -1000000000.0)
        alpha_attention = torch.softmax(attention_logits, dim=-1)
        slot_context = torch.einsum('bst,btd->bsd', alpha_attention, V)
        pattern_context = self.pattern_norm(slot_context + Q)
        pattern_logits = self.pattern_score_head(pattern_context).squeeze(-1)
        membership_q = self.membership_query(pattern_context)
        membership_k = self.membership_key(macro_hidden)
        membership_logits = torch.einsum('bsd,btd->bst', membership_q, membership_k) / max(scale, 1e-06)
        membership_logits = membership_logits.masked_fill(~community_mask.unsqueeze(1), -30.0)
        q_membership = torch.sigmoid(membership_logits)
        pattern_community_scores = membership_logits + pattern_logits.unsqueeze(-1)
        macro_logits = torch.logsumexp(pattern_community_scores, dim=1) + self.macro_score_bias(macro_hidden).squeeze(-1)
        macro_logits = macro_logits.masked_fill(~community_mask, -30.0)
        return {'pattern_scores': torch.sigmoid(pattern_logits), 'q_membership': q_membership, 'alpha_attention': alpha_attention,
                'macro_logits': macro_logits, 'p_community': torch.sigmoid(macro_logits), 'community_hidden': macro_hidden}


BATCH_TENSOR_KEYS = ('center_signature', 'center_state', 'community_signatures', 'community_features', 'community_mask',
                     'witness_mask', 'edge_mask', 'community_pos', 'member_feats', 'member_hops', 'segment_ids')


def collate_center_records(records: list[dict[str, Any]], *, community_feature_dim: int, edge_feature_dim: int) -> dict[str, Any]:
    if not records:
        raise ValueError('collate_center_records requires at least one record')
    batch = len(records)
    n_communities = [len(r['communities']) for r in records]
    n_edges = [len(r['edge_keys']) for r in records]
    rmax = max(1, max(n_communities))
    nmax = max(1, max(n_edges))
    sig_k = int(np.asarray(records[0]['center_sig']).shape[0])
    state_dim = int(np.asarray(records[0]['center_state']).shape[0])
    feat_dim = int(community_feature_dim)
    for r in records:
        if len(r['communities']):
            feat_dim = int(np.asarray(r['community_feats']).shape[1])
            break
    center_signature = np.zeros((batch, sig_k), dtype=np.int64)
    center_state = np.zeros((batch, state_dim), dtype=np.float32)
    community_signatures = np.zeros((batch, rmax, sig_k), dtype=np.int64)
    community_features = np.zeros((batch, rmax, feat_dim), dtype=np.float32)
    community_mask = np.zeros((batch, rmax), dtype=bool)
    witness_mask = np.zeros((batch, nmax), dtype=bool)
    edge_mask = np.zeros((batch, nmax), dtype=bool)
    community_pos = np.zeros((batch, rmax), dtype=np.float32)
    member_feats, member_hops, segment_ids, inc_community, inc_edge = [], [], [], [], []
    for b, rec in enumerate(records):
        n_r, n_e = (n_communities[b], n_edges[b])
        center_signature[b] = np.asarray(rec['center_sig'], dtype=np.int64)
        center_state[b] = np.asarray(rec['center_state'], dtype=np.float32)
        if n_r:
            community_signatures[b, :n_r] = np.asarray(rec['community_sigs'], dtype=np.int64)
            community_features[b, :n_r] = np.asarray(rec['community_feats'], dtype=np.float32)
            community_mask[b, :n_r] = True
        edge_mask[b, :n_e] = True
        witness_set = {int(i) for i in rec['witness_indices']}
        for e in witness_set:
            witness_mask[b, e] = True
        rows = np.asarray(rec['edge_rows'], dtype=np.float32)
        hops = np.asarray(rec['hops'], dtype=np.int64)
        for r, community in enumerate(rec['communities']):
            mi = [int(i) for i in community['member_indices']]
            flat_community = b * rmax + r
            member_feats.append(rows[mi])
            member_hops.append(hops[mi])
            segment_ids.append(np.full(len(mi), flat_community, dtype=np.int64))
            inc_community.extend([flat_community] * len(mi))
            inc_edge.extend([b * nmax + int(e) for e in mi])
            if witness_set.intersection(mi):
                community_pos[b, r] = 1.0
    member_feats_arr = np.concatenate(member_feats, axis=0) if member_feats else np.zeros((0, int(edge_feature_dim)), dtype=np.float32)
    member_hops_arr = np.concatenate(member_hops, axis=0) if member_hops else np.zeros((0,), dtype=np.int64)
    segment_arr = np.concatenate(segment_ids, axis=0) if segment_ids else np.zeros((0,), dtype=np.int64)
    return {'center_signature': torch.from_numpy(center_signature), 'center_state': torch.from_numpy(center_state),
            'community_signatures': torch.from_numpy(community_signatures), 'community_features': torch.from_numpy(community_features),
            'community_mask': torch.from_numpy(community_mask), 'witness_mask': torch.from_numpy(witness_mask),
            'edge_mask': torch.from_numpy(edge_mask), 'community_pos': torch.from_numpy(community_pos),
            'member_feats': torch.from_numpy(member_feats_arr.astype(np.float32)), 'member_hops': torch.from_numpy(member_hops_arr),
            'segment_ids': torch.from_numpy(segment_arr),
            'incidence': {'community_index': torch.as_tensor(np.asarray(inc_community, dtype=np.int64)),
                          'edge_index': torch.as_tensor(np.asarray(inc_edge, dtype=np.int64)),
                          'batch_size': batch, 'rmax': rmax, 'n_edges_max': nmax},
            'n_communities': n_communities, 'n_edges': n_edges,
            'center_ids': [_coerce_center_id(r['center_id']) for r in records]}


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in BATCH_TENSOR_KEYS:
        out[key] = out[key].to(device)
    incidence = dict(out['incidence'])
    incidence['community_index'] = incidence['community_index'].to(device)
    incidence['edge_index'] = incidence['edge_index'].to(device)
    out['incidence'] = incidence
    return out


def model_forward(model: SetTransformer, batch: dict[str, Any]) -> dict[str, Any]:
    return model(batch['center_signature'], batch['center_state'], batch['community_signatures'],
                 batch['community_features'], batch['community_mask'], batch['member_feats'], batch['member_hops'],
                 batch['segment_ids'])


def selector_probabilities(selector: SetTransformer, record, device):
    batch = move_batch(collate_center_records([record], community_feature_dim=selector.community_feature_dim,
                                              edge_feature_dim=selector.edge_feature_dim), torch.device(device))
    with torch.no_grad():
        out = model_forward(selector, batch)
    return out['p_community'][0, :len(record['communities'])].detach().cpu().numpy()
