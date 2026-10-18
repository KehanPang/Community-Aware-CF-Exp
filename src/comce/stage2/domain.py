from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DomainParams:
    """tau, the unique-edge cap of the admitted domain, the exponent on a community's marginal edit cost, and the
    domain size the soft fallback fills up to when no community passes tau."""
    relevance_threshold: float
    domain_edge_cap: int | None
    admission_exponent: float = 1.0
    fallback_min_domain_edges: int = 10

    def __post_init__(self):
        if not 0.0 <= float(self.relevance_threshold) <= 1.0:
            raise ValueError('relevance_threshold must be in [0,1]')
        if self.domain_edge_cap is not None and int(self.domain_edge_cap) <= 0:
            raise ValueError('domain_edge_cap must be positive or null')
        if not 0.0 <= float(self.admission_exponent) <= 4.0:
            raise ValueError('admission_exponent must be in [0,4]')
        if type(self.fallback_min_domain_edges) is not int or self.fallback_min_domain_edges < 1:
            raise ValueError('fallback_min_domain_edges must be a positive integer')


def _coerce_center_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _community_members(record: dict[str, Any], index: int) -> frozenset[int]:
    members = np.asarray(record['communities'][int(index)]['member_indices'], dtype=np.int64).reshape(-1)
    unique = frozenset(members.tolist())
    if len(unique) != members.size:
        raise ValueError('duplicate member index inside Community')
    return unique


def _admission_utility(probability: float, delta: float, exponent: float) -> float:
    try:
        return probability / (max(1e-12, float(delta)) ** float(exponent))
    except OverflowError:
        return 0.


def build_domain(record: dict[str, Any], selected_indices: Iterable[int], *, max_community_tokens: int, max_domain_edges: int,
                 fallback_used: bool = False, community_scores: np.ndarray | None = None) -> dict[str, Any]:
    communities = record['communities']
    edge_keys = [str(k) for k in record['edge_keys']]
    selected = [int(i) for i in selected_indices]
    selected_communities: list[dict[str, Any]] = []
    union_keys: set[str] = set()
    for i in selected:
        community = communities[i]
        for e in community['member_indices']:
            union_keys.add(edge_keys[int(e)])
        selected_communities.append({'community_id': str(community['community_id']), 'relation_family': str(community['relation_family']),
                                     'relation_key': str(community['relation_key']), 'size': int(community['size']),
                                     'selection_score': float(np.asarray(community_scores, dtype=np.float64).reshape(-1)[i])
                                     if community_scores is not None else None})
    sorted_keys = sorted(union_keys)
    key_to_edge = {k: e for k, e in zip(edge_keys, record['edges'])}
    domain_edges = sorted((key_to_edge[k] for k in sorted_keys), key=lambda e: str(e.get('canonical_key', '')))
    hop_histogram = {'1': 0, '2': 0, '3': 0}
    for edge in domain_edges:
        hop_histogram[str(int(edge['hop']))] = hop_histogram.get(str(int(edge['hop'])), 0) + 1
    witness = [str(k) for k in record['witness_edge_keys']]
    covered = sum(1 for k in witness if k in union_keys)
    receptive = len(edge_keys)
    domain_size = len(sorted_keys)
    return {'center_id': _coerce_center_id(record['center_id']), 'title': str(record.get('title', '')),
            'pred_before': int(record['pred_before']), 'pred_before_label': str(record['pred_before_label']),
            'receptive_field_size': int(receptive), 'community_count': int(len(communities)),
            'selected_communities': selected_communities,
            'selected_community_ids': [r['community_id'] for r in selected_communities],
            'macro_domain_edge_keys': sorted_keys, 'macro_domain_edges': domain_edges, 'macro_domain_size': int(domain_size),
            'hop_histogram': hop_histogram, 'witness_edge_keys': witness, 'witness_size': int(len(witness)),
            'witness_covered_count': int(covered), 'witness_full_coverage': bool(len(witness) > 0 and covered == len(witness)),
            'compression_ratio': float(domain_size / max(1, receptive)), 'fallback_used': bool(fallback_used),
            'overflow_community_tokens': bool(len(communities) > int(max_community_tokens)),
            'overflow_macro_domain': bool(domain_size > int(max_domain_edges))}


def select_domain(record: dict[str, Any], probabilities: np.ndarray, params: DomainParams,
                  edit_cost=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Admit communities with p >= tau greedily by probability per marginal edit cost**exponent under the edge cap."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError('selector probabilities must be one-dimensional')
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError('selector probabilities must be finite and in [0,1]')
    communities = record['communities']
    if len(p) != len(communities):
        raise ValueError('selector probability/community mismatch')
    edge_keys = list(map(str, record['edge_keys']))
    if len(edge_keys) != len(set(edge_keys)):
        raise ValueError('record contains duplicate canonical edge keys')
    member_sets = [_community_members(record, index) for index in range(len(communities))]
    domain_size = len(edge_keys)
    for members in member_sets:
        if members and (min(members) < 0 or max(members) >= domain_size):
            raise ValueError('Community member outside record edge domain')
    # Trade-off: stage2.domain_edge_cap bounds the policy's action space (latency) at the risk of excluding needed
    # edges; stage3.expansion_budget lets the search recover them later.
    cap = None if params.domain_edge_cap is None else int(params.domain_edge_cap)
    exponent = float(params.admission_exponent)
    eligible = [i for i in range(len(p)) if p[i] >= float(params.relevance_threshold) and member_sets[i]]
    selected: list[int] = []
    union: set[int] = set()
    steps: list[dict[str, Any]] = []
    remaining = set(eligible)
    while remaining:
        candidates = []
        for index in remaining:
            members = member_sets[index]
            delta = len(members - union)
            if delta == 0:
                continue
            if cap is not None and len(union) + delta > cap:
                continue
            community_id = str(communities[index].get('community_id', index))
            marginal_cost = delta if edit_cost is None else edit_cost(edge_keys[j] for j in members - union)
            utility = _admission_utility(float(p[index]), marginal_cost, exponent)
            candidates.append((-utility, -float(p[index]), delta, community_id, int(index), members))
        if not candidates:
            break
        _, neg_p, delta, community_id, index, members = min(candidates)
        marginal_cost = delta if edit_cost is None else edit_cost(edge_keys[j] for j in members - union)
        selected.append(index)
        union.update(members)
        remaining.remove(index)
        steps.append({'community_index': index, 'community_id': community_id, 'probability': -neg_p, 'delta': delta,
                      'utility': _admission_utility(float(p[index]), marginal_cost, exponent),
                      'union_after': len(union), 'fallback': False})

    fallback = False
    fallback_reason = None
    if not selected and len(p):
        # Soft fallback: communities in descending p_i; keep the fewest top-ranked ones whose union reaches
        # stage2.fallback_min_domain_edges (whole communities that bring new edges and fit the cap).
        ranked = sorted(range(len(p)), key=lambda i: (-float(p[i]), str(communities[i].get('community_id', i)), i))
        for index in ranked:
            members = member_sets[index]
            delta = len(members - union)
            if delta == 0 or (cap is not None and len(union) + delta > cap):
                continue
            selected.append(index)
            union.update(members)
            steps.append({'community_index': index, 'community_id': str(communities[index].get('community_id', index)),
                          'probability': float(p[index]), 'delta': delta, 'utility': float(p[index]) / delta,
                          'union_after': len(union), 'fallback': True})
            if len(union) >= params.fallback_min_domain_edges:
                break
        fallback = bool(selected)
        fallback_reason = 'FEWEST_TOP_RANKED_COMMUNITIES' if selected else 'NO_FEASIBLE_WHOLE_COMMUNITY'

    domain = build_domain(record, selected, max_community_tokens=2 ** 31 - 1,
                          max_domain_edges=(cap if cap is not None else 2 ** 31 - 1),
                          fallback_used=fallback, community_scores=p)
    expected = {str(record['edge_keys'][index]) for index in union}
    actual = set(domain['macro_domain_edge_keys'])
    if actual != expected:
        raise RuntimeError('Domain is not the exact selected-Community union')
    if cap is not None and len(actual) > cap:
        raise RuntimeError('unique union exceeded the domain edge cap')
    sum_sizes = sum(len(member_sets[index]) for index in selected)
    trace = {
        'alpha': float(params.relevance_threshold), 'macro_cap': cap,
        'eligible_count': len(eligible), 'selected_count': len(selected),
        'selected_indices': selected, 'fallback': fallback,
        'fallback_reason': fallback_reason, 'steps': steps,
        'sum_selected_community_sizes': sum_sizes,
        'unique_union': len(actual),
        'overlap_memberships': sum_sizes - len(actual),
        'overlap_ratio': 0.0 if sum_sizes == 0 else (sum_sizes - len(actual)) / sum_sizes,
        'selected_community_members': [[edge_keys[item] for item in sorted(member_sets[index])] for index in selected],
    }
    return domain, trace
