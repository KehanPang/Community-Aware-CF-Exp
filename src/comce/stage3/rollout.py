from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np

from comce.stage2.domain import build_domain
from comce.stage3.sampler import BitSource, sample
from comce.stage3.settings import SearchSettings

# Domain-separation label of the per-request RNG seed. It fixes which trajectory a seed produces, never the policy's
# sampling distribution; execution.random_seed selects another trajectory.
ROLLOUT_SEED_DOMAIN = 'comce.stage3.rollout'


def canonical(keys) -> tuple[str, ...]:
    return tuple(sorted(set(map(str, keys))))


def request_seed(base_seed: int, center_id: str) -> int:
    value = f'{ROLLOUT_SEED_DOMAIN}|{base_seed}|{center_id}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big')


@dataclass
class SearchContext:
    keys: list[str]
    policy_logits: Callable
    world_model_score: Callable
    final_ranking_score: Callable
    scoring_contract: dict | None = None


class CommunityExpansion:
    """Append-only domain growth by whole communities in descending selector score, capped at B_ext new edges."""

    def __init__(self, record, probabilities, domain, budget_edges, max_communities_per_trigger='all'):
        self.record = record
        self.probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        communities = record['communities']
        if len(communities) != len(self.probabilities) or not np.isfinite(self.probabilities).all():
            raise ValueError('invalid frozen community scores')
        self.edge_keys = list(map(str, record['edge_keys']))
        if len(self.edge_keys) != len(set(self.edge_keys)):
            raise ValueError('duplicate legal edge key')
        ids = [str(c.get('community_id', i)) for i, c in enumerate(communities)]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate community id')
        self.selected = [ids.index(str(k)) for k in domain['selected_community_ids']]
        self.members = []
        for c in communities:
            indices = list(map(int, c['member_indices']))
            if len(indices) != len(set(indices)) or any(i < 0 or i >= len(self.edge_keys) for i in indices):
                raise ValueError('invalid whole-community membership')
            self.members.append(frozenset(self.edge_keys[i] for i in indices))
        self.initial = frozenset(map(str, domain['macro_domain_edge_keys']))
        if self.initial != frozenset().union(*(self.members[i] for i in self.selected)):
            raise ValueError('D0 differs from selected whole-community union')
        self.current = set(self.initial)
        self.budget = int(budget_edges)
        if self.budget < 0:
            raise ValueError('negative expansion budget')
        self.used = 0
        self.order = sorted(range(len(communities)), key=lambda i: (-self.probabilities[i], ids[i]))
        self.ids = ids
        self.domain = domain
        self.history = []
        self.max_communities = max_communities_per_trigger

    def expand(self, after_rollout):
        before = set(self.current)
        admitted, skipped = [], []
        for i in self.order:
            if self.max_communities != 'all' and len(admitted) >= self.max_communities:
                break
            if i in self.selected or self.used >= self.budget:
                continue
            extra = self.members[i] - self.current
            if not extra:
                continue
            if len(extra) > self.budget - self.used:
                skipped.append(self.ids[i])
                continue
            self.current.update(extra)
            self.used += len(extra)
            self.selected.append(i)
            admitted.append(self.ids[i])
        if not before <= self.current or self.used != len(self.current - self.initial):
            raise RuntimeError('append-only expansion/accounting violation')
        if admitted:
            self.domain = build_domain(self.record, self.selected, max_community_tokens=2 ** 31 - 1,
                                       max_domain_edges=len(self.initial) + self.budget,
                                       fallback_used=bool(self.domain.get('fallback_used', False)),
                                       community_scores=self.probabilities)
            if set(self.domain['macro_domain_edge_keys']) != self.current:
                raise RuntimeError('expanded domain is not the exact whole-community union')
        event = {'after_rollout': int(after_rollout), 'admitted': admitted,
                 'admitted_community_members': [sorted(self.members[self.ids.index(cid)]) for cid in admitted],
                 'skipped_not_whole_fit': skipped, 'before': sorted(before),
                 'after': sorted(self.current), 'b_ext_used': self.used}
        self.history.append(event)
        return bool(admitted)


class GuidanceScores:
    """Request-local cache of S for proposed one-step outcomes."""
    # Trade-off: outcomes are scored in batches of stage3.guidance_batch_size and cached until the domain grows,
    # so repeated proposals cost no WorldModel call.

    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.cache: dict[tuple[str, ...], float] = {}
        self.calls = 0
        self.sets_scored = 0
        self.cache_hits = 0

    def scores(self, context, sets):
        need = [s for s in dict.fromkeys(sets) if s not in self.cache]
        if need:
            for i in range(0, len(need), self.batch_size):
                chunk = need[i:i + self.batch_size]
                values = np.asarray(context.final_ranking_score(chunk), dtype=np.float64)
                if values.shape != (len(chunk),) or not np.isfinite(values).all():
                    raise ValueError('invalid frozen guidance scores')
                self.calls += 1
                self.sets_scored += len(chunk)
                for key, value in zip(chunk, values):
                    self.cache[key] = float(value)
        self.cache_hits += len(sets) - len(need)
        return [self.cache[s] for s in sets]

    def reset(self):
        self.cache = {}

    def payload(self):
        return {'kind': 'final', 'calls': self.calls, 'sets_scored': self.sets_scored, 'cache_hits': self.cache_hits}


def run_rollouts(initial_domain, expansion, context_factory, *, seed, settings: SearchSettings, bits=None):
    temperature = float(settings.temperature)
    K, lam = int(settings.proposals_per_step), float(settings.guidance_scale)
    bits = BitSource(seed) if bits is None else bits
    context = context_factory(initial_domain)
    guidance = GuidanceScores(settings.guidance_batch_size)
    pool = {}
    trajectories = []
    failure_count = 0
    proposal_stats = {'steps': 0, 'scored_steps': 0, 'changed_vs_first_proposal': 0}
    for rollout in range(settings.rollouts):
        if context.keys != sorted(expansion.current):
            raise RuntimeError('policy/WM domain mismatch')
        path, prefixes = [], []
        stop_type = 'NO_LEGAL_ADD'
        min_logp = None
        for _ in range(min(settings.explanation_budget, len(context.keys))):
            selected = set(path)
            legal = [i for i, key in enumerate(context.keys) if key not in selected]
            if not legal:
                break
            all_logits = np.asarray(context.policy_logits(canonical(path)), dtype=np.float64)
            if all_logits.shape != (len(context.keys) + 1,):
                raise ValueError('policy must return ADD logits and one STOP logit')
            actions = legal + [len(context.keys)]
            weights = all_logits[actions] / temperature
            if not np.isfinite(weights).all():
                raise ValueError('non-finite legal policy logit')
            shifted = weights - weights.max()
            logp = shifted - np.logaddexp.reduce(shifted)
            min_logp = float(logp.min()) if min_logp is None else min(min_logp, float(logp.min()))
            proposals = [actions[sample(weights, bits)] for _ in range(K)]
            proposal_stats['steps'] += 1
            if lam == 0.0 or len(set(proposals)) == 1:
                choice = sample(np.zeros(K), bits) if lam == 0.0 and K > 1 else 0
                action = proposals[choice]
            else:
                stop_index = len(context.keys)
                outcomes = [canonical(path) if a == stop_index else canonical(path + [context.keys[a]]) for a in proposals]
                scorable = [o for o in outcomes if o]
                values = dict(zip(scorable, guidance.scores(context, scorable))) if scorable else {}
                log_weights = np.array([lam * values[o] if o else -np.inf for o in outcomes], dtype=np.float64)
                finite = np.isfinite(log_weights)
                if finite.any():
                    index = np.flatnonzero(finite)
                    choice = int(index[sample(log_weights[index], bits)])
                else:
                    choice = 0
                action = proposals[choice]
                proposal_stats['scored_steps'] += 1
                proposal_stats['changed_vs_first_proposal'] += int(action != proposals[0])
            if action == len(context.keys):
                stop_type = 'STOP'
                break
            path.append(context.keys[action])
            key = canonical(path)
            prefixes.append(key)
            pool.setdefault(key, tuple(path))
            stop_type = 'BUDGET' if len(path) == settings.explanation_budget else 'NO_LEGAL_ADD'
        raw = np.asarray(context.world_model_score(prefixes), dtype=np.float64) if prefixes else np.empty(0)
        if raw.shape != (len(prefixes),) or not np.isfinite(raw).all():
            raise ValueError('invalid frozen WM prefix scores')
        best = float(np.exp(-np.logaddexp(0., -raw.max()))) if len(raw) else 0.0
        failure_count = failure_count + 1 if best < settings.failure_threshold else 0
        count_before_expand = failure_count
        expanded = False
        if (failure_count >= settings.failures_before_expansion and rollout + 1 < settings.rollouts
                and expansion.used < expansion.budget):
            expanded = expansion.expand(rollout + 1)
            failure_count = 0
            if expanded:
                context = context_factory(expansion.domain)
                guidance.reset()
        trajectories.append({'rollout': rollout + 1, 'path': path, 'prefixes': [list(k) for k in prefixes],
                             'stop_type': stop_type, 'best_rollout_wm': best,
                             'low_wm': best < settings.failure_threshold,
                             'failure_count_before_expand': count_before_expand, 'failure_count': failure_count,
                             'expanded': expanded, 'min_legal_logp': min_logp})
    sets = sorted(pool)
    scores = np.asarray(context.final_ranking_score(sets), dtype=np.float64) if sets else np.empty(0)
    if scores.shape != (len(sets),) or not np.isfinite(scores).all():
        raise ValueError('invalid frozen final scores')
    return {'sets': sets, 'paths': [pool[s] for s in sets], 'scores': scores,
            'context': context, 'domain': expansion.domain,
            'stats': {'mode': 'stochastic', 'seed': seed, 'n_roll': settings.rollouts,
                      'proposal_K': K, 'wm_guidance_lambda': lam, 'guidance_score': 'final',
                      'guidance': guidance.payload(), 'proposals': proposal_stats,
                      'trajectories': trajectories, 'expansion_events': expansion.history,
                      'b_ext': expansion.budget, 'b_ext_used': expansion.used,
                      'initial_domain_keys': sorted(expansion.initial),
                      'final_domain_keys': sorted(expansion.current),
                      'all_prefixes_retained': all(canonical(s) in pool for t in trajectories for s in t['prefixes']),
                      'stop_count': sum(t['stop_type'] == 'STOP' for t in trajectories),
                      'prefix_count': sum(len(t['prefixes']) for t in trajectories),
                      'exact_unique_count': len(sets), 'bit_words': bits.words}}
