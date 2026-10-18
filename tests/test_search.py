import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from comce.stage3 import sampler as es
from comce.stage3.candidates import jaccard_greedy
from comce.stage3.certification import QueryLedger, certify_candidates
from comce.stage3.rollout import CommunityExpansion, SearchContext, run_rollouts
from comce.stage3.settings import SearchSettings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = SearchSettings()


@pytest.fixture(autouse=True)
def no_models_or_learning(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError('model load/forward/backward forbidden in this unit phase')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.autograd, 'backward', forbidden)


def realised_log(log_x):
    """log of the probability exact_bernoulli realises for exp(log_x)."""
    if log_x >= 0.0:
        return 0.0
    e, m = es._split(log_x)
    return e * es.LN2 + math.log(math.ceil(m * 2.0 ** 53) * es.TWO_M53)


def selection_probabilities(log_w):
    """Exact log-probability of each index under es.sample."""
    order, log_q, log_r = es._prepare(log_w)
    out = np.empty(len(order))
    carry = 0.0
    for j in range(len(order)):
        if j == len(order) - 1:
            out[order[j]] = carry
            break
        if log_q[j] <= es.LOG_HALF:
            acc = realised_log(float(log_q[j]))
            rej = math.log1p(-math.exp(acc))
        else:
            rej = realised_log(float(log_r[j]))
            acc = math.log1p(-math.exp(rej))
        out[order[j]] = carry + acc
        carry += rej
    return out


def record(groups=(('a',), ('b', 'c'), ('c', 'd'))):
    keys = sorted({k for group in groups for k in group})
    return dict(center_id='0', pred_before=1, pred_before_label='1', u123_size=len(keys), witness_edge_keys=[],
                edge_keys=keys, edges=[dict(canonical_key=k, hop=1) for k in keys],
                communities=[dict(community_id=f'c{i}', member_indices=[keys.index(k) for k in group],
                                  relation_family='synthetic', relation_key='r', size=len(group))
                             for i, group in enumerate(groups)])


def initial(rec):
    keys = [rec['edge_keys'][i] for i in rec['communities'][0]['member_indices']]
    return dict(selected_community_ids=['c0'], macro_domain_edge_keys=keys, macro_domain_size=len(keys), fallback_used=False)


def run(rec=None, *, n=5, fmax=4, budget=10, wm_values=None, b_ext=3, K=1):
    rec = rec or record()
    domain = initial(rec)
    expansion = CommunityExpansion(rec, [1.0] * len(rec['communities']), domain, b_ext)
    seen = []
    wm = iter(wm_values) if wm_values is not None else None

    def factory(d):
        keys = sorted(d['macro_domain_edge_keys'])
        seen.append(keys)

        def logits(chosen):
            return np.arange(len(keys) + 1, dtype=float)

        def omega(sets):
            return np.full(len(sets), next(wm) if wm is not None else -2.)
        return SearchContext(keys, logits, omega, lambda s: np.arange(len(s), dtype=float))
    settings = replace(DEFAULT, rollouts=n, failures_before_expansion=fmax, explanation_budget=budget, proposals_per_step=K,
                       expansion_budget=b_ext)
    result = run_rollouts(domain, expansion, factory, seed=7, settings=settings)
    return result, expansion, seen


def choose_add(monkeypatch):
    monkeypatch.setattr('comce.stage3.rollout.sample', lambda w, bits: 0)


def test_stop_actually_terminates_empty_and_nonempty(monkeypatch):
    monkeypatch.setattr('comce.stage3.rollout.sample', lambda w, bits: len(w) - 1)
    result, _, _ = run(n=2)
    assert result['sets'] == []
    assert all(t['stop_type'] == 'STOP' and not t['path'] for t in result['stats']['trajectories'])
    calls = iter([0, 1])
    monkeypatch.setattr('comce.stage3.rollout.sample', lambda w, bits: next(calls))
    result, _, _ = run(record((('a', 'b'),)), n=1)
    assert result['sets'] == [('a',)]
    assert result['stats']['trajectories'][0]['stop_type'] == 'STOP'


def test_add_legality_prefix_retention_and_exact_not_jaccard(monkeypatch):
    choose_add(monkeypatch)
    result, _, _ = run(record((('a', 'b', 'c'),)), n=3)
    assert result['sets'] == [('a',), ('a', 'b'), ('a', 'b', 'c')]
    assert result['stats']['prefix_count'] == 9 and result['stats']['exact_unique_count'] == 3
    assert result['stats']['all_prefixes_retained']
    assert all(len(t['path']) == len(set(t['path'])) for t in result['stats']['trajectories'])


def test_failure_reset_then_fmax_expand_and_earlier_sets_survive(monkeypatch):
    choose_add(monkeypatch)
    result, expansion, seen = run(n=8, wm_values=[-2., 2., -2., -2., -2., -2., -2., -2.])
    traces = result['stats']['trajectories']
    assert [t['failure_count'] for t in traces[:6]] == [1, 0, 1, 2, 3, 0]
    assert expansion.history[0]['after_rollout'] == 6
    assert seen == [['a'], ['a', 'b', 'c', 'd']]
    assert ('a',) in result['sets'] and ('a', 'b', 'c', 'd') in result['sets']


def test_no_pre_expansion_and_final_rollout_does_not_change_context(monkeypatch):
    choose_add(monkeypatch)
    result, expansion, seen = run(n=4)
    assert not expansion.history and seen == [['a']]
    assert result['stats']['trajectories'][-1]['failure_count'] == 4


def test_whole_community_skip_accounting_and_append_only():
    rec = record((('a',), ('b', 'c', 'd'), ('e', 'f'), ('f', 'g')))
    domain = initial(rec)
    exp = CommunityExpansion(rec, [1., .9, .8, .7], domain, 2)
    exp.expand(4)
    assert exp.current == {'a', 'e', 'f'} and exp.used == 2
    assert exp.history[0]['admitted'] == ['c2']
    assert exp.history[0]['skipped_not_whole_fit'] == ['c1']
    assert set(exp.history[0]['before']) <= set(exp.history[0]['after'])
    assert set(domain['macro_domain_edge_keys']) == {'a'}


def test_probability_then_canonical_community_order():
    rec = record((('a',), ('b',), ('c',)))
    rec['communities'][1]['community_id'] = 'z'
    rec['communities'][2]['community_id'] = 'b'
    exp = CommunityExpansion(rec, [1., .5, .5], initial(rec), 1)
    exp.expand(4)
    assert exp.current == {'a', 'c'}


@pytest.mark.parametrize('count,expected', [(1, ['c1']), (2, ['c1', 'c2']), ('all', ['c1', 'c2', 'c3'])])
def test_expansion_counts_overlap_and_no_readmission(count, expected):
    rec = record([('a',), ('b', 'c'), ('c', 'd'), ('e',)])
    exp = CommunityExpansion(rec, [1., .9, .8, .7], initial(rec), 4, count)
    exp.expand(4)
    assert exp.history[0]['admitted'] == expected
    assert exp.used == len(exp.current - {'a'}) <= 4
    exp.expand(8)
    exp.expand(12)
    admitted = [c for event in exp.history for c in event['admitted']]
    assert len(admitted) == len(set(admitted))
    assert all(count == 'all' or len(event['admitted']) <= count for event in exp.history)
    assert exp.current == {'a', 'b', 'c', 'd', 'e'}


@pytest.mark.parametrize('budget,expected', [(0, {'a'}), (1, {'a', 'e'}), (2, {'a', 'b', 'c'}), (3, {'a', 'b', 'c', 'd'})])
def test_expansion_edge_budget_boundaries(budget, expected):
    rec = record([('a',), ('b', 'c'), ('c', 'd'), ('e',)])
    exp = CommunityExpansion(rec, [1., .9, .8, .7], initial(rec), budget)
    exp.expand(4)
    assert exp.current == expected and exp.used <= budget


@pytest.mark.parametrize('weights', [
    [0., -745.], [0., -100000., -100003.], [0., -40., -60., -144.],
    [0.] * 49, [1e30, 1e30, 1e30], [1e30, -1e30, -1e30],
    [float(np.finfo(np.float32).max), -float(np.finfo(np.float32).max)] * 2,
])
def test_all_finite_legal_add_and_stop_have_sampling_support(weights):
    logs = selection_probabilities(np.asarray(weights))
    assert np.isfinite(logs).all()
    assert np.all(logs <= 1e-12)
    if max(weights) - min(weights) < 1e6:
        shifted = np.asarray(weights) - max(weights)
        target = shifted - np.logaddexp.reduce(shifted)
        np.testing.assert_allclose(logs, target, atol=2e-10, rtol=1e-12)


def test_constructive_sampler_support_for_each_action():
    weights = np.array([0., -1., -60., -100000.])
    order, logq, logr = es._prepare(weights)
    for desired in range(len(weights)):
        target_position = list(order).index(desired)
        decisions = [(j == target_position if logq[j] <= es.LOG_HALF else j != target_position)
                     for j in range(target_position + 1)]

        class ScriptedBits:
            words = 0

            def __init__(self):
                self.decisions = iter(decisions)

            def zero_run(self, count):
                return next(self.decisions)

            def uniform53(self):
                return 0.0
        assert es.sample(weights, ScriptedBits()) == desired


def test_sampler_seed_repeatable():
    a, b = es.BitSource(13), es.BitSource(13)
    assert [es.sample(np.array([0., -1., -20.]), a) for _ in range(50)] == [es.sample(np.array([0., -1., -20.]), b) for _ in range(50)]


def test_request_seed_selects_a_trajectory_but_not_the_policy_distribution():
    from comce.stage3.rollout import request_seed
    weights = np.array([0., -1., -2.5])
    target = np.exp(selection_probabilities(weights))
    draws = [es.sample(weights, es.BitSource(request_seed(20260919, str(c)))) for c in range(20000)]
    np.testing.assert_allclose(np.bincount(draws, minlength=3) / len(draws), target, atol=0.012)
    assert request_seed(20260919, '7') != request_seed(20260910, '7')


def certify(candidates, scores, flips, **kwargs):
    calls = []

    def oracle(keys, phase):
        calls.append((keys, phase))
        return {'flip': keys in flips}
    return certify_candidates(candidates, scores, oracle, lambda s: list(range(len(s))), **kwargs), calls


def test_kappa_keeps_first_hit_after_later_failures():
    result, calls = certify([('a', 'b'), ('c',), ('d',), ('e',)], [4, 3, 2, 1], {('a', 'b')}, q_clean=0)
    assert result['success'] and result['returned_edge_keys'] == ['a', 'b']
    assert len(calls) == 3


def test_smaller_later_certificate_then_score_and_canonical_tie():
    result, _ = certify([('a', 'b'), ('c',), ('d',), ('e',)], [4, 3, 3, 1], {('a', 'b'), ('c',), ('d',)}, q_clean=0)
    assert result['returned_edge_keys'] == ['c'] and result['first_hit_rank'] == 1


def test_lookahead_does_not_reset_on_second_hit():
    result, _ = certify([('a', 'b', 'c'), ('a', 'b'), ('a', 'c'), ('d',)], [4, 3, 2, 1],
                        {('a', 'b', 'c'), ('a', 'b'), ('a', 'c'), ('d',)}, q_clean=0)
    assert result['candidate_queries'] == 3 and result['returned_edge_keys'] == ['a', 'b']


def test_independent_cert_cleanup_ledgers_total_twelve_and_cleanup_shrinks_the_certified_set():
    negatives = [(f'x{i}',) for i in range(8)]
    result, calls = certify(negatives + [('a', 'b', 'c')], list(range(9, 0, -1)), {('a', 'b', 'c'), ('b', 'c')})
    assert result['query_ledger'] == dict(q0=1, certification=9, cleanup=2, physical_total=12, Q_cert=9, Q_clean=2)
    assert calls[-2:] == [(('b', 'c'), 'cleanup'), (('c',), 'cleanup')]
    assert result['returned_edge_keys'] == ['b', 'c']
    assert result['cleanup_attempts'][0]['keys'] == ['b', 'c']


def test_single_pass_does_not_retry_previously_failed_edge():
    result, calls = certify([('a', 'b', 'c')], [1], {('a', 'b', 'c'), ('a', 'c'), ('c',)}, q_clean=3)
    assert result['returned_edge_keys'] == ['a', 'c']
    assert [k for k, p in calls if p == 'cleanup'] == [('b', 'c'), ('a', 'c'), ('a',)]
    assert not result['strict_local_minimality']


def test_rank_dedup_keeps_near_sets_below_tau_dup_and_skips_exact_copies():
    result, calls = certify([('a',), ('a', 'b'), ('a',)], [3, 2, 1], {('a', 'b')}, q_clean=0)
    assert result['success'] and result['schedule_count'] == 2
    assert len(calls) == 2
    result, calls = certify([('a', 'b', 'c'), ('a', 'b'), ('d',)], [3, 2, 1], {('a', 'b')}, q_clean=0, dedup_threshold=0.6)
    assert result['schedule_count'] == 2 and [k for k, _ in calls] == [('a', 'b', 'c'), ('d',)] and not result['success']


def test_jaccard_greedy_is_strict_and_skips_against_retained_candidates_only():
    nine = tuple('abcdefghi')
    ten = nine + ('j',)
    assert jaccard_greedy([ten, nine], 0.9) == [ten, nine]
    assert jaccard_greedy([ten, nine], 0.89) == [ten]
    assert jaccard_greedy([('a', 'b'), ('a', 'b', 'c'), ('a', 'c')], 0.6) == [('a', 'b'), ('a', 'c')]
    assert jaccard_greedy([('a', 'b'), ('a', 'b')], 1.0) == [('a', 'b'), ('a', 'b')]
    with pytest.raises(ValueError):
        jaccard_greedy([('a',)], 1.5)


@pytest.mark.parametrize('budget', [3, 7, 10])
def test_jaccard_greedy_equals_exact_dedup_when_tau_dup_is_at_least_one_minus_one_over_b(budget):
    rng = np.random.default_rng(budget)
    universe = [f'e{i}' for i in range(budget + 3)]
    pool = []
    for _ in range(400):
        size = int(rng.integers(1, budget + 1))
        pool.append(tuple(sorted(rng.choice(universe, size=size, replace=False).tolist())))
    exact = list(dict.fromkeys(pool))
    assert jaccard_greedy(exact, 1 - 1 / budget) == exact and jaccard_greedy(exact, 0.9 if budget <= 10 else 1.) == exact
    distinct = [(a, b) for a in exact[:60] for b in exact[:60] if a != b]
    worst = max(len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in distinct)
    assert worst <= (budget - 1) / budget
    if budget > 1:
        assert len(jaccard_greedy(exact, (budget - 2) / (budget - 1) - 1e-9)) < len(exact)


def test_empty_pool_failure_and_budget_guards():
    result, calls = certify([], [], set())
    assert not result['success'] and result['total_queries'] == 1 and not calls
    ledger = QueryLedger()
    for _ in range(9):
        ledger.charge('certification')
    with pytest.raises(RuntimeError):
        ledger.charge('certification')
    for _ in range(2):
        ledger.charge('cleanup')
    with pytest.raises(RuntimeError):
        ledger.charge('cleanup')


def test_custom_query_budgets_cleanup_certification_and_soundness():
    queries = []

    def oracle(k, p):
        queries.append((k, p))
        return {'flip': len(k) >= 2}
    result = certify_candidates([('a', 'b', 'c')], [1.], oracle, lambda s: [0.] * len(s), q_cert=1, q_clean=5)
    assert result['success'] and len(result['returned_edge_keys']) == 2
    assert result['query_ledger']['physical_total'] == 1 + len(queries) <= 7
    assert not result['strict_local_minimality'] and result['pruning_guarantee'] == 'BUDGET_BOUNDED_SINGLE_PASS'


def test_zero_certification_budget_never_calls_the_oracle():
    def forbidden(*args):
        raise AssertionError('Zero certification budget must not call the oracle')
    result = certify_candidates([('a',)], [1.], forbidden, lambda s: [0.] * len(s), q_cert=0, q_clean=4)
    assert not result['success'] and result['total_queries'] == 1


def test_default_search_settings_are_the_paper_budgets():
    s = DEFAULT
    assert (s.rollouts, s.explanation_budget, s.lookahead, s.certification_queries, s.cleanup_queries) == (32, 10, 2, 9, 2)
    assert (s.failures_before_expansion, s.failure_threshold, s.expansion_budget) == (4, 0.5, 8)
    assert (s.proposals_per_step, s.guidance_scale) == (4, 0.5)
    assert s.query_cap == 12 and s.payload()['query_cap'] == 12
    for bad in (dict(proposals_per_step=0), dict(explanation_budget=11), dict(guidance_scale=float('nan')),
                dict(expansion_communities_per_trigger=0), dict(failure_threshold=1.5), dict(temperature=0.)):
        with pytest.raises(ValueError):
            replace(s, **bad)


def exact_marginal(p, w, K):
    p, w = np.asarray(p, float), np.asarray(w, float)
    q = np.zeros(len(p))
    for t in itertools.product(range(len(p)), repeat=K):
        prob = np.prod(p[list(t)])
        total = w[list(t)].sum()
        if total == 0:
            q[t[0]] += prob
            continue
        for j in t:
            q[j] += prob * w[j] / total
    return q


KEYS7 = [f'e{i}' for i in range(7)]
ADD7 = np.array([1.2, 0.3, -0.5, 2.0, -1.0, 0.0, 0.7])


def one_step(lam, K=4, n=30000, record_scored=None):
    rec = record(tuple((k,) for k in KEYS7))
    domain = dict(selected_community_ids=[f'c{i}' for i in range(7)], macro_domain_edge_keys=list(KEYS7),
                  macro_domain_size=7, fallback_used=False)
    expansion = CommunityExpansion(rec, [1.0] * 7, domain, 0)

    def score(sets):
        if record_scored is not None:
            record_scored.extend(sets)
        return np.array([np.sin(3 * len(s) + sum(KEYS7.index(k) * 1.7 for k in s)) * 2.0 for s in sets])

    def factory(d):
        return SearchContext(sorted(KEYS7), lambda chosen: np.r_[ADD7, -2.5], score, score)
    settings = replace(DEFAULT, rollouts=n, explanation_budget=1, failures_before_expansion=10 ** 9, proposals_per_step=K,
                       guidance_scale=lam, expansion_budget=0)
    return run_rollouts(domain, expansion, factory, seed=2026, settings=settings), score


def root_frequencies(result):
    counts = np.zeros(len(KEYS7) + 1)
    for t in result['stats']['trajectories']:
        counts[KEYS7.index(t['path'][0]) if t['path'] else len(KEYS7)] += 1
    return counts / counts.sum()


def root_policy():
    w = np.r_[ADD7, -2.5]
    return np.exp(w - np.logaddexp.reduce(w))


def test_lambda_zero_proposal_is_exactly_the_policy_distribution():
    result, _ = one_step(0.0)
    assert np.max(np.abs(root_frequencies(result) - root_policy())) < 0.012


def test_guided_frequencies_match_the_exact_sir_marginal_and_stop_at_root_has_weight_zero():
    result, score = one_step(1.0)
    p = root_policy()
    g = np.r_[score([(k,) for k in sorted(KEYS7)]), -np.inf]
    w = np.where(np.isfinite(g), np.exp(g), 0.0)
    q = exact_marginal(p, w, 4)
    assert np.max(np.abs(root_frequencies(result) - q)) < 0.012
    assert np.isclose(q[-1], p[-1] ** 4)


@pytest.mark.parametrize('seed', range(5))
def test_every_legal_action_keeps_probability_at_least_pi_to_the_K(seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(4)); g = rng.normal(0, 3, 4)
    for lam in (0.5, 1.0, 5.0):
        q = exact_marginal(p, np.exp(lam * g - (lam * g).max()), 4)
        assert np.all(q >= p ** 4 - 1e-15) and np.all(q > 0) and np.isclose(q.sum(), 1.0)


def test_scorer_sees_only_proposed_outcomes_and_pool_is_visited_prefixes():
    scored = []
    result, _ = one_step(0.5, n=50, record_scored=scored)
    visited = {tuple(p) for t in result['stats']['trajectories'] for p in t['prefixes']}
    assert set(result['sets']) == visited and result['stats']['all_prefixes_retained']
    guidance = result['stats']['guidance']
    assert guidance['sets_scored'] <= 4 * result['stats']['proposals']['scored_steps']
    assert all(len(s) == 1 for s in scored[:guidance['sets_scored']])


def test_proposal_K1_is_bit_identical_to_the_policy_only_engine():
    rec = record((('a', 'b', 'c', 'd'),))

    def factory(d):
        keys = sorted(d['macro_domain_edge_keys'])
        return SearchContext(keys, lambda chosen: np.r_[np.linspace(-1, 1, len(keys)), -1.5],
                             lambda sets: np.full(len(sets), 2.0), lambda sets: np.arange(len(sets), dtype=float))
    domain = initial(rec)
    settings = replace(DEFAULT, rollouts=40, proposals_per_step=1, temperature=1.3, expansion_budget=0)
    got = run_rollouts(domain, CommunityExpansion(rec, [1.0], domain, 0), factory, seed=99, settings=settings)
    bits = es.BitSource(99); keys = sorted(domain['macro_domain_edge_keys']); paths = []
    for _ in range(40):
        path = []
        for _ in range(min(10, len(keys))):
            legal = [i for i, k in enumerate(keys) if k not in set(path)]
            if not legal:
                break
            actions = legal + [len(keys)]
            w = np.r_[np.linspace(-1, 1, len(keys)), -1.5][actions] / 1.3
            a = actions[es.sample(w, bits)]
            if a == len(keys):
                break
            path.append(keys[a])
        paths.append(path)
    assert [t['path'] for t in got['stats']['trajectories']] == paths
    assert got['stats']['bit_words'] == bits.words and got['stats']['guidance']['calls'] == 0


def test_single_proposal_stop_seed_and_all_stop(monkeypatch):
    rec = record([('a', 'b')]); domain = initial(rec)

    def factory(d):
        return SearchContext(['a', 'b'], lambda s: np.array([0., 0., 0.]), lambda s: np.zeros(len(s)), lambda s: np.zeros(len(s)))
    single = replace(DEFAULT, rollouts=40, explanation_budget=2, proposals_per_step=1, expansion_budget=0)

    def run_with(settings):
        return run_rollouts(domain, CommunityExpansion(rec, [1.], domain, 0), factory, seed=9, settings=settings)
    a = run_with(single)
    assert a['stats']['trajectories'] == run_with(single)['stats']['trajectories']
    assert a['stats']['guidance']['calls'] == 0
    assert any(t['stop_type'] == 'STOP' and not t['path'] for t in a['stats']['trajectories'])
    monkeypatch.setattr('comce.stage3.rollout.sample', lambda w, bits: len(w) - 1)
    all_stop = run_with(replace(single, proposals_per_step=4))
    assert all_stop['sets'] == [] and all_stop['stats']['guidance']['calls'] == 0


def test_empty_domain_runs_without_queries_or_expansion():
    rec = record([('a',)])
    domain = dict(selected_community_ids=[], macro_domain_edge_keys=[], macro_domain_size=0, fallback_used=False)

    def factory(d):
        return SearchContext([], lambda s: np.array([0.]), lambda s: np.zeros(len(s)), lambda s: np.zeros(len(s)))
    result = run_rollouts(domain, CommunityExpansion(rec, [.2], domain, 0), factory, seed=7,
                          settings=replace(DEFAULT, rollouts=2, expansion_budget=0))
    assert result['sets'] == [] and result['stats']['initial_domain_keys'] == []
    assert not result['stats']['expansion_events']


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='real CUDA transfer test')
def test_gpu_logits_cpu_sampler_preserves_dtype_values_and_seeded_choices():
    import torch
    cpu = torch.tensor([-13.125, 0., 3.5, -41.75], dtype=torch.float32)
    transferred = cpu.to('cuda:0').detach().double().cpu().numpy()
    reference = cpu.detach().double().cpu().numpy()
    assert transferred.dtype == np.float64 and np.array_equal(transferred, reference)
    a, b = es.BitSource(918), es.BitSource(918)
    assert [es.sample(transferred / 2., a) for _ in range(100)] == [es.sample(reference / 2., b) for _ in range(100)]
    assert 'detach().double().cpu().numpy()' in (ROOT / 'src/comce/stage3/policy.py').read_text()
