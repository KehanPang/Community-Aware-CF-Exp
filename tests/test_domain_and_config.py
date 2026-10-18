from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from comce.pipeline.config import eligibility_reason, load_template, parse_override, resolve
from comce.stage2.domain import DomainParams, select_domain
from comce.stage3.certification import certify_candidates
from comce.stage3.edit_cost import EdgeCost
from comce.training.policy import candidate_reward
from comce.training.selector import alpha_slot_diversity, selector_loss, unique_edge_coverage

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'yaml.template'


def record(groups):
    keys = sorted(set().union(*map(set, groups)))
    return dict(center_id='target', edge_keys=keys, u123_size=len(keys), witness_edge_keys=[], pred_before=1,
                pred_before_label='1', edges=[dict(canonical_key=k, hop=1) for k in keys],
                communities=[dict(community_id=f'c{i}', member_indices=[keys.index(k) for k in g], relation_family='r',
                                  relation_key='r', size=len(g)) for i, g in enumerate(groups)])


def test_threshold_admission_keeps_whole_communities_under_the_cap():
    rec = record([('a', 'b'), ('b', 'c'), ('d',)])
    domain, trace = select_domain(rec, [.9, .8, .4], DomainParams(.5, 2))
    assert domain['macro_domain_edge_keys'] == ['a', 'b'] and trace['selected_indices'] == [0]
    domain, trace = select_domain(rec, [.9, .8, .4], DomainParams(.5, None))
    assert domain['macro_domain_edge_keys'] == ['a', 'b', 'c'] and trace['unique_union'] == 3
    assert trace['sum_selected_community_sizes'] == 4 and trace['overlap_memberships'] == 1


def test_soft_fallback_takes_the_fewest_top_ranked_communities_not_a_single_top1():
    rec = record([('a',), ('b', 'c'), ('d',), ('e', 'f', 'g')])
    domain, trace = select_domain(rec, [.3, .2, .1, .05], DomainParams(.5, None, 1., 3))
    assert domain['macro_domain_edge_keys'] == ['a', 'b', 'c'] and trace['fallback']
    assert trace['selected_indices'] == [0, 1] and trace['fallback_reason'] == 'FEWEST_TOP_RANKED_COMMUNITIES'
    domain, trace = select_domain(rec, [.3, .2, .1, .05], DomainParams(.5, None, 1., 10))
    assert domain['macro_domain_size'] == 7 and trace['selected_indices'] == [0, 1, 2, 3]
    domain, trace = select_domain(rec, [.3, .2, .1, .05], DomainParams(.5, 2, 1., 10))
    assert domain['macro_domain_edge_keys'] == ['a', 'd'] and trace['selected_indices'] == [0, 2]
    domain, trace = select_domain(record([('a', 'b')]), [.3], DomainParams(.5, 1))
    assert not domain['macro_domain_edge_keys'] and trace['fallback_reason'] == 'NO_FEASIBLE_WHOLE_COMMUNITY'
    with pytest.raises(ValueError):
        DomainParams(.5, None, 1., 0)


def test_admission_utility_is_probability_per_marginal_cost():
    rec = record([('a', 'b'), ('c',)])
    domain, _ = select_domain(rec, [.9, .8], DomainParams(.1, 2))
    assert domain['macro_domain_edge_keys'] == ['c']
    domain, _ = select_domain(rec, [.9, .8], DomainParams(.1, 2, 0.))
    assert domain['macro_domain_edge_keys'] == ['a', 'b']


def test_weighted_admission_extreme_finite_cost_and_exponent_bounds():
    with pytest.raises(ValueError, match='in \\[0,4\\]'):
        DomainParams(.1, 2, 5.)
    rec = record([('a',), ('b',)])
    cost = EdgeCost(mode='mapping', weights={'a': 1e308, 'b': 1.}).bind(rec)
    domain, trace = select_domain(rec, [.9, .8], DomainParams(.1, 1, 4.), cost)
    assert domain['macro_domain_edge_keys'] == ['b'] and np.isfinite(trace['steps'][0]['utility'])


def test_eligibility_is_explicit_and_does_not_change_flip_direction():
    task = dict(prediction='checkpoint', min_receptive_edges_exclusive=200, allowed_target_ids=None)
    reason = eligibility_reason(task, center_id='x', receptive_edges=200, original_prediction=2, positive_class=1,
                                checkpoint_eligible=True)
    assert reason == 'RECEPTIVE_EDGES_MUST_EXCEED_200'
    assert eligibility_reason(task, center_id='x', receptive_edges=201, original_prediction=2, positive_class=1,
                              checkpoint_eligible=True) is None
    positive = dict(task, prediction='positive_only', min_receptive_edges_exclusive=None)
    assert eligibility_reason(positive, center_id='x', receptive_edges=5, original_prediction=0, positive_class=1,
                              checkpoint_eligible=True) == 'ORIGINAL_PREDICTION_NOT_POSITIVE'
    cohort = dict(task, allowed_target_ids=['y'])
    assert eligibility_reason(cohort, center_id='x', receptive_edges=500, original_prediction=1, positive_class=1,
                              checkpoint_eligible=True) == 'TARGET_NOT_IN_CONFIGURED_COHORT'


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf'), True, '1'])
def test_invalid_edge_weights_rejected(weight):
    with pytest.raises(ValueError):
        EdgeCost(mode='mapping', weights={'a': weight})


def test_edge_cost_mapping_callable_and_missing_ids():
    rec = record([('a', 'b', 'c')])
    with pytest.raises(ValueError, match='Missing edge costs'):
        EdgeCost(mode='mapping', weights={'a': 1.}).bind(rec)
    mapped = EdgeCost(mode='mapping', weights={'a': .25, 'b': 2., 'c': 0.}).bind(rec)
    called = EdgeCost(mode='callable', function=lambda e: {'a': .25, 'b': 2., 'c': 0.}[e['canonical_key']],
                      identity='example-costs').bind(rec)
    assert mapped(['b', 'a']) == called(['a', 'b']) == 2.25
    assert mapped(['c']) == 0.
    with pytest.raises(ValueError, match='Unknown canonical'):
        mapped(['unknown'])
    with pytest.raises(ValueError, match='Duplicate'):
        mapped(['a', 'a'])


def test_uniform_cost_matches_and_weighted_objective_selects_cheaper_certificate():
    rec = record([('a', 'b', 'c')])
    candidates = [('a',), ('b', 'c')]; scores = [2., 1.]

    def run(cost=None):
        return certify_candidates(candidates, scores, lambda k, p: {'flip': True}, lambda s: [0.] * len(s), q_clean=0, edit_cost=cost)
    plain = run(); uniform = run(EdgeCost().bind(rec))
    assert plain == uniform and uniform['returned_edge_keys'] == ['a']
    weighted = run(EdgeCost(mode='mapping', weights={'a': 10., 'b': .2, 'c': .3}).bind(rec))
    assert weighted['returned_edge_keys'] == ['b', 'c'] and weighted['weighted_edit_cost'] == .5
    assert weighted['returned_size'] == 2


def test_weighted_admission_and_reward_share_cost_but_keep_count_budget():
    rec = record([('a',), ('b', 'c')])
    cost = EdgeCost(mode='mapping', weights={'a': 10., 'b': .1, 'c': .1}).bind(rec)
    domain, trace = select_domain(rec, [.9, .8], DomainParams(.1, 2), cost)
    assert domain['macro_domain_edge_keys'] == ['b', 'c'] and trace['unique_union'] == 2
    exp = SimpleNamespace(rl_lambda_flip=1., rl_lambda_size=.1, rl_lambda_div=.1, rl_max_candidate_edges=10)
    base = candidate_reward(.8, 2, .9, exp=exp)
    assert candidate_reward(.8, 2, .9, exp=exp, edit_cost=EdgeCost().bind(rec), edge_keys=['b', 'c']) == base
    assert candidate_reward(.8, 2, .9, exp=exp, edit_cost=cost, edge_keys=['b', 'c']) > base
    assert np.isclose(base, .8 - .1 * 2 / 10 - .1 * (1 - .9))


def test_selector_objective_is_the_weighted_sum_of_its_terms():
    p = torch.tensor([[.7, .4]]); q = torch.tensor([[[.7, .4], [.2, .5], [.1, .2]]])
    a = torch.tensor([[[.8, .2], [.4, .6], [.1, .9]]]); logits = torch.logit(p)
    batch = dict(community_mask=torch.tensor([[True, True]]), witness_mask=torch.tensor([[1., 0.]]), edge_mask=torch.tensor([[1., 1.]]),
                 community_pos=torch.tensor([[1., 0.]]),
                 incidence=dict(n_edges_max=2, community_index=torch.tensor([0, 1]), edge_index=torch.tensor([0, 1])))
    weights = dict(lambda_cover=1., lambda_size=.2, lambda_community=1., lambda_membership=.1, lambda_membership_entropy=.02,
                   lambda_slot_diversity=.08)
    total = selector_loss(dict(p_community=p, q_membership=q, alpha_attention=a, macro_logits=logits), batch, weights)
    p_edge = unique_edge_coverage(p, batch['community_mask'], batch['incidence'])['p_edge']
    l_cover = -(torch.log(p_edge + 1e-08) * batch['witness_mask']).sum() / 1.
    l_size = p_edge.sum() / 2.
    community = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], batch['community_pos'][0])
    membership = torch.nn.functional.binary_cross_entropy(q.reshape(-1), batch['community_pos'][0].repeat(3))
    entropy = -(q * torch.log(q) + (1 - q) * torch.log(1 - q)).mean()
    expected = l_cover + .2 * l_size + community + .1 * membership + .02 * entropy + .08 * alpha_slot_diversity(a)
    assert torch.allclose(total, expected, atol=1e-7)


def test_configuration_precedence_command_line_tuple_dataset_paper():
    template = load_template(TEMPLATE)
    claim = resolve(template, 'claim', 'gcn')
    assert claim.parameters['stage2']['relevance_threshold'] == 0.1
    assert claim.parameters['stage3']['rollouts'] == 32 and claim.search.query_cap == 12
    hgt = resolve(template, 'claim', 'hgt')
    assert hgt.parameters['stage2']['relevance_threshold'] == 0.005 and hgt.domain.domain_edge_cap == 12
    goodreads = resolve(template, 'goodreads', 'gcn')
    assert goodreads.search.temperature == 2.0 and goodreads.execution['random_seed'] == 20260910
    assert claim.execution['random_seed'] == 20260919 and claim.task['prediction'] == 'positive_only'
    overridden = resolve(template, 'goodreads', 'gcn', ['stage3.temperature=1.5', 'stage2.relevance_threshold=0.2'])
    assert overridden.search.temperature == 1.5 and overridden.domain.relevance_threshold == 0.2
    paper = yaml.safe_load(TEMPLATE.read_text())['paper_defaults']
    assert paper['stage2'] == dict(relevance_threshold=0.10, query_slots=3, attention_heads=4, hidden_width=128,
                                   domain_edge_cap=None)
    assert paper['stage3'] == dict(explanation_budget=10, rollouts=32, failure_threshold=0.5, failures_before_expansion=4,
                                   expansion_budget=8, expansion_communities_per_trigger=1, certification_queries=9,
                                   lookahead=2, cleanup_queries=2, dedup_threshold=0.9, proposals_per_step=1,
                                   guidance_scale=0.0, temperature=1.0)
    assert claim.search.dedup_threshold == 0.9 and claim.domain.fallback_min_domain_edges == 10
    assert claim.search.expansion_communities_per_trigger == 'all' and claim.search.proposals_per_step == 4


def test_paper_settings_are_reachable_through_configuration():
    paper = load_template(TEMPLATE)
    for section in ('stage2', 'stage3'):
        paper['dataset_defaults'][section] = {k: v for k, v in paper['dataset_defaults'][section].items()
                                              if k not in paper['paper_defaults'][section]}
    for dataset in ('claim', 'computer', 'goodreads'):
        paper['tuples'][dataset] = {b: {k: v for k, v in t.items() if k in ('model', 'gnn')} for b, t in paper['tuples'][dataset].items()}
        paper['datasets'][dataset].pop('stage3', None)
        config = resolve(paper, dataset, 'gcn')
        assert config.domain.domain_edge_cap is None and config.domain.relevance_threshold == 0.10
        assert config.search.expansion_communities_per_trigger == 1 and config.search.proposals_per_step == 1
        assert config.search.guidance_scale == 0.0 and config.search.temperature == 1.0
        assert config.search.expansion_budget == 8 and config.search.dedup_threshold == 0.9


@pytest.mark.parametrize('override', ['stage3.proposals_per_step=0', 'stage3.guidance_scale=.nan', 'stage3.cleanup_queries=-1',
                                      'stage3.explanation_budget=11', 'stage3.expansion_communities_per_trigger=0',
                                      'stage3.dedup_threshold=1.5', 'stage2.fallback_min_domain_edges=0',
                                      'stage3.edit_cost={mode: eval}', 'stage3.unknown=1', 'stage2.admission_exponent=5',
                                      'task.prediction=anything', 'world_model.mode=other', 'stage1.encoder=x'])
def test_invalid_configuration_fails(override):
    with pytest.raises(ValueError):
        resolve(load_template(TEMPLATE), 'computer', 'gcn', [override])


def test_override_syntax_and_unknown_tuple():
    assert parse_override('stage2.domain_edge_cap=null') == ('stage2', 'domain_edge_cap', None)
    with pytest.raises(ValueError):
        resolve(load_template(TEMPLATE), 'claim', 'gat')
    with pytest.raises(ValueError):
        resolve(load_template(TEMPLATE), 'cora', 'gcn')
