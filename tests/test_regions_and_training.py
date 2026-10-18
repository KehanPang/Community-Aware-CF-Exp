from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from comce.pipeline.config import ResourcePaths, load_template, resolve
from comce.stage1.signatures import SignatureBank, SignatureContext, random_directions
from comce.stage2.operators import quantize
from comce.stage2.regions import RegionConstructor, region_inputs
from comce.stage3.rollout import CommunityExpansion, SearchContext, run_rollouts
from comce.stage3.scoring import RequestScorer
from comce.stage3.set_context import SetContext
from comce.stage3.settings import SearchSettings
from comce.stage3.world_model import world_model_states
from comce.stage3.ranking import RankingResidual

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / 'yaml.template')


def toy_bank(types=('x', 'y')):
    rng = np.random.default_rng(4)
    coordinates = {t: rng.normal(size=(6, 8)).astype(np.float32) for t in types}
    pooled = np.concatenate(list(coordinates.values())).astype(np.float64)
    from comce.stage1.signatures import quantile_thresholds
    return SignatureBank(coordinates, quantile_thresholds(pooled), {})


def toy_edges(src='x', dst='y', relation='r'):
    return [dict(canonical_key=f'{relation}:{i}:{j}', edge_type=relation, src_type=src, dst_type=dst, src_id=str(i),
                 dst_id=str(j), src_row=i, dst_row=j, hop=1 + (i + j) % 3) for i, j in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]]


def construct(src='x', dst='y', relation='r', **extra):
    bank = toy_bank((src, dst))
    inputs = region_inputs(center_id=0, center_type=src, center_row=0, edges=toy_edges(src, dst, relation), bank=bank,
                           relation_names=[relation], probabilities=[.2, .8], prediction=1)
    inputs.fields.update(extra)
    return RegionConstructor().construct(inputs), inputs, bank


def test_region_construction_depends_on_fields_and_operators_not_on_names():
    first, _, _ = construct()
    alias, _, _ = construct(dataset='unseen_graph', backbone='unseen_encoder')
    renamed, _, _ = construct('p', 'q', 's')
    assert first['communities'] == alias['communities']
    assert [c['member_indices'] for c in first['communities']] == [c['member_indices'] for c in renamed['communities']]
    for key in ('edge_rows', 'center_sig', 'center_state', 'community_sigs', 'community_feats', 'edge_signatures', 'hops'):
        np.testing.assert_array_equal(first[key], renamed[key])


def test_center_and_edge_signatures_share_one_projection_and_bucketizer():
    record, inputs, bank = construct()
    fields = inputs.fields
    np.testing.assert_array_equal(record['center_sig'], quantize(fields['center_z'][None], bank.thresholds)[0])
    np.testing.assert_array_equal(record['edge_signatures'], quantize((fields['source_z'] + fields['target_z']) / 2., bank.thresholds))
    context = record.signature_context
    assert context.coordinate_spaces['center'] == context.coordinate_spaces['token']
    assert set(context.base_coordinate_spaces.values()) == {context.coordinate_spaces['center']}
    relations = 1
    rows = record['edge_rows']
    assert rows.shape[1] == 3 + relations + 6 + 8 + 8
    np.testing.assert_array_equal(rows[:, 3 + relations + 6:3 + relations + 14] * 15, record['edge_signatures'])
    np.testing.assert_array_equal(np.argmax(rows[:, :3], 1) + 1, record['hops'])


def test_signature_context_is_validated_but_never_conditions_the_scores():
    keys = ['a', 'b']
    record = dict(edge_keys=keys, edge_rows=np.array([[1., 2.], [3., -1.]], np.float32), center_state=np.array([.5], np.float32))
    sets = SetContext(record, dict(macro_domain_edge_keys=keys))
    torch.manual_seed(0)
    states, residual = world_model_states(2, sets.set_dim, 4).eval(), RankingResidual(2).eval()
    one = SignatureContext(tuple(keys), np.array([1, 2]), np.array([[2, 1], [1, 2]]), {'center': 'c', 'token': 'c'},
                           region_members={'region': ('a', 'b')})
    other = SignatureContext(tuple(keys), np.array([9, 9]), one.edge_tokens, one.coordinate_spaces)
    a, b = RequestScorer(states, residual, sets, one), RequestScorer(states, residual, sets, other)
    np.testing.assert_array_equal(a.world_model_scores([('a',), ('b',)]), b.world_model_scores([('a',), ('b',)]))
    assert one.statistics(keys)['regions'] == 1 and other.statistics(keys)['regions'] == 0
    with pytest.raises(ValueError, match='unknown edge'):
        SignatureContext(('x', 'b'), one.center, one.edge_tokens, one.coordinate_spaces).validate(['a'])
    with pytest.raises(ValueError, match='alignment'):
        SignatureContext(('a', 'b'), one.center, np.zeros((1, 2)), one.coordinate_spaces).validate(['a'])
    with pytest.raises(ValueError, match='Nonfinite'):
        SignatureContext(('a', 'b'), one.center, np.full((2, 2), np.nan), one.coordinate_spaces).validate(['a'])


def test_text_coordinates_are_seeded_projections_of_encoder_features(resource_template):
    config = resolve(resource_template, 'computer', 'gcn')
    data = config.paths.resource('data')
    bank = SignatureBank.load(config.signature_bank_path)
    directions = random_directions(768, TEMPLATE['dataset_defaults']['stage1']['projection_seed'])
    papers = np.load(data / 'paper_text_embeddings.npy', mmap_mode='r')
    rows = np.asarray([0, 17, 4096, 400000])
    expected = np.asarray(papers[rows], np.float64) @ directions
    np.testing.assert_allclose(bank.coordinates['paper'][rows], expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    arrays = np.load(data / 'graph_arrays.npz')
    owners = np.repeat(np.arange(len(arrays['author_ptr']) - 1), np.diff(arrays['author_ptr']))
    author = int(arrays['author_idx'][123])
    mean = bank.coordinates['paper'][owners[arrays['author_idx'] == author]].astype(np.float64).mean(0)
    np.testing.assert_allclose(bank.coordinates['author'][author], mean, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dataset', ['claim', 'computer', 'goodreads'])
def test_trained_world_model_reads_signature_hop_relation_columns_and_residual_was_supervised(dataset, resource_template):
    for backbone in resource_template['tuples'][dataset]:
        config = resolve(resource_template, dataset, backbone)
        payload = torch.load(config.model_path, map_location='cpu', weights_only=True)
        relations = len(config.dataset_config['relations'])
        weight = payload['world_model']['omega']['edge_enc.0.weight']
        assert weight.shape[1] == 3 + relations + 6 + 8 + 8
        for block in (slice(0, 3), slice(3, 3 + relations), slice(3 + relations + 6, 3 + relations + 14)):
            assert float(weight[:, block].abs().sum()) > 0
        history = payload['training']['world_model_history']
        assert len(history) > 1 and history[-1]['train_bce'] < history[0]['train_bce']
        adaptation = payload['training']['adaptation']
        assert adaptation['residual_weight_change'] > 0 and adaptation['residual_weight_l1'] > 0
        assert len(adaptation['residual']) > 1 and len(adaptation['adapted']) > 1
        assert not torch.equal(payload['world_model']['omega_bar']['head_flip.2.weight'],
                               payload['world_model']['omega']['head_flip.2.weight'])
        assert payload['ranking_residual']['linear.weight'].shape == (1, 2 * weight.shape[1] + 1)


def test_guidance_keeps_duplicate_proposal_multiplicity(monkeypatch):
    keys = ['a', 'b', 'c']
    rec = dict(center_id='0', pred_before=1, pred_before_label='1', u123_size=3, witness_edge_keys=[], edge_keys=keys,
               edges=[dict(canonical_key=k, hop=1) for k in keys],
               communities=[dict(community_id='c0', member_indices=[0, 1, 2], relation_family='r', relation_key='r', size=3)])
    domain = dict(selected_community_ids=['c0'], macro_domain_edge_keys=keys, macro_domain_size=3, fallback_used=False)
    draws = iter([0, 0, 1, 0])
    seen = []

    def draw(weights, bits):
        seen.append(np.asarray(weights).copy())
        return next(draws)
    monkeypatch.setattr('comce.stage3.rollout.sample', draw)
    scored = []

    def score(sets):
        scored.append(list(sets))
        return np.array([1. if s == ('a',) else 2. for s in sets])
    settings = replace(SearchSettings(), rollouts=1, explanation_budget=1, proposals_per_step=3, expansion_budget=0)
    result = run_rollouts(domain, CommunityExpansion(rec, [1.], domain, 0),
                          lambda d: SearchContext(keys, lambda s: np.zeros(4), score, score), seed=1, settings=settings)
    assert scored[0] == [('a',), ('b',)]
    np.testing.assert_array_equal(seen[-1], [.5, .5, 1.])
    assert result['sets'] == [('a',)]


def test_training_targets_follow_the_labeled_fraction_unless_a_fixed_count_is_declared(tmp_path):
    from types import SimpleNamespace
    from comce.training.trainer import target_quota
    (tmp_path / 'labeled.txt').write_text(' '.join(str(i) for i in range(50)))
    template = dict(training=dict(target_fraction=0.2))

    def quota(available, **training):
        dataset_config = dict(resources=dict(populations=dict(labeled='labeled.txt')),
                              training=dict(labeled_population='labeled', **training))
        paths = ResourcePaths(dict(_path=str(tmp_path / 'config.yaml'), resources=dict(root=str(tmp_path)),
                                   datasets=dict(toy=dataset_config)), 'toy')
        return target_quota(SimpleNamespace(dataset_config=dataset_config, paths=paths), template, available)
    assert quota(40) == dict(mode='labeled_fraction', fraction=0.2, labeled=50, inspected=10)
    assert quota(40, target_fraction=0.5) == dict(mode='labeled_fraction', fraction=0.5, labeled=50, inspected=25)
    with pytest.raises(ValueError, match='needs 10 TRAIN targets'):
        quota(9)
    assert quota(40, targets=7) == dict(mode='eligible_count', targets=7)
    assert quota(40, targets='all') == dict(mode='all_eligible')
    with pytest.raises(ValueError, match='mutually exclusive'):
        quota(40, targets=7, target_fraction=0.2)
    for wrong in (0, 2.5, 'some'):
        with pytest.raises(ValueError, match='positive integer, all, or null'):
            quota(40, targets=wrong)
    with pytest.raises(ValueError):
        target_quota(SimpleNamespace(dataset_config=dict(training={}), paths=None), dict(training={}), 40)
    claim = TEMPLATE['datasets']['claim']['training']
    assert TEMPLATE['training']['target_fraction'] == 0.2 and claim['labeled_population'] == 'labeled'
    assert all(TEMPLATE['datasets'][d]['training']['target_fraction'] is None for d in TEMPLATE['datasets'])


def mixed_record():
    bank = toy_bank(('x', 'y', 'w'))
    edges = toy_edges('x', 'y', 'r') + [dict(e, canonical_key=f's:{e["src_row"]}:{e["dst_row"]}', edge_type='s', dst_type='w',
                                             hop=1 + e['src_row'] % 2) for e in toy_edges('x', 'w', 's')]
    inputs = region_inputs(center_id=0, center_type='x', center_row=0, edges=edges, bank=bank, relation_names=['r', 's'],
                           probabilities=[.2, .8], prediction=1)
    return RegionConstructor().construct(inputs)


def test_structural_view_is_the_edge_type_and_hop_partition():
    # The endpoint-motif key reads endpoint types and hop only, so with one endpoint pair per edge type its
    # communities are the (edge_type, hop) blocks, which the computational (hop, type, b_v) communities refine.
    record = mixed_record()
    views = {}
    for c in record['communities']:
        views.setdefault(c['relation_family'], set()).add(frozenset(map(int, c['member_indices'])))
    blocks = {}
    for i, edge in enumerate(sorted(record['edges'], key=lambda e: str(e['canonical_key']))):
        blocks.setdefault((edge['edge_type'], int(edge['hop'])), set()).add(i)
    assert views['endpoint_motif'] == {frozenset(b) for b in blocks.values()}
    assert all(any(c <= s for s in views['endpoint_motif']) for c in views['hop_type_community'])


def test_communities_with_one_member_set_are_identical_tokens():
    # A community token is a function of its member set; the view never enters the token.
    record = mixed_record()
    groups = {}
    for i, c in enumerate(record['communities']):
        groups.setdefault(frozenset(map(int, c['member_indices'])), []).append(i)
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    assert duplicates
    for ids in duplicates:
        assert len({record['communities'][i]['relation_family'] for i in ids}) == len(ids)
        for i in ids[1:]:
            np.testing.assert_array_equal(record['community_sigs'][i], record['community_sigs'][ids[0]])
            np.testing.assert_array_equal(record['community_feats'][i], record['community_feats'][ids[0]])
