import ast
from pathlib import Path

import numpy as np
import torch

from comce.stage1.signatures import SignatureContext
from comce.stage3.ranking import CandidateRanker, RankingResidual, pooled_set_descriptors
from comce.stage3.scoring import RequestScorer
from comce.stage3.set_context import SetContext
from comce.stage3.world_model import STATES, WorldModel, world_model_states

SRC = Path(__file__).resolve().parents[1] / 'src' / 'comce'
NETWORKS = {'WorldModel', 'RankingResidual', 'Policy', 'SetTransformer', 'SignatureEmbedding', 'CommunityEdgeReadout',
            'ExplainedModel', 'StandardBackbone'}


def module_classes():
    found = []
    for path in sorted(SRC.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                bases = {ast.unparse(b) for b in node.bases}
                found.append((node.name, bases, path.relative_to(SRC)))
    return found


def test_exactly_one_world_model_class_and_no_second_network_family():
    classes = module_classes()
    networks = [(name, path) for name, bases, path in classes if bases & {'nn.Module', 'torch.nn.Module', 'nn.ModuleDict'}]
    assert {name for name, _ in networks} == NETWORKS
    assert len([n for n, _ in networks if n == 'WorldModel']) == 1
    assert [n for n, _, _ in classes if 'world' in n.lower()] == ['WorldModel']
    forwards = [path for path in SRC.rglob('*.py') if 'head_flip' in path.read_text()]
    assert [p.relative_to(SRC).as_posix() for p in forwards] == ['stage3/world_model.py']


def test_omega_and_omega_bar_are_two_parameter_sets_of_one_architecture():
    torch.manual_seed(0)
    states = world_model_states(edge_in=5, set_dim=4, hidden=6)
    assert tuple(states.keys()) == STATES
    assert {type(states[s]) for s in STATES} == {WorldModel}
    assert type(states['omega']).forward is type(states['omega_bar']).forward is WorldModel.forward
    shapes = [{k: tuple(v.shape) for k, v in states[s].state_dict().items()} for s in STATES]
    assert shapes[0] == shapes[1]
    assert not any(torch.equal(a, b) for a, b in zip(states['omega'].parameters(), states['omega_bar'].parameters()))


def test_world_model_is_permutation_invariant_over_the_set():
    torch.manual_seed(1)
    model = WorldModel(5, 4, 6).eval()
    embeddings = model.encode_edges(torch.randn(2, 3, 5))
    setf = torch.randn(2, 4)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    perm = embeddings[:, [2, 0, 1]]
    perm_mask = mask[:, [2, 0, 1]]
    assert torch.allclose(model(embeddings, setf, mask), model(perm, setf, perm_mask), atol=1e-6)


def toy_scorer(m=4, edge_in=5):
    torch.manual_seed(2)
    keys = [f'e{i}' for i in range(m)]
    rows = np.random.default_rng(3).normal(size=(m, edge_in)).astype(np.float32)
    record = dict(edge_keys=keys, edge_rows=rows, center_state=np.array([.1, .2], np.float32))
    domain = dict(macro_domain_edge_keys=keys)
    sets = SetContext(record, domain)
    signatures = SignatureContext(tuple(keys), np.zeros(8), np.zeros((m, 8)), dict(center='c', token='c'))
    states = world_model_states(edge_in, sets.set_dim, 6).eval()
    residual = RankingResidual(edge_in).eval()
    return RequestScorer(states, residual, sets, signatures), states, residual, sets


def test_ranking_score_is_adapted_world_model_logit_plus_linear_residual():
    scorer, states, residual, sets = toy_scorer()
    ranker = CandidateRanker(scorer)
    candidates = [('e0',), ('e1', 'e3'), ('e0', 'e2', 'e3')]
    s = ranker(candidates)
    for candidate, value in zip(candidates, s):
        indices, mask = sets.pack([sorted(sets.index[k] for k in candidate)])
        with torch.no_grad():
            rows = sets.edge_rows[:, 0, :][indices]
            z_bar = states['omega_bar'](states['omega_bar'].encode_edges(sets.edge_rows)[indices, torch.arange(indices.shape[1])[None]],
                                        sets.set_features(indices, mask), mask)
            psi = pooled_set_descriptors(rows, mask)
            expected = z_bar + psi @ residual.linear.weight.T.squeeze(-1) + residual.linear.bias
        assert np.isclose(value, float(expected), atol=1e-5)
    np.testing.assert_array_equal(ranker([('e3', 'e1')]), s[1:2])
    np.testing.assert_array_equal(ranker(candidates[::-1]), s[::-1])


def test_rollout_score_uses_omega_and_ranking_uses_omega_bar():
    scorer, states, _, sets = toy_scorer()
    with torch.no_grad():
        for p in states['omega_bar'].parameters():
            p.add_(1.0)
    fresh = RequestScorer(states, scorer.ranking_residual, sets, scorer.signatures)
    z = fresh.world_model_scores([('e1',)])
    indices, mask = sets.pack([[1]])
    with torch.no_grad():
        omega = states['omega'](states['omega'].encode_edges(sets.edge_rows)[indices, torch.arange(1)[None]],
                                sets.set_features(indices, mask), mask)
    assert np.isclose(z[0], float(omega), atol=1e-6)
    assert not np.isclose(CandidateRanker(fresh)([('e1',)])[0], z[0])
