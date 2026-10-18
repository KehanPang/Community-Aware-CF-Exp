import json
from types import SimpleNamespace

import pytest
import torch

from comce.pipeline.config import ResourcePaths
from comce.stage3.world_model import STATES, WorldModel, world_model_states
from comce.training.trainer import module_logits, training_targets
from comce.training.world_model_update import (UPDATE_FORMAT, apply_update, certification_labels, fit_update,
                                               world_model_logits)

SETTINGS = dict(epochs=150, lr=0.01, l2_weight=0.0, stability_weight=0.0, validation_fraction=0.3, min_validation_labels=20,
                min_improvement=0.005, seed=1)


def batch(n, seed, teacher=None):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(n, 10, 5, generator=generator)
    sizes = torch.randint(1, 11, (n,), generator=generator)
    mask = torch.arange(10)[None] < sizes[:, None]
    features = torch.randn(n, 4, generator=generator)
    if teacher is None:
        labels = torch.randint(0, 2, (n,), generator=generator).float()
    else:
        with torch.no_grad():
            labels = (module_logits(teacher, rows, mask, features) > 0).float()
    return [rows, mask, features, labels]


def model(seed):
    torch.manual_seed(seed)
    return WorldModel(5, 4, 6).eval()


def test_zero_delta_is_omega_and_every_parameter_gets_a_gradient_at_initialization():
    omega = model(0)
    rows, mask, features, labels = batch(64, 1)
    parameters = {k: v.detach() for k, v in omega.named_parameters()}
    assert torch.equal(world_model_logits(omega, parameters, rows, mask, features), module_logits(omega, rows, mask, features))
    delta, gate, result = fit_update(omega, [rows, mask, features, labels], batch(40, 2), dict(SETTINGS, epochs=0), device='cpu')
    assert all(not d.any() for d in delta.values()) and gate == 0.0
    assert result['validation']['updated'] == result['validation']['omega']
    delta, _, _ = fit_update(omega, [rows, mask, features, labels], batch(40, 2), dict(SETTINGS, epochs=1), device='cpu')
    assert all(d.abs().sum() > 0 for d in delta.values())
    assert torch.equal(module_logits(omega, rows, mask, features), world_model_logits(omega, parameters, rows, mask, features))


def test_update_is_accepted_only_when_held_out_labels_improve():
    teacher, omega = model(7), model(0)
    fit, validation = batch(400, 3, teacher), batch(120, 4, teacher)
    _, gate, result = fit_update(omega, fit, validation, SETTINGS, device='cpu')
    assert gate == 1.0 and result['decision']['improved']
    assert result['validation']['updated']['bce'] < result['validation']['omega']['bce'] - SETTINGS['min_improvement']
    _, gate, result = fit_update(omega, fit, validation, dict(SETTINGS, min_improvement=10.0), device='cpu')
    assert gate == 0.0 and result['decision']['enough_validation_labels'] and not result['decision']['improved']
    _, gate, result = fit_update(omega, fit, batch(10, 5, teacher), SETTINGS, device='cpu')
    assert gate == 0.0 and not result['decision']['enough_validation_labels']
    _, gate, _ = fit_update(omega, fit, None, SETTINGS, device='cpu')
    assert gate == 0.0


def write_update(path, gate, delta, supervised=('t1',)):
    torch.save(dict(format=UPDATE_FORMAT, state='omega', gate=gate, delta=delta, base_bundle_sha256='m' * 64,
                    supervision=dict(fit_targets=list(supervised), validation_targets=['t2'])), path)
    return path


def test_applied_update_changes_only_omega_of_the_one_world_model(tmp_path):
    states = world_model_states(5, 4, 6)
    before = {s: {k: v.clone() for k, v in states[s].state_dict().items()} for s in STATES}
    delta = {k: torch.full_like(v, 0.5) for k, v in states['omega'].named_parameters()}
    models = SimpleNamespace(world_model=states)
    rejected = apply_update(models, write_update(tmp_path / 'rejected.pt', 0.0, delta), bundle_sha256='m' * 64, target_ids=['t3'])
    assert rejected['gate'] == 0.0 and states['omega'] is models.world_model['omega']
    info = apply_update(models, write_update(tmp_path / 'accepted.pt', 1.0, delta), bundle_sha256='m' * 64, target_ids=['t3'])
    assert info['gate'] == 1.0 and type(models.world_model['omega']) is WorldModel
    for name, value in models.world_model['omega'].state_dict().items():
        torch.testing.assert_close(value, before['omega'][name] + 0.5)
    for name, value in models.world_model['omega_bar'].state_dict().items():
        assert torch.equal(value, before['omega_bar'][name])
    with pytest.raises(ValueError, match='another model bundle'):
        apply_update(models, tmp_path / 'accepted.pt', bundle_sha256='x' * 64, target_ids=['t3'])
    with pytest.raises(ValueError, match='supplied the real-GNN labels'):
        apply_update(models, tmp_path / 'accepted.pt', bundle_sha256='m' * 64, target_ids=['t3', 't2'])


def toy_tuple(tmp_path, train='t1 t2 t3'):
    root = tmp_path / 'resources'
    (root / 'pop').mkdir(parents=True)
    (root / 'pop/train.txt').write_text(train)
    (root / 'pop/dev.txt').write_text('d1 d2')
    (root / 'pop/test.txt').write_text('x1')
    dataset_config = dict(resources=dict(populations=dict(dev='pop/dev.txt', test1000='pop/test.txt')),
                          training=dict(population_path='pop/train.txt', evaluation_populations=['dev', 'test1000']))
    paths = ResourcePaths(dict(_path=str(tmp_path / 'config.yaml'), resources=dict(root=str(root)), datasets=dict(toy=dataset_config)),
                          'toy', 'gcn')
    config = SimpleNamespace(dataset='toy', backbone='gcn', tuple_config=dict(gnn=dict(sha256='g' * 64)), dataset_config=dataset_config,
                             paths=paths)
    return config, SimpleNamespace(metadata=dict(signature_bank=dict(sha256='b' * 64)))


def write_run(path, *, centers=('t1',), split='train', complete=True, bundle='m' * 64):
    path.mkdir()
    (path / 'CONTRACT.json').write_text(json.dumps(dict(
        dataset='toy', backbone='gcn', explained_model_sha256='g' * 64, model_bundle_sha256=bundle, signature_bank_sha256='b' * 64,
        population=dict(split=split, population_id='p'), run_contract_id='r')))
    rows, journal = [], []
    for i, center in enumerate(centers):
        event = f'r:{i}:1:2'
        rows.append(dict(center_id=center, run_contract_id='r', events=[
            dict(stage='q0', deleted=[], flip=False, physical_event_id=f'r:{i}:1:1'),
            dict(stage='certification', deleted=['e2', 'e1'], flip=True, physical_event_id=event)]))
        journal += [dict(event='INTENT', physical_event_id=event, deleted=['e2', 'e1'])]
        journal += [dict(event='COMPLETE', physical_event_id=event)] if complete else []
    (path / 'REQUESTS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (path / 'PHYSICAL_EVENTS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal))
    return path


def test_certification_labels_come_only_from_journaled_real_gnn_calls_on_train_targets(tmp_path):
    config, models = toy_tuple(tmp_path)
    labeled, logs, isolation = certification_labels(config, models, [write_run(tmp_path / 'ok', centers=('t1', 't3'))],
                                                    bundle_sha256='m' * 64)
    assert [(r['row']['center_id'], r['labels']) for r in labeled] == [('t1', [(('e1', 'e2'), True)]), ('t3', [(('e1', 'e2'), True)])]
    assert isolation['evaluation_populations'] == dict(dev=dict(ids=2, shared_with_train=0), test1000=dict(ids=1, shared_with_train=0))
    for name, run, message in (('dev', dict(split='val'), 'not a run on TRAIN targets'),
                               ('evaluation', dict(centers=('t1', 'd1')), r"belongs to evaluation population\(s\) \['dev'\]"),
                               ('unknown', dict(centers=('zz',)), 'zz is not a TRAIN target'),
                               ('journal', dict(complete=False), 'no journaled real-GNN call'),
                               ('bundle', dict(bundle='x' * 64), 'another model or graph')):
        with pytest.raises(ValueError, match=message):
            certification_labels(config, models, [write_run(tmp_path / name, **run)], bundle_sha256='m' * 64)


def test_training_population_must_not_meet_any_evaluation_population(tmp_path):
    config, _ = toy_tuple(tmp_path, train='t1 d2 t3')
    with pytest.raises(ValueError, match=r"shares target IDs with evaluation populations \{'dev': 1\}; training stops"):
        training_targets(config)
    config, _ = toy_tuple(tmp_path / 'twice', train='t1 t2 t1')
    with pytest.raises(ValueError, match='twice'):
        training_targets(config)
    config, _ = toy_tuple(tmp_path / 'ordered', train=' '.join(f't{i}' for i in range(20)))
    ordered, isolation = training_targets(config)
    assert ordered == [f't{i}' for i in range(20)] and isolation['sampling_seed'] is None
    config.dataset_config['training']['sampling_seed'] = 5
    shuffled, again = training_targets(config)[0], training_targets(config)[0]
    assert shuffled == again != ordered and sorted(shuffled) == sorted(ordered)
