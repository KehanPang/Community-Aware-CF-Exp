from pathlib import Path

import numpy as np
import pytest
import torch

from comce.pipeline.config import load_template, resolve, resolve_stage1
from comce.pipeline.features import stage1_plan
from comce.pipeline.models import BUNDLE_FORMAT, load_models
from comce.stage1 import signatures
from comce.stage1.encoders import fit_signature_bank
from comce.stage3.world_model import STATES, WorldModel

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / 'yaml.template')
TUPLES = [(d, b) for d in TEMPLATE['tuples'] for b in TEMPLATE['tuples'][d]]


def test_configuration_declares_nine_tuples():
    assert sorted(TUPLES) == sorted((d, b) for d in ('claim', 'computer', 'goodreads') for b in ('gcn', 'graphsage', 'hgt'))


@pytest.mark.parametrize('dataset,backbone', TUPLES)
def test_bundle_strict_load_architecture_and_bindings(dataset, backbone, resource_template):
    config = resolve(resource_template, dataset, backbone)
    models = load_models(config.model_path, bank_path=config.signature_bank_path, device='cpu')
    meta = models.metadata
    assert meta['format'] == BUNDLE_FORMAT and (meta['dataset'], meta['backbone']) == (dataset, backbone)
    assert meta['explained_model_sha256'] == config.tuple_config['gnn']['sha256']
    assert {type(models.world_model[s]) for s in STATES} == {WorldModel}
    architecture = meta['architecture']['set_transformer']
    stage2 = config.parameters['stage2']
    assert (architecture['num_pattern_slots'], architecture['num_heads'], architecture['d_model']) == (
        stage2['query_slots'], stage2['attention_heads'], stage2['hidden_width'])
    edge_in = meta['architecture']['world_model']['edge_in']
    assert edge_in == len(meta['input_schema']['edge_rows']) == 25 + len(config.dataset_config['relations'])
    assert meta['architecture']['ranking_residual']['edge_in'] == edge_in
    payload = torch.load(config.model_path, map_location='cpu', weights_only=True)
    for state in STATES:
        assert set(payload['world_model'][state]) == set(models.world_model[state].state_dict())
    assert meta['training']['hidden_states_used'] is False
    bank = torch.load(config.signature_bank_path, map_location='cpu', weights_only=True)
    assert bank['schema'] == signatures.SCHEMA and tuple(bank['thresholds'].shape) == (signatures.DIMENSIONS, signatures.BUCKETS - 1)


@pytest.mark.parametrize('dataset', ['claim', 'computer', 'goodreads'])
def test_signature_bank_refits_from_the_stage1_declaration_exactly(dataset, resource_template):
    stage1, paths = resolve_stage1(resource_template, dataset)
    saved = signatures.SignatureBank.load(paths.resource('signature_bank'))
    assert saved.provenance['model_hidden_states_used'] is False and saved.provenance['backbone_independent'] is True
    plan = stage1_plan(resource_template, dataset, stage1, paths)
    again = fit_signature_bank(plan)
    assert again.provenance['train_nodes'] == {t: len(r) for t, r in plan.train_rows.items()}
    np.testing.assert_array_equal(again.thresholds, saved.thresholds)
    assert set(again.coordinates) == set(saved.coordinates)
    for node_type, values in saved.coordinates.items():
        np.testing.assert_array_equal(again.coordinates[node_type], values)


def test_quantize_counts_the_thresholds_below_each_coordinate():
    from comce.stage2.operators import quantize
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(50, 8))
    thresholds = signatures.quantile_thresholds(coords)
    np.testing.assert_array_equal(quantize(coords, thresholds), (coords[:, :, None] > thresholds[None]).sum(-1))
    directions = signatures.random_directions(16, 20261017)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=0), 1.0)
    np.testing.assert_array_equal(directions, signatures.random_directions(16, 20261017))


def test_resource_pins_name_files_inside_the_data_directory():
    for dataset, spec in TEMPLATE['datasets'].items():
        pins = spec['resources']['files']
        assert pins and all(len(digest) == 64 for digest in pins.values())
        assert all(Path(name).name == name for name in pins)
        for backbone, tuple_config in TEMPLATE['tuples'][dataset].items():
            assert len(tuple_config['gnn']['sha256']) == 64
    assert TEMPLATE['resources']['verify_checksums'] is True
