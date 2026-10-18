import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from comce.pipeline.config import load_template, resolve
from comce.pipeline.run import run_population

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / 'yaml.template')
TAGS = [f'{d}/{b}' for d in TEMPLATE['tuples'] for b in TEMPLATE['tuples'][d]]
SCIENTIFIC = ('center_id', 'success', 'returned_edge_keys', 'total_queries', 'macro_domain_edge_keys', 'pool_size',
              'community_count', 'q0', 'events', 'attempts', 'cleanup_attempts', 'selection_trace', 'stage3_stats', 'query_ledger',
              'exact_certified', 'true_e123')
REFERENCE = ('success', 'returned_edge_keys', 'total_queries', 'macro_domain_edge_keys', 'pool_size', 'community_count')

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def run(config, center, tmp_path, name='run'):
    run_population(config, ids=[center], split='val', result_dir=tmp_path / name)
    return json.loads((tmp_path / name / 'REQUESTS.jsonl').read_text().splitlines()[0])


def scientific(row):
    out = {k: row[k] for k in SCIENTIFIC if k in row}
    out['events'] = [{k: v for k, v in e.items() if k not in ('physical_event_id', 'oracle_wall')} for e in out['events']]
    return out


@pytest.mark.parametrize('tag', TAGS)
def test_frozen_dev_request_is_reproduced_and_certified(tag, tmp_path, resource_template, reference):
    dataset, backbone = tag.split('/')
    expected = reference[tag]
    row = run(resolve(resource_template, dataset, backbone), expected['center_id'], tmp_path)
    for key in REFERENCE:
        assert row[key] == expected[key], key
    assert row['q0']['probs'] == expected['q0_probs'] and row['stage3_stats']['bit_words'] == expected['bit_words']
    assert row['actual_blackbox_forwards'] == row['total_queries'] <= 12
    if row['success']:
        assert row['exact_certified']
        assert any(e['flip'] and set(e['deleted']) == set(row['returned_edge_keys']) for e in row['events'])


@pytest.mark.parametrize('dataset', ['claim', 'computer', 'goodreads'])
def test_explanations_never_read_hidden_activations(dataset, tmp_path, monkeypatch, resource_template, reference):
    from comce.datasets import backbones
    backbone = 'hgt' if dataset == 'goodreads' else 'gcn'
    expected = reference[f'{dataset}/{backbone}']

    class Poison(dict):
        def __getitem__(self, key):
            raise AssertionError('explanation code read a hidden activation of the explained model')
        items = values = keys = __iter__ = lambda self: (_ for _ in ()).throw(AssertionError('hidden activation accessed'))
    original = backbones.StandardBackbone.forward

    def poisoned(self, graph):
        logits, _ = original(self, graph)
        return logits, Poison()
    monkeypatch.setattr(backbones.StandardBackbone, 'forward', poisoned)
    row = run(resolve(resource_template, dataset, backbone), expected['center_id'], tmp_path)
    assert {k: row[k] for k in REFERENCE} == {k: expected[k] for k in REFERENCE}
    assert row['q0']['probs'] == expected['q0_probs'] and row['stage3_stats']['bit_words'] == expected['bit_words']


def renamed_template(template, dataset, backbone, alias_dataset, alias_backbone, tmp_path, root=None):
    document = yaml.safe_load(Path(template['_path']).read_text())
    document['resources']['root'] = str(root or resolve(template, dataset, backbone).paths.root)
    document['datasets'] = {alias_dataset: document['datasets'][dataset]}
    document['tuples'] = {alias_dataset: {alias_backbone: copy.deepcopy(document['tuples'][dataset][backbone])}}
    path = tmp_path / 'renamed.yaml'
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_template(path)


@pytest.mark.parametrize('dataset,backbone', [tuple(tag.split('/')) for tag in TAGS])
def test_dataset_and_backbone_names_do_not_change_the_algorithm(dataset, backbone, tmp_path, resource_template, reference):
    center = reference[f'{dataset}/{backbone}']['center_id']
    original = run(resolve(resource_template, dataset, backbone), center, tmp_path, 'original')
    template = renamed_template(resource_template, dataset, backbone, 'anonymous_graph', 'anonymous_gnn', tmp_path)
    renamed = run(resolve(template, 'anonymous_graph', 'anonymous_gnn'), center, tmp_path, 'renamed')
    assert renamed['dataset'] == 'anonymous_graph' and renamed['backbone'] == 'anonymous_gnn'
    assert scientific(renamed) == scientific(original)


def test_relocated_resource_root_gives_identical_outputs(tmp_path, resource_template, reference):
    dataset, backbone = 'goodreads', 'hgt'
    center = reference[f'{dataset}/{backbone}']['center_id']
    original = run(resolve(resource_template, dataset, backbone), center, tmp_path, 'original')
    moved = tmp_path / 'elsewhere'
    moved.symlink_to(resolve(resource_template, dataset, backbone).paths.root, target_is_directory=True)
    template = renamed_template(resource_template, dataset, backbone, dataset, backbone, tmp_path, root=moved)
    relocated = run(resolve(template, dataset, backbone), center, tmp_path, 'relocated')
    assert scientific(relocated) == scientific(original)
