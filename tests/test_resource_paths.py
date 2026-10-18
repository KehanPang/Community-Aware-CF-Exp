import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from comce.pipeline.config import ResourceConfigurationError, ResourcePaths, load_template, resolve, resolve_stage1
from comce.pipeline.explainer import Explainer
from comce.pipeline.features import require_encoder_models, stage1_plan
from comce.pipeline.run import run_population
from comce.training.trainer import TupleTrainer

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / 'yaml.template')
TUPLES = [(d, b) for d in TEMPLATE['tuples'] for b in TEMPLATE['tuples'][d]]
SCRIPTS = ('run_comce', 'fit_signatures', 'train_comce', 'update_world_model')


@pytest.fixture
def nothing_is_loaded(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a resource was loaded although its path is empty')
    monkeypatch.setattr(torch, 'load', forbidden)


def release_files():
    return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if '__pycache__' not in p.parts)


def toy_paths(tmp_path, **resources):
    template = dict(_path=str(tmp_path / 'config.yaml'), resources=dict(root=resources.pop('root', '')),
                    datasets=dict(toy=dict(resources=resources)))
    return ResourcePaths(template, 'toy')


@pytest.mark.parametrize('dataset,backbone', TUPLES)
def test_inference_names_every_empty_key_before_loading(dataset, backbone, tmp_path, nothing_is_loaded):
    config = resolve(TEMPLATE, dataset, backbone)
    expected = [f'datasets.{dataset}.resources.data', f'datasets.{dataset}.resources.signature_bank',
                f'tuples.{dataset}.{backbone}.gnn.checkpoint', f'tuples.{dataset}.{backbone}.model']
    with pytest.raises(ResourceConfigurationError) as error:
        Explainer(config, allowed_ids={'0': 'val'}, device='cpu')
    assert error.value.keys == expected and all(key in str(error.value) for key in expected)
    with pytest.raises(ResourceConfigurationError):
        run_population(config, ids=['0'], split='val', result_dir=tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
    for attribute in ('model_path', 'gnn_path', 'signature_bank_path'):
        with pytest.raises(ResourceConfigurationError):
            getattr(config, attribute)


@pytest.mark.parametrize('dataset', ['claim', 'computer', 'goodreads'])
def test_training_and_stage1_name_their_empty_keys_before_loading(dataset, tmp_path, nothing_is_loaded):
    config = resolve(TEMPLATE, dataset, 'gcn')
    with pytest.raises(ResourceConfigurationError) as error:
        TupleTrainer(config, TEMPLATE, device='cpu', work=tmp_path / 'work', signature_bank=tmp_path / 'bank.pt')
    assert f'datasets.{dataset}.training.population_path' in error.value.keys and not (tmp_path / 'work').exists()
    stage1, paths = resolve_stage1(TEMPLATE, dataset)
    with pytest.raises(ResourceConfigurationError) as error:
        stage1_plan(TEMPLATE, dataset, stage1, paths, template_output=tmp_path / 'stage1.yaml')
    assert f'datasets.{dataset}.resources.data' in error.value.keys and not (tmp_path / 'stage1.yaml').exists()


def test_paths_are_absolute_or_relative_to_the_resource_root_and_never_guessed(tmp_path):
    with pytest.raises(ResourceConfigurationError) as error:
        toy_paths(tmp_path, data='data').resource('data')
    assert error.value.keys == ['resources.root'] and 'datasets.toy.resources.data' in str(error.value)
    assert toy_paths(tmp_path, data=str(tmp_path / 'd')).resource('data') == tmp_path / 'd'
    assert toy_paths(tmp_path, data='d', root=str(tmp_path / 'r')).resource('data') == tmp_path / 'r' / 'd'
    assert toy_paths(tmp_path, data='d', root='r').resource('data') == tmp_path / 'r' / 'd'
    for value in ('', '   ', None):
        with pytest.raises(ResourceConfigurationError, match='datasets.toy.resources.data'):
            toy_paths(tmp_path, data=value).resource('data')
    with pytest.raises(ResourceConfigurationError) as error:
        toy_paths(tmp_path, data='', signature_bank='').require(['datasets.toy.resources.data', 'datasets.toy.resources.signature_bank'])
    assert error.value.keys == ['datasets.toy.resources.data', 'datasets.toy.resources.signature_bank']


def test_an_sbert_field_without_a_model_path_is_a_resource_error(tmp_path):
    stage1 = dict(node_types=dict(item=dict(fields=[dict(name='title', encoder='sbert', model='', output_dimension=8)])))
    with pytest.raises(ResourceConfigurationError, match=r'datasets.toy.stage1.node_types.item.fields\[0\].model'):
        require_encoder_models(toy_paths(tmp_path), stage1)
    require_encoder_models(toy_paths(tmp_path), dict(node_types=dict(item=dict(fields=[dict(stage1['node_types']['item']['fields'][0],
                                                                                         model='local')]))))


def blank_run(tmp_path, script, *arguments, optimize=False):
    command = [sys.executable, *(['-O'] if optimize else []), str(ROOT / 'scripts' / f'{script}.py'),
               '--config', str(ROOT / 'yaml.template'), *arguments]
    return subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))


@pytest.mark.parametrize('script', SCRIPTS)
def test_help_needs_no_resources(script, tmp_path):
    result = blank_run(tmp_path, script, '--help')
    assert result.returncode == 0 and 'usage:' in result.stdout


@pytest.mark.parametrize('script,arguments,keys', [
    ('run_comce', ['--dataset', 'claim', '--backbone', 'gcn', '--target-id', '1'],
     ['tuples.claim.gcn.model', 'tuples.claim.gcn.gnn.checkpoint', 'datasets.claim.resources.data',
      'datasets.claim.resources.signature_bank', 'outputs.root']),
    ('run_comce', ['--dataset', 'goodreads', '--backbone', 'hgt', '--population-name', 'dev'],
     ['datasets.goodreads.resources.populations.dev', 'outputs.root']),
    ('fit_signatures', ['--dataset', 'claim'],
     ['datasets.claim.resources.data', 'datasets.claim.resources.populations.test1000', 'outputs.root']),
    ('fit_signatures', ['--dataset', 'goodreads', '--output', 'bank.pt'], ['datasets.goodreads.resources.train_centers']),
    ('train_comce', ['--dataset', 'computer', '--backbone', 'hgt', '--models-root', 'models', '--work', 'work'],
     ['tuples.computer.hgt.gnn.checkpoint', 'datasets.computer.training.population_path',
      'datasets.computer.resources.signature_bank']),
    ('update_world_model', ['--dataset', 'claim', '--backbone', 'hgt', '--certification-log', 'log', '--output', 'update.pt'],
     ['tuples.claim.hgt.model', 'datasets.claim.resources.populations.dev_full', 'datasets.claim.training.population_path']),
])
def test_every_entry_point_stops_on_an_empty_path_and_writes_nothing(script, arguments, keys, tmp_path):
    before = release_files()
    for optimize in (False, True):
        result = blank_run(tmp_path, script, *arguments, optimize=optimize)
        assert result.returncode == 1 and 'ResourceConfigurationError' in result.stderr, result.stderr
        assert 'Traceback' not in result.stderr and all(key in result.stderr for key in keys)
    assert list(tmp_path.iterdir()) == [] and release_files() == before


def test_a_configured_population_name_must_be_declared(tmp_path):
    result = blank_run(tmp_path, 'run_comce', '--dataset', 'claim', '--backbone', 'gcn', '--population-name', 'nowhere')
    assert result.returncode == 2 and 'unknown population' in result.stderr


def test_a_relative_resource_root_resolves_against_its_configuration_file(tmp_path):
    raw = yaml.safe_load((ROOT / 'yaml.template').read_text())
    raw['resources']['root'] = 'assets'
    raw['datasets']['claim']['resources']['data'] = 'claim'
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'c.yaml').write_text(yaml.safe_dump(raw))
    config = resolve(load_template(tmp_path / 'config' / 'c.yaml'), 'claim', 'gcn')
    assert config.paths.resource('data') == tmp_path / 'config' / 'assets' / 'claim'
