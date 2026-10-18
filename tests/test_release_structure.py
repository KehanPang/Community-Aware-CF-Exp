import ast
import re
from pathlib import Path

import pytest
import yaml

from comce.stage2.operators import FeatureProgram

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'comce'
CORE = [p for d in ('stage2', 'stage3', 'pipeline', 'training') for p in (SRC / d).rglob('*.py')]
DATA_OR_WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npy', '.npz', '.parquet', '.pkl', '.pickle', '.h5',
                   '.onnx', '.csv', '.tsv', '.jsonl', '.json', '.zip', '.tar', '.gz'}


def test_readme_is_present_and_empty_and_no_prose_documents_ship():
    assert (ROOT / 'README.md').is_file() and (ROOT / 'README.md').stat().st_size == 0
    prose = [p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in {'.md', '.rst', '.ipynb', '.log'}
             and p.name != 'README.md']
    assert prose == []
    assert (ROOT / 'ComCE.pdf').is_file()


def test_no_data_model_weights_or_signature_banks_in_the_release():
    files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert [p.relative_to(ROOT).as_posix() for p in files if p.suffix.lower() in DATA_OR_WEIGHTS] == []
    assert [p.relative_to(ROOT).as_posix() for p in files if p.suffix.lower() == '.txt'] == ['requirements.txt']
    assert not (ROOT / 'models').exists() and not (ROOT / 'tests' / 'data').exists()
    assert not [p for p in ROOT.rglob('*') if p.is_symlink()]


def test_public_configuration_leaves_every_resource_path_empty():
    template = yaml.safe_load((ROOT / 'yaml.template').read_text())
    locations = [('resources', 'root'), ('outputs', 'root')]
    for name, dataset in template['datasets'].items():
        resources = dataset['resources']
        locations += [('datasets', name, 'resources', key) for key in resources if key not in ('populations', 'files')]
        locations += [('datasets', name, 'resources', 'populations', key) for key in resources['populations']]
        locations.append(('datasets', name, 'training', 'population_path'))
        for backbone in template['tuples'][name]:
            locations += [('tuples', name, backbone, 'model'), ('tuples', name, backbone, 'gnn', 'checkpoint')]
    for location in locations:
        value = template
        for part in location:
            value = value[part]
        assert value == '', '.'.join(location)
    assert {key for d in template['datasets'].values() for key in d['resources']} == {
        'data', 'signature_bank', 'populations', 'files', 'train_centers'}


def test_core_stages_name_no_dataset_or_backbone():
    names = re.compile(r'(?i)\b(claim|computer|goodreads|gcn|graphsage|sage|hgt|std_gcn|std_graphsage|std_hgt|book|review|paper|author)\b')
    for path in CORE:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split('#', 1)[0]
            assert not names.search(code), f'{path.relative_to(ROOT)}:{number}: {line.strip()}'


def test_core_never_branches_on_dataset_or_backbone_identity():
    # pipeline/config.py looks datasets up by name in yaml.template; every other core module must not branch on identity.
    for path in [p for p in CORE if p.name != 'config.py']:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp, ast.Compare)):
                text = ast.unparse(node.test if isinstance(node, (ast.If, ast.IfExp)) else node)
                assert not re.search(r'\b(dataset|backbone|gnn_type|feature_dim)\b', text), f'{path}: {text}'


def test_configuration_selects_no_algorithm_variant():
    template = yaml.safe_load((ROOT / 'yaml.template').read_text())
    forbidden = re.compile(r'(?i)^(mode|.*_mode|profile|variant|.*_variant|algorithm|.*_algorithm|.*_type|region_program|scorer|ranker)$')

    def walk(value, path=''):
        if isinstance(value, dict):
            for key, item in value.items():
                # train_nodes.seed.node_type names the node type whose TRAIN vertices seed the bucketizer fit.
                if path.split('.')[-1] not in ('adapter', 'options', 'edit_cost', 'gnn', 'seed') and forbidden.match(str(key)):
                    raise AssertionError(f'{path}.{key}')
                walk(item, f'{path}.{key}')
    walk({k: v for k, v in template.items() if k != 'datasets'})
    for name, dataset in template['datasets'].items():
        walk({k: v for k, v in dataset.items() if k != 'adapter'}, f'datasets.{name}')
    assert len(list((SRC / 'stage2').glob('*.yaml'))) == 1


def test_comce_is_imported_from_this_repository():
    import comce
    assert Path(comce.__file__).resolve().is_relative_to(ROOT.resolve())


@pytest.mark.parametrize('source', ["__import__('os')", 'x.__class__', '(lambda: 1)()', '[v for v in x]'])
def test_feature_program_has_no_python_execution_escape(source):
    with pytest.raises((ValueError, KeyError, AttributeError)):
        FeatureProgram({'x': [1]}).evaluate(source)
