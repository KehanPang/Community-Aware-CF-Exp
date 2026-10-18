import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['MKL_THREADING_LAYER'] = 'GNU'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def pytest_addoption(parser):
    group = parser.getgroup('comce', 'external ComCE resources; tests that read them are skipped without these options')
    group.addoption('--comce-config', type=Path, help='YAML configuration whose resource paths are set')
    group.addoption('--comce-reference', type=Path, help='JSON of the expected DEV request of every dataset/backbone')
    group.addoption('--sbert-model', type=Path, help='local Sentence Transformer directory')


def _option(request, name):
    value = request.config.getoption(name)
    if value is None:
        pytest.skip(f'needs {name}')
    return Path(value).resolve()


@pytest.fixture(scope='session')
def resource_template(request):
    from comce.pipeline.config import load_template
    return load_template(_option(request, '--comce-config'))


@pytest.fixture(scope='session')
def reference(request):
    import json
    return json.loads(_option(request, '--comce-reference').read_text())


@pytest.fixture(scope='session')
def sbert_model(request):
    return _option(request, '--sbert-model')
