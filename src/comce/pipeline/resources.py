"""External graph data and explained-GNN checkpoints, located through yaml.template."""
from pathlib import Path

from comce.datasets import GRAPH_ADAPTERS
from comce.pipeline.io import digest, sha256


def class_vocabulary_sha256(vocabulary):
    import hashlib
    import json
    return hashlib.sha256(json.dumps([str(name) for name in vocabulary], separators=(',', ':')).encode()).hexdigest()


def verify_files(root, pins, key):
    """pins maps file names inside the directory of the YAML key to their SHA256."""
    for name, expected in pins.items():
        path = Path(root) / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing external resource {path}; check {key}')
        if sha256(path) != expected:
            raise ValueError(f'External resource digest mismatch: {path}')


def check_decision(model, declared):
    spec = model.decision_spec
    frozen = dict(kind=spec.kind, num_classes=int(spec.num_classes), threshold=float(spec.threshold),
                  positive_class=int(spec.positive_class), deployment_temperature=float(model.deployment_temperature))
    mismatched = {k: (declared.get(k), v) for k, v in frozen.items() if declared.get(k) != v}
    if mismatched:
        raise ValueError(f'Configured decision disagrees with the explained model: {mismatched}')
    expected = declared.get('class_vocabulary_sha256') or class_vocabulary_sha256(declared['class_vocabulary'])
    if expected != class_vocabulary_sha256(model.class_vocabulary):
        raise ValueError('Class vocabulary digest mismatch')


def load_explained_model(config, *, allowed_ids, device, explained_model_sha256):
    """Graph adapter for the dataset plus the sealed explained model M of the selected backbone."""
    paths = config.paths
    paths.require([paths.key('resources.data'), paths.tuple_key('gnn.checkpoint')])
    resources = config.resources
    pins = resources.get('files', {})
    if config.verify_checksums:
        verify_files(resources['data'], pins, paths.key('resources.data'))
    gnn_path = config.gnn_path
    actual = sha256(gnn_path)
    if actual != config.tuple_config['gnn']['sha256'] or actual != explained_model_sha256:
        raise ValueError(f'Explained GNN checkpoint {gnn_path} is not the model the ComCE bundle was trained for')
    adapter = config.dataset_config['adapter']
    io = GRAPH_ADAPTERS[adapter['name']](resources, adapter['options'], device=device, allowed_ids=allowed_ids,
                                         identity=digest(pins))
    model = io.load_model(gnn_path, config.dataset_config['decision'], device=device)
    check_decision(model, config.dataset_config['decision'])
    return io, model
