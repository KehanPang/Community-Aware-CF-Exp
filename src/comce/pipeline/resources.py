"""External graph data and explained-GNN checkpoints, located through yaml.template."""
from comce.datasets import GRAPH_ADAPTERS


def check_decision(model, declared):
    spec = model.decision_spec
    frozen = dict(kind=spec.kind, num_classes=int(spec.num_classes), threshold=float(spec.threshold),
                  positive_class=int(spec.positive_class), deployment_temperature=float(model.deployment_temperature))
    mismatched = {k: (declared.get(k), v) for k, v in frozen.items() if declared.get(k) != v}
    if mismatched:
        raise ValueError(f'Configured decision disagrees with the explained model: {mismatched}')


def load_explained_model(config, *, allowed_ids, device):
    """Graph adapter for the dataset plus the explained model M of the selected backbone."""
    paths = config.paths
    paths.require([paths.key('resources.data'), paths.tuple_key('gnn')])
    adapter = config.dataset_config['adapter']
    io = GRAPH_ADAPTERS[adapter['name']](config.resources, adapter['options'], device=device, allowed_ids=allowed_ids)
    model = io.load_model(config.gnn_path, config.dataset_config['decision'], device=device)
    check_decision(model, config.dataset_config['decision'])
    return io, model
