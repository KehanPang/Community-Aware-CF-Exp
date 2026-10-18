"""Stage 1 entry points: validate a dataset's field declaration against its data (writing an editable template when it
is missing) before its signature bank is fitted."""
import tempfile
from pathlib import Path

from comce.datasets import GRAPH_SCHEMAS
from comce.pipeline.config import ResourceConfigurationError
from comce.pipeline.io import PROJECT_ROOT
from comce.stage1 import fields
from comce.stage1.fields import Stage1ConfigurationError


def outputs_root(template):
    """outputs.root (absolute, or relative to the configuration file), or None when it is empty."""
    value = (template.get('outputs') or {}).get('root')
    if not value:
        return None
    root = Path(value)
    return root if root.is_absolute() else Path(template['_path']).parent / root


def outside_release(path, what):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f'{what} must be outside the release: {path}')
    return path


def evaluation_populations(paths):
    """Target IDs of every population a TRAIN set must not intersect (datasets.<d>.training.evaluation_populations)."""
    names = (paths.dataset_config.get('training') or {}).get('evaluation_populations') or []
    populations = paths.dataset_config['resources'].get('populations', {})
    unknown = [n for n in names if n not in populations]
    if unknown:
        raise ValueError(f'evaluation populations {unknown} are not declared under resources.populations')
    return {n: paths.population(n).read_text().split() for n in names}


def dataset_schema(paths):
    adapter = paths.dataset_config['adapter']
    resources = dict(paths.dataset_config['resources'], data=str(paths.resource('data')))
    return GRAPH_SCHEMAS[adapter['name']](resources, adapter['options'])


def stage1_resource_keys(paths, stage1):
    """YAML keys of the files a Stage 1 fit reads: the data directory, the evaluation populations and a TRAIN seed list."""
    train_nodes = stage1.get('train_nodes')
    source = (train_nodes.get('seed') or {}).get('source') if isinstance(train_nodes, dict) else None
    seed = [paths.key(f'resources.{source["resource"]}')] if isinstance(source, dict) and set(source) == {'resource'} else []
    return [paths.key('resources.data'), *paths.evaluation_keys(), *seed]


def require_encoder_models(paths, stage1):
    """An sbert field names a local Sentence Transformer directory; an empty model is an error naming its key."""
    node_types = stage1.get('node_types') or {}
    empty = [paths.key(f'stage1.node_types.{t}.fields[{i}].model') for t, spec in node_types.items() if isinstance(spec, dict)
             for i, field in enumerate(spec.get('fields') or [])
             if isinstance(field, dict) and field.get('encoder') == 'sbert' and not str(field.get('model') or '').strip()]
    if empty:
        raise ResourceConfigurationError(empty)


def stage1_plan(template, dataset, stage1, paths, *, template_output=None):
    """Validated Stage 1 plan. A missing or incomplete declaration stops here, before anything is encoded, and leaves a
    template of the node types and fields found in the data at template_output, else under outputs.root, else in a new
    system temporary directory (an existing file is never replaced)."""
    paths.require(stage1_resource_keys(paths, stage1))
    require_encoder_models(paths, stage1)
    adapter = paths.dataset_config['adapter']
    data_root = paths.resource('data')
    trusted = {v for v in adapter['options'].values() if isinstance(v, str) and Path(v).suffix in fields.SUFFIXES}
    sources = fields.FieldSource(data_root, trusted)
    schema = dataset_schema(paths)
    try:
        return fields.validate(stage1, schema, sources, resource=paths.resource, evaluation=evaluation_populations(paths))
    except Stage1ConfigurationError as error:
        if not error.missing:
            raise
        root = outputs_root(template)
        name = f'{dataset}.stage1.yaml'
        path = outside_release(template_output or (root / 'stage1' / name if root is not None else
                                                   Path(tempfile.mkdtemp(prefix='comce-stage1-')) / name), 'the Stage 1 template')
        if path.exists():
            note = f'the template already at {path} was left unchanged'
        else:
            fields.write_template(path, dataset=dataset, schema=schema, found=fields.scan_fields(sources, schema), stage1=stage1)
            note = f'a template of the node types and fields found in the data was written to {path}'
        raise Stage1ConfigurationError(
            f'{error} for dataset {dataset!r} (datasets.{dataset}.stage1). Nothing was encoded or fitted; {note}. Choose an '
            f'encoder for every field to use, copy the block into the configuration and run again.',
            missing=True, template_path=path) from None

