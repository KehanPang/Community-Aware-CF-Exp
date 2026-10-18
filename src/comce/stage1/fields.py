"""Stage 1 field declarations (datasets.<d>.stage1 in yaml.template): which field of which node type is encoded, how.

Raw field -> feature representation x_v (one encoder per field, one aggregation per node type, an optional reducer
to the representation dimension) -> shared signature projection r_1..r_ds -> one quantile bucketizer fitted on TRAIN
nodes. Encoders: sbert (text field, pre-trained Sentence Transformer), lsh (numeric field, TRAIN standardization and a
seeded Gaussian random projection), precomputed (an existing embedding, used as it is), neighbor_mean (the mean of a
neighbouring node type's representation along one relation).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ENCODERS = {
    'sbert': ({'name', 'encoder', 'source', 'model', 'output_dimension'}, {'normalize', 'batch_size', 'device'}),
    'lsh': ({'name', 'encoder', 'source', 'input_kind', 'standardize', 'output_dimension', 'seed'}, set()),
    'precomputed': ({'name', 'encoder', 'source', 'output_dimension'}, set()),
    'neighbor_mean': ({'name', 'encoder', 'relation', 'empty'}, set()),
}
ENCODER_INPUT = {'sbert': {'text'}, 'lsh': {'numeric_vector', 'numeric_scalar'}, 'precomputed': {'numeric_vector'}}
# Keys that schedule a field but never change its values; they are left out of the feature schema.
NON_VALUE_KEYS = {'batch_size', 'device'}
SCHEMA_KEYS = ('projection_dimensions', 'quantile_buckets', 'representation_dimension', 'fit_split', 'projection_seed',
               'reducer', 'node_types', 'train_nodes')
SUFFIXES = {'.npy', '.npz', '.pt', '.parquet', '.txt'}


class Stage1ConfigurationError(ValueError):
    """Stage 1 cannot run: field encoding not declared (missing=True) or declared inconsistently with the data."""

    def __init__(self, message, *, missing=False, template_path=None):
        super().__init__(message)
        self.missing = missing
        self.template_path = template_path


def feature_schema(stage1):
    """Canonical description of every Stage 1 choice that changes the signature coordinates."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in sorted(value.items()) if k not in NON_VALUE_KEYS}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    return clean({k: stage1.get(k) for k in SCHEMA_KEYS})


def declared(stage1):
    return bool(stage1.get('node_types'))


def value_kind(value):
    if isinstance(value, list):
        if value and all(isinstance(v, str) for v in value):
            return 'text'
        if value and all(isinstance(v, (int, float, bool)) and not isinstance(v, str) for v in value):
            return 'numeric_scalar'
        return 'other'
    array = np.asarray(value) if not isinstance(value, np.ndarray) else value
    if array.dtype.kind in 'USO':
        return 'text' if array.ndim == 1 and (array.dtype.kind in 'US' or all(isinstance(v, str) for v in array[:100])) else 'other'
    if array.dtype.kind in 'biuf':
        return {1: 'numeric_scalar', 2: 'numeric_vector'}.get(array.ndim, 'other')
    return 'other'


class FieldSource:
    """Reads one declared field: {file} | {file, key} (dotted path in .pt/.npz) | {file, column | columns} (.parquet)."""

    def __init__(self, data_root, trusted_files):
        self.root = Path(data_root)
        self.trusted = set(map(str, trusted_files))
        self.loaded = {}

    def _container(self, name):
        path = self.root / name
        if not path.is_file():
            raise Stage1ConfigurationError(f'field source {path} does not exist')
        if name not in self.loaded:
            if path.suffix == '.pt':
                try:
                    self.loaded[name] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                except Exception as exc:
                    # Only a file the input adapter already unpickles may hold more than tensors and containers.
                    if name not in self.trusted:
                        raise Stage1ConfigurationError(f'{path} is not a plain tensor file: {exc}') from exc
                    self.loaded[name] = torch.load(path, map_location='cpu', weights_only=False)
            elif path.suffix == '.npz':
                self.loaded[name] = np.load(path)
            else:
                self.loaded[name] = None
        return path, self.loaded[name]

    def read(self, spec):
        if not isinstance(spec, dict) or not isinstance(spec.get('file'), str):
            raise Stage1ConfigurationError(f'field source must be a mapping with a file: {spec!r}')
        extra = set(spec) - {'file', 'key', 'column', 'columns'}
        if extra:
            raise Stage1ConfigurationError(f'unknown field source key(s) {sorted(extra)} in {spec!r}')
        path, container = self._container(spec['file'])
        if path.suffix not in SUFFIXES:
            raise Stage1ConfigurationError(f'{path.name}: field sources are .npy, .npz, .pt, .parquet or .txt files')
        if path.suffix == '.npy':
            self._no_keys(spec, path)
            return np.load(path, mmap_mode='r')
        if path.suffix == '.txt':
            self._no_keys(spec, path)
            return path.read_text().splitlines()
        if path.suffix == '.parquet':
            columns = spec.get('columns') or ([spec['column']] if 'column' in spec else None)
            if not columns or 'key' in spec:
                raise Stage1ConfigurationError(f'{path.name}: a parquet source names column or columns')
            import pyarrow.parquet as pq
            available = pq.read_schema(path).names
            missing = [c for c in columns if c not in available]
            if missing:
                raise Stage1ConfigurationError(f'{path.name}: column(s) {missing} do not exist; available: {available}')
            frame = pd.read_parquet(path, columns=list(columns))
            if 'column' in spec:
                values = frame[spec['column']]
                return values.astype(object).tolist() if values.dtype == object or pd.api.types.is_string_dtype(values) else values.to_numpy()
            return frame.to_numpy()
        if 'key' not in spec or 'column' in spec or 'columns' in spec:
            raise Stage1ConfigurationError(f'{path.name}: a {path.suffix} source names a key')
        value = container
        for part in str(spec['key']).split('.'):
            options = list(value.keys()) if hasattr(value, 'keys') else []
            if part not in options:
                raise Stage1ConfigurationError(f'{path.name}: key {spec["key"]!r} does not exist ({part!r} not in {options})')
            value = value[part]
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        if isinstance(value, (tuple, list)):
            return list(value)
        if not isinstance(value, np.ndarray):
            raise Stage1ConfigurationError(f'{path.name}: key {spec["key"]!r} holds {type(value).__name__}, not a field')
        return value

    @staticmethod
    def _no_keys(spec, path):
        if set(spec) != {'file'}:
            raise Stage1ConfigurationError(f'{path.name}: a {path.suffix} source is the whole file (no key or column)')


def _rows(value):
    return len(value) if isinstance(value, list) else int(np.shape(value)[0])


def _width(value):
    return 1 if isinstance(value, list) or np.ndim(value) == 1 else int(np.shape(value)[1])


class Stage1Plan:
    """A validated declaration: every source opened, every dimension and model checked, TRAIN nodes resolved."""

    def __init__(self, stage1, schema, sources, fields, train_rows, models):
        self.stage1, self.schema, self.sources = stage1, schema, sources
        self.fields, self.train_rows, self.models = fields, train_rows, models
        self.feature_schema = feature_schema(stage1)


def _seed_rows(seed, schema, sources, resource):
    if not isinstance(seed, dict) or seed.get('node_type') not in schema.node_counts:
        raise Stage1ConfigurationError(f'train_nodes.seed must name a node type of the graph {sorted(schema.node_counts)}: {seed!r}')
    node_type, source = seed['node_type'], seed.get('source')
    if isinstance(source, dict) and set(source) == {'resource'}:
        if set(seed) != {'node_type', 'source'} or node_type != schema.target_type:
            raise Stage1ConfigurationError('train_nodes.seed {resource: ...} lists target IDs of the target node type, without equals')
        name = source['resource']
        if resource is None:
            raise Stage1ConfigurationError(f'train_nodes.seed resource {name!r} is not declared under resources')
        rows, unresolved = schema.target_rows(resource(name).read_text().split())
        if unresolved:
            raise Stage1ConfigurationError(f'{len(unresolved)} TRAIN seed ID(s) of resource {name!r} name no {node_type} vertex')
        return node_type, set(rows.tolist())
    if set(seed) != {'node_type', 'source', 'equals'}:
        raise Stage1ConfigurationError('train_nodes.seed is {node_type, source: {file, key|column}, equals} or {node_type, source: {resource}}')
    values = sources.read(source)
    if value_kind(values) not in {'numeric_scalar', 'text'} or _rows(values) != schema.node_counts[node_type]:
        raise Stage1ConfigurationError(f'train_nodes.seed source must be one value per {node_type} row')
    rows = np.flatnonzero(np.asarray(values, dtype=object) == seed['equals']) if isinstance(values, list) else \
        np.flatnonzero(np.asarray(values) == seed['equals'])
    return node_type, set(rows.tolist())


def _train_rows(stage1, schema, sources, resource):
    spec = stage1['train_nodes']
    if not isinstance(spec, dict) or set(spec) != {'seed', 'propagate'}:
        raise Stage1ConfigurationError('stage1.train_nodes needs seed and propagate')
    node_type, seed = _seed_rows(spec['seed'], schema, sources, resource)
    if not seed:
        raise Stage1ConfigurationError('train_nodes.seed selects no vertex')
    train = {t: set() for t in schema.node_counts}
    train[node_type] = seed
    for triple in spec['propagate'] or []:
        source_type, _, destination_type = map(str, triple)
        try:
            src, dst = schema.relation(triple)
        except KeyError:
            reverse = (destination_type, str(triple[1]), source_type)
            if reverse not in schema.edges:
                raise Stage1ConfigurationError(f'train_nodes.propagate: relation {list(triple)} is not in the graph; '
                                               f'available: {[list(t) for t in schema.edges]}') from None
            dst, src = schema.relation(reverse)
        train[destination_type] |= set(dst[np.isin(src, list(train[source_type]))].tolist())
    return train


def train_overlap(train_rows, schema, evaluation):
    """Evaluation-population targets among the TRAIN vertices of the target type (canonical IDs). Stage 1 reads only
    feature values, never labels, witnesses or GNN outputs, so an overlap is recorded in the bank, not fatal; supervised
    training refuses any overlap (comce.training.trainer.training_targets)."""
    report = {}
    for name, ids in evaluation.items():
        rows, unresolved = schema.target_rows(ids)
        shared = sorted(set(rows.tolist()) & train_rows.get(schema.target_type, set()))
        report[name] = dict(ids=len(ids), unresolved=len(unresolved), shared_with_train=len(shared),
                            shared_ids=[schema.target_ids[r] for r in shared])
    return report


def validate(stage1, schema, sources, *, evaluation, resource=None):
    """Check a Stage 1 declaration against the data before anything is encoded or fitted. resource(name) is the path of
    datasets.<d>.resources.<name>, read by a {resource: name} TRAIN seed."""
    seed = (stage1.get('train_nodes') or {}).get('seed') if isinstance(stage1.get('train_nodes'), dict) else None
    if not declared(stage1) or not isinstance(seed, dict) or seed.get('node_type') is None:
        raise Stage1ConfigurationError('Stage 1 field encoding is not configured (node_types and train_nodes)', missing=True)
    if stage1.get('fit_split') != 'train':
        raise Stage1ConfigurationError('stage1.fit_split must be train: the bucketizer is fitted on TRAIN nodes only')
    dimension = stage1.get('representation_dimension')
    if type(dimension) is not int or dimension < 1:
        raise Stage1ConfigurationError('stage1.representation_dimension must be a positive integer')
    reducer = stage1.get('reducer')
    if reducer is not None and not (reducer == {'method': 'pca'} or (isinstance(reducer, dict) and reducer.get('method') == 'random_projection'
                                                                    and set(reducer) == {'method', 'seed'} and type(reducer['seed']) is int)):
        raise Stage1ConfigurationError('stage1.reducer is null, {method: pca} or {method: random_projection, seed: <int>}')
    node_types = stage1['node_types']
    undeclared = sorted(set(schema.node_counts) - set(node_types))
    unknown = sorted(set(node_types) - set(schema.node_counts))
    if unknown:
        raise Stage1ConfigurationError(f'stage1.node_types {unknown} are not node types of the graph {sorted(schema.node_counts)}')
    if undeclared:
        raise Stage1ConfigurationError(f'node type(s) {undeclared} have no Stage 1 encoding; every endpoint type needs '
                                       'coordinates', missing=True)
    incomplete = [f'{t}.{f.get("name")}' for t, spec in node_types.items() for f in (spec or {}).get('fields') or []
                  if isinstance(f, dict) and f.get('encoder') is None]
    if incomplete:
        raise Stage1ConfigurationError(f'no encoder chosen for field(s) {incomplete}', missing=True)
    fields, models = {}, {}
    for node_type, spec in node_types.items():
        if not isinstance(spec, dict) or set(spec) - {'fields', 'aggregation'} or not spec.get('fields'):
            raise Stage1ConfigurationError(f'stage1.node_types.{node_type} needs a non-empty fields list (and aggregation)')
        entries = []
        for field in spec['fields']:
            entries.append(_validate_field(node_type, field, schema, sources, models))
        aggregation = spec.get('aggregation')
        if len(entries) > 1 and aggregation not in {'mean', 'concat'}:
            raise Stage1ConfigurationError(f'{node_type}: several fields need aggregation mean or concat')
        if any(e['encoder'] == 'neighbor_mean' for e in entries) and len(entries) > 1:
            raise Stage1ConfigurationError(f'{node_type}: a neighbor_mean field is the only field of its node type')
        if aggregation == 'mean' and len({e['width'] for e in entries}) > 1:
            raise Stage1ConfigurationError(f'{node_type}: aggregation mean needs fields of one dimension, got {[e["width"] for e in entries]}')
        width = sum(e['width'] for e in entries) if aggregation == 'concat' else entries[0]['width']
        direct = entries[0]['encoder'] != 'neighbor_mean'
        if direct and reducer is None and width != dimension:
            raise Stage1ConfigurationError(f'{node_type}: representation has {width} dimensions but stage1.representation_dimension '
                                           f'is {dimension}; declare stage1.reducer or change representation_dimension')
        fields[node_type] = dict(entries=entries, aggregation=aggregation if len(entries) > 1 else None, width=width, direct=direct)
    for node_type, entry in fields.items():
        if not entry['direct']:
            other = entry['entries'][0]['neighbor']
            if not fields[other]['direct']:
                raise Stage1ConfigurationError(f'{node_type}: neighbor_mean reads {other}, which is itself a neighbor_mean type')
    train = _train_rows(stage1, schema, sources, resource)
    if reducer == {'method': 'pca'} and any(e['direct'] and len(train[t]) < 2 for t, e in fields.items()):
        raise Stage1ConfigurationError('reducer pca is fitted on TRAIN nodes; every encoded node type needs at least 2')
    for node_type, entry in fields.items():
        if any(e['encoder'] == 'lsh' and e['spec']['standardize'] == 'train' for e in entry['entries']) and len(train[node_type]) < 2:
            raise Stage1ConfigurationError(f'{node_type}: lsh standardize: train needs at least 2 TRAIN nodes')
    isolation = train_overlap(train, schema, evaluation)
    plan = Stage1Plan(stage1, schema, sources, fields, train, models)
    plan.isolation = isolation
    return plan


def _validate_field(node_type, field, schema, sources, models):
    if not isinstance(field, dict) or field.get('encoder') not in ENCODERS:
        raise Stage1ConfigurationError(f'{node_type}: field {field!r} needs encoder in {sorted(ENCODERS)}')
    encoder = field['encoder']
    required, optional = ENCODERS[encoder]
    missing, extra = required - set(field), set(field) - required - optional
    if missing or extra:
        raise Stage1ConfigurationError(f'{node_type}.{field.get("name")}: encoder {encoder} needs {sorted(required)}'
                                       + (f' (missing {sorted(missing)})' if missing else '')
                                       + (f' (unknown {sorted(extra)})' if extra else ''))
    name = f'{node_type}.{field["name"]}'
    if encoder == 'neighbor_mean':
        relation = field['relation']
        if not isinstance(relation, (list, tuple)) or len(relation) != 3 or tuple(map(str, relation)) not in schema.edges:
            raise Stage1ConfigurationError(f'{name}: relation {relation!r} is not in the graph; available: {[list(t) for t in schema.edges]}')
        if node_type not in (relation[0], relation[2]) or relation[0] == relation[2]:
            raise Stage1ConfigurationError(f'{name}: relation {list(relation)} must join {node_type} to another node type')
        if field['empty'] not in {'zero', 'error'}:
            raise Stage1ConfigurationError(f'{name}: empty is zero (vertices without neighbours get the zero vector) or error')
        neighbor = relation[0] if relation[2] == node_type else relation[2]
        return dict(encoder=encoder, spec=field, neighbor=str(neighbor), width=None, name=name)
    width = field['output_dimension']
    if type(width) is not int or width < 1:
        raise Stage1ConfigurationError(f'{name}: output_dimension must be a positive integer')
    values = sources.read(field['source'])
    kind = value_kind(values)
    if _rows(values) != schema.node_counts[node_type]:
        raise Stage1ConfigurationError(f'{name}: source has {_rows(values)} rows but node type {node_type} has '
                                       f'{schema.node_counts[node_type]}')
    if kind not in ENCODER_INPUT[encoder]:
        hint = ' (text fields use sbert)' if kind == 'text' else ' (numeric fields use lsh)' if encoder == 'sbert' else ''
        raise Stage1ConfigurationError(f'{name}: encoder {encoder} needs {sorted(ENCODER_INPUT[encoder])} input, the source is {kind}{hint}')
    if encoder == 'lsh':
        if field['input_kind'] not in ENCODER_INPUT['lsh'] or field['input_kind'] != kind:
            raise Stage1ConfigurationError(f'{name}: input_kind {field["input_kind"]!r} but the source is {kind}')
        if field['standardize'] not in {'train', 'none'} or type(field['seed']) is not int:
            raise Stage1ConfigurationError(f'{name}: lsh needs standardize train|none and an integer seed')
        if not np.isfinite(np.asarray(values, np.float64)).all():
            raise Stage1ConfigurationError(f'{name}: lsh input has non-finite values')
    if encoder == 'precomputed':
        if _width(values) != width:
            raise Stage1ConfigurationError(f'{name}: the precomputed embedding has {_width(values)} dimensions, output_dimension says {width}')
    if encoder == 'sbert':
        model = models.get(field['model']) or _load_sbert(name, field)
        models[field['model']] = model
        output = (getattr(model, 'get_embedding_dimension', None) or model.get_sentence_embedding_dimension)()
        if output != width:
            raise Stage1ConfigurationError(f'{name}: model {field["model"]!r} outputs {output} dimensions, output_dimension '
                                           f'says {width}')
    return dict(encoder=encoder, spec=field, width=width, name=name)


def _load_sbert(name, field):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise Stage1ConfigurationError(f'{name}: encoder sbert needs the sentence-transformers package') from exc
    try:
        return SentenceTransformer(str(field['model']), device=str(field.get('device', 'cpu')), local_files_only=True)
    except Exception as exc:
        raise Stage1ConfigurationError(f'{name}: Sentence Transformer {field["model"]!r} is not available locally '
                                       f'(models are never downloaded during Stage 1): {exc}') from exc


def scan_fields(sources, schema):
    """Arrays and columns under the data directory whose row count equals a node type's row count."""
    found = {t: [] for t in schema.node_counts}
    counts = schema.node_counts

    def add(source, value, hint=None):
        try:
            rows, kind = _rows(value), value_kind(value)
        except Exception:
            return
        types = [t for t, n in counts.items() if n == rows]
        if hint in types:
            types = [hint]
        shape = [rows] if isinstance(value, list) or np.ndim(value) == 1 else list(np.shape(value))
        dtype = 'str' if kind == 'text' else (type(value[0]).__name__ if isinstance(value, list) and value else str(getattr(value, 'dtype', '')))
        for t in types:
            found[t].append(dict(source=source, kind=kind, shape=shape, dtype=dtype))

    def walk(file, value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(key, str) and '.' not in key:
                    walk(file, item, path + [key])
        elif isinstance(value, (np.ndarray, torch.Tensor, list)) and path:
            array = value.numpy() if isinstance(value, torch.Tensor) else value
            add(dict(file=file, key='.'.join(path)), array, hint=path[-1] if path[-1] in counts else None)

    for path in sorted(p for p in sources.root.iterdir() if p.is_file() and p.suffix in SUFFIXES - {'.txt'}):
        try:
            if path.suffix == '.npy':
                add(dict(file=path.name), np.load(path, mmap_mode='r'))
            elif path.suffix == '.parquet':
                import pyarrow.parquet as pq
                frame = pd.read_parquet(path)
                for column in pq.read_schema(path).names:
                    values = frame[column]
                    add(dict(file=path.name, column=column), values.astype(object).tolist()
                        if values.dtype == object or pd.api.types.is_string_dtype(values) else values.to_numpy())
            else:
                _, container = sources._container(path.name)
                walk(path.name, dict(container.items()) if path.suffix == '.npz' else container, [])
        except Exception as exc:
            found.setdefault('_unreadable', []).append(dict(file=path.name, reason=str(exc).splitlines()[0][:160]))
    return found


def write_template(path, *, dataset, schema, found, stage1):
    """Editable YAML listing node types and candidate fields; every encoder is left null for the user to choose."""
    import yaml

    def flow(value):
        return yaml.safe_dump(value, default_flow_style=True, sort_keys=False, width=1000).strip()
    relations = [list(t) for t in schema.edges]
    lines = [f"# Stage 1 field encoding of dataset '{dataset}'. Set every null, delete the fields not to encode, then copy",
             f'# datasets.{dataset}.stage1 into the configuration and run scripts/fit_signatures.py again.',
             '# encoder: sbert (model, output_dimension) | lsh (input_kind, standardize: train | none, output_dimension, seed)',
             '#          | precomputed (output_dimension) | neighbor_mean (relation: [src_type, relation, dst_type], empty: zero | error)',
             '# aggregation: mean | concat (several fields). train_nodes: TRAIN vertices the bucketizer is fitted on.',
             f'# Relations: {flow(relations)}',
             'datasets:', f'  {dataset}:', '    stage1:',
             f'      representation_dimension: {stage1.get("representation_dimension")}',
             '      node_types:']
    for node_type, count in schema.node_counts.items():
        lines += [f'        {node_type}:            # {count} rows' + (' (target node type)' if node_type == schema.target_type else ''),
                  '          aggregation: null', '          fields:']
        candidates = found.get(node_type, [])
        if not candidates:
            lines.append(f'            # no array or column with {count} rows was found; use neighbor_mean over a relation')
            lines.append('            - {name: null, encoder: null}')
        for item in candidates:
            name = (item['source'].get('column') or item['source'].get('key') or Path(item['source']['file']).stem).split('.')[-1]
            lines.append(f'            - name: {name}          # {item["kind"]} {item["shape"]} {item["dtype"]}')
            lines.append(f'              source: {flow(item["source"])}')
            lines.append('              encoder: null')
    lines += ['      train_nodes:',
              '        seed: {node_type: null, source: null, equals: null}',
              '        propagate: []']
    for item in found.get('_unreadable', []):
        lines.append(f'# not inspected: {item["file"]}: {item["reason"]}')
    text = '\n'.join(lines) + '\n'
    yaml.safe_load(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(text)
    return path
