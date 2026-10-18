"""Signature bank fitting from a validated Stage 1 declaration (comce.stage1.fields)."""
from pathlib import Path

import numpy as np

from comce.pipeline.io import sha256
from comce.stage1 import fields as declarations
from comce.stage1.signatures import SignatureBank, project_rows, quantile_thresholds, random_directions


def lsh_representation(values, spec, train_rows):
    """TRAIN-standardized (or raw) numeric vector times a seeded Gaussian matrix W ~ N(0, 1/n_in)."""
    values = np.asarray(values, np.float64)
    values = values[:, None] if values.ndim == 1 else values
    if spec['standardize'] == 'train':
        rows = np.asarray(sorted(train_rows), np.int64)
        mean, std = values[rows].mean(0), np.maximum(values[rows].std(0), 1e-6)
        values = (values - mean) / std
    weight = np.random.default_rng(int(spec['seed'])).normal(0.0, 1.0, size=(values.shape[1], int(spec['output_dimension'])))
    return values @ (weight / np.sqrt(values.shape[1]))


def sbert_representation(values, spec, model):
    return np.asarray(model.encode(list(values), batch_size=int(spec.get('batch_size', 64)), convert_to_numpy=True,
                                   normalize_embeddings=bool(spec.get('normalize', False)), show_progress_bar=False), np.float32)


def reduce(representation, reducer, dimension, train_rows):
    if reducer['method'] == 'random_projection':
        weight = np.random.default_rng(int(reducer['seed'])).normal(0.0, 1.0, size=(representation.shape[1], dimension))
        return project_rows(representation, weight / np.sqrt(dimension))
    rows = np.asarray(sorted(train_rows), np.int64)
    if representation.shape[1] < dimension:
        raise declarations.Stage1ConfigurationError(f'pca cannot raise {representation.shape[1]} dimensions to {dimension}')
    train = np.asarray(representation[rows], np.float64)
    mean = train.mean(0)
    _, vectors = np.linalg.eigh(np.cov(train - mean, rowvar=False))
    components = vectors[:, ::-1][:, :dimension]
    components *= np.sign(components[np.abs(components).argmax(0), np.arange(dimension)])
    return project_rows(representation, components) - mean @ components


def fit_signature_bank(plan):
    """x_v per node type -> z_v = r^T x_v on d_s shared directions -> one TRAIN-quantile bucketizer for every type."""
    stage1, schema = plan.stage1, plan.schema
    dimension = int(stage1['representation_dimension'])
    directions = random_directions(dimension, int(stage1['projection_seed']), int(stage1['projection_dimensions']))
    coordinates, described, inputs = {}, {}, set()
    for node_type, entry in plan.fields.items():
        if not entry['direct']:
            continue
        parts = []
        for field in entry['entries']:
            spec = field['spec']
            values = plan.sources.read(spec['source'])
            inputs.add(spec['source']['file'])
            if field['encoder'] == 'precomputed':
                parts.append(values)
            elif field['encoder'] == 'lsh':
                parts.append(lsh_representation(values, spec, plan.train_rows[node_type]))
            else:
                parts.append(sbert_representation(values, spec, plan.models[spec['model']]))
        if len(parts) == 1:
            representation = parts[0]
        elif entry['aggregation'] == 'mean':
            representation = sum(np.asarray(p, np.float64) for p in parts) / len(parts)
        else:
            representation = np.concatenate([np.asarray(p, np.float64) for p in parts], axis=1)
        if stage1['reducer'] is not None:
            representation = reduce(representation, stage1['reducer'], dimension, plan.train_rows[node_type])
        if representation.shape[1] != dimension:
            raise declarations.Stage1ConfigurationError(f'{node_type}: representation has {representation.shape[1]} dimensions, not {dimension}')
        coordinates[node_type] = project_rows(representation, directions)
    for node_type, entry in plan.fields.items():
        if entry['direct']:
            continue
        spec = entry['entries'][0]['spec']
        relation = tuple(map(str, spec['relation']))
        src, dst = plan.schema.relation(relation)
        into, start = (dst, src) if relation[2] == node_type else (src, dst)
        count = np.bincount(into, minlength=schema.node_counts[node_type]).astype(np.float64)
        if spec['empty'] == 'error' and (count == 0).any():
            raise declarations.Stage1ConfigurationError(f'{node_type}: {int((count == 0).sum())} vertices have no neighbour along {list(relation)}')
        # The projection is linear, so the mean is taken after it: mean_j r^T x_j = r^T mean_j x_j.
        values = np.zeros((schema.node_counts[node_type], directions.shape[1]), np.float64)
        np.add.at(values, into, coordinates[entry['entries'][0]['neighbor']][start])
        values /= np.maximum(count, 1.)[:, None]
        coordinates[node_type] = values
        described[node_type] = dict(vertices_without_neighbours=int((count == 0).sum()))
    coordinates = {t: np.asarray(coordinates[t], np.float32) for t in plan.fields}
    pooled = np.concatenate([coordinates[t][np.asarray(sorted(rows), np.int64)] for t, rows in plan.train_rows.items() if len(rows)])
    thresholds = quantile_thresholds(pooled.astype(np.float64), int(stage1['quantile_buckets']))
    root = plan.sources.root
    field_provenance = {}
    for node_type, entry in plan.fields.items():
        field_provenance[node_type] = []
        for field in entry['entries']:
            spec = dict(field['spec'])
            if 'source' in spec:
                spec['source_sha256'] = sha256(root / spec['source']['file'])
            if field['encoder'] == 'sbert':
                spec['model_sha256'] = declarations.model_digest(spec['model'])
            field_provenance[node_type].append(spec)
    provenance = dict(feature_schema=dict(fingerprint=plan.fingerprint, stage1=plan.feature_schema), fields=field_provenance,
                      neighbor_aggregation=described, projection='fixed_seeded_gaussian_unit_columns',
                      projection_seed=int(stage1['projection_seed']), representation_dimension=dimension,
                      dimensions=int(stage1['projection_dimensions']), buckets=int(stage1['quantile_buckets']),
                      threshold_fit='quantiles over TRAIN-split nodes only',
                      train_nodes={t: len(rows) for t, rows in plan.train_rows.items()}, train_isolation=plan.isolation,
                      inputs={name: sha256(Path(root) / name) for name in sorted(inputs)},
                      model_hidden_states_used=False, backbone_independent=True)
    return SignatureBank(coordinates, thresholds, provenance)
