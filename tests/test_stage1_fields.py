import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from comce.datasets.schema import GraphSchema
from comce.pipeline.config import load_template, resolve_stage1
from comce.pipeline.features import stage1_plan
from comce.stage1.encoders import fit_signature_bank, lsh_representation
from comce.stage1.fields import FieldSource, Stage1ConfigurationError, validate
from comce.stage1.signatures import project_rows, quantile_thresholds, random_directions

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = load_template(ROOT / 'yaml.template')


def toy(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    np.save(data / 'item_vectors.npy', rng.normal(size=(6, 5)))
    np.save(data / 'tag_embeddings.npy', rng.normal(size=(4, 8)).astype(np.float32))
    pd.DataFrame({'title': [f'a short title number {i}' for i in range(6)], 'price': rng.normal(size=6),
                  'split': ['train'] * 4 + ['val', 'test']}).to_parquet(data / 'items.parquet')
    schema = GraphSchema(node_counts={'item': 6, 'tag': 4},
                         edges={('item', 'has_tag', 'tag'): (np.array([0, 1, 2, 3, 4, 5, 0]), np.array([0, 1, 2, 3, 0, 1, 3]))},
                         target_type='item', target_ids=tuple(f'i{i}' for i in range(6)))
    return schema, FieldSource(data, set())


LSH = dict(name='vectors', source=dict(file='item_vectors.npy'), encoder='lsh', input_kind='numeric_vector',
           standardize='train', output_dimension=8, seed=3)
TAGS = dict(name='embedding', source=dict(file='tag_embeddings.npy'), encoder='precomputed', output_dimension=8)


def declaration(node_types, dimension=8, reducer=None):
    return dict(projection_dimensions=8, quantile_buckets=16, representation_dimension=dimension, fit_split='train',
                projection_seed=7, reducer=reducer, node_types=node_types,
                train_nodes=dict(seed=dict(node_type='item', source=dict(file='items.parquet', column='split'), equals='train'),
                                 propagate=[['item', 'has_tag', 'tag']]))


def check(stage1, schema, sources, evaluation=None):
    return validate(stage1, schema, sources, evaluation=evaluation or {})


def test_missing_or_incomplete_declarations_stop_before_encoding(tmp_path):
    schema, sources = toy(tmp_path)
    for stage1 in (declaration(None), declaration({'item': dict(fields=[LSH])}),
                   declaration({'item': dict(fields=[dict(LSH, encoder=None)]), 'tag': dict(fields=[TAGS])})):
        with pytest.raises(Stage1ConfigurationError) as error:
            check(stage1, schema, sources)
        assert error.value.missing
    unseeded = declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS])})
    unseeded['train_nodes']['seed'] = dict(node_type=None, source=None, equals=None)
    with pytest.raises(Stage1ConfigurationError) as error:
        check(unseeded, schema, sources)
    assert error.value.missing
    with pytest.raises(Stage1ConfigurationError, match='not node types of the graph') as error:
        check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS]), 'user': dict(fields=[TAGS])}), schema, sources)
    assert not error.value.missing


def test_lsh_projects_train_standardized_numeric_fields_and_buckets_use_train_nodes_only(tmp_path):
    schema, sources = toy(tmp_path)
    plan = check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS])}), schema, sources)
    assert plan.train_rows == {'item': {0, 1, 2, 3}, 'tag': {0, 1, 2, 3}}
    bank = fit_signature_bank(plan)
    values = np.load(sources.root / 'item_vectors.npy')
    standardized = (values - values[:4].mean(0)) / values[:4].std(0)
    weight = np.random.default_rng(3).normal(size=(5, 8)) / np.sqrt(5)
    directions = random_directions(8, 7)
    np.testing.assert_allclose(bank.coordinates['item'], project_rows(standardized @ weight, directions), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lsh_representation(values, LSH, {0, 1, 2, 3}), standardized @ weight)
    pooled = np.concatenate([bank.coordinates['item'][:4], bank.coordinates['tag']]).astype(np.float64)
    np.testing.assert_array_equal(bank.thresholds, quantile_thresholds(pooled, 16))
    assert bank.provenance['feature_schema'] == plan.feature_schema and bank.provenance['fields']['tag'] == [TAGS]
    shifted = values.copy()
    shifted[4:] += 100.0
    np.save(sources.root / 'item_vectors.npy', shifted)
    again = fit_signature_bank(check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS])}), schema,
                                     FieldSource(sources.root, set())))
    np.testing.assert_array_equal(again.thresholds, bank.thresholds)


@pytest.mark.parametrize('field,message', [
    (dict(LSH, source=dict(file='items.parquet', column='title'), input_kind='numeric_scalar'), 'text fields use sbert'),
    (dict(LSH, source=dict(file='items.parquet', column='price')), "input_kind 'numeric_vector' but the source is numeric_scalar"),
    (dict(LSH, standardize='zscore'), 'standardize train|none'),
    (dict(LSH, source=dict(file='missing.npy')), 'does not exist'),
    (dict(LSH, source=dict(file='items.parquet', column='colour')), r"column\(s\) \['colour'\] do not exist"),
    ({k: v for k, v in LSH.items() if k != 'seed'}, r"missing \['seed'\]"),
    (dict(LSH, source=dict(file='tag_embeddings.npy')), 'source has 4 rows but node type item has 6'),
])
def test_illegal_field_declarations_fail_explicitly(tmp_path, field, message):
    schema, sources = toy(tmp_path)
    with pytest.raises(Stage1ConfigurationError, match=message):
        check(declaration({'item': dict(fields=[field]), 'tag': dict(fields=[TAGS])}), schema, sources)


def test_precomputed_embeddings_need_a_matching_width(tmp_path):
    schema, sources = toy(tmp_path)
    item = dict(TAGS, source=dict(file='item_vectors.npy'), output_dimension=5)
    for tag, message in ((dict(TAGS, output_dimension=16), 'has 8 dimensions, output_dimension says 16'),
                         ({k: v for k, v in TAGS.items() if k != 'output_dimension'}, r"missing \['output_dimension'\]"),
                         (dict(TAGS, model='bge'), r"unknown \['model'\]")):
        with pytest.raises(Stage1ConfigurationError, match=message):
            check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[tag])}), schema, sources)
    with pytest.raises(Stage1ConfigurationError, match='declare stage1.reducer'):
        check(declaration({'item': dict(fields=[item]), 'tag': dict(fields=[TAGS])}), schema, sources)


def test_reducers_and_multi_field_aggregation(tmp_path):
    schema, sources = toy(tmp_path)
    price = dict(LSH, name='price', source=dict(file='items.parquet', column='price'), input_kind='numeric_scalar', seed=4)
    with pytest.raises(Stage1ConfigurationError, match='several fields need aggregation'):
        check(declaration({'item': dict(fields=[LSH, price]), 'tag': dict(fields=[TAGS])}), schema, sources)
    mean = check(declaration({'item': dict(fields=[LSH, price], aggregation='mean'), 'tag': dict(fields=[TAGS])}), schema, sources)
    assert mean.fields['item']['width'] == 8
    with pytest.raises(Stage1ConfigurationError, match='representation has 16 dimensions'):
        check(declaration({'item': dict(fields=[LSH, price], aggregation='concat'), 'tag': dict(fields=[TAGS])}), schema, sources)
    for reducer in ({'method': 'pca'}, {'method': 'random_projection', 'seed': 11}):
        plan = check(declaration({'item': dict(fields=[LSH, price], aggregation='concat'), 'tag': dict(fields=[TAGS])},
                                 reducer=reducer), schema, sources)
        bank = fit_signature_bank(plan)
        assert bank.coordinates['item'].shape == (6, 8) and np.isfinite(bank.coordinates['item']).all()
    with pytest.raises(Stage1ConfigurationError, match='reducer is null'):
        check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS])}, reducer={'method': 'truncate'}), schema, sources)


def test_neighbor_mean_is_the_mean_of_projected_neighbour_coordinates(tmp_path):
    schema, sources = toy(tmp_path)
    neighbors = dict(name='items', encoder='neighbor_mean', relation=['item', 'has_tag', 'tag'], empty='zero')
    bank = fit_signature_bank(check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[neighbors])}), schema, sources))
    items = bank.coordinates['item'].astype(np.float64)
    np.testing.assert_allclose(bank.coordinates['tag'][3], (items[3] + items[0]) / 2, rtol=1e-6)
    np.testing.assert_allclose(bank.coordinates['tag'][0], (items[0] + items[4]) / 2, rtol=1e-6)
    sparse = GraphSchema(schema.node_counts, {('item', 'has_tag', 'tag'): (np.array([0, 1]), np.array([0, 1]))}, 'item', schema.target_ids)
    with pytest.raises(Stage1ConfigurationError, match='have no neighbour'):
        fit_signature_bank(check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[dict(neighbors, empty='error')])}),
                                 sparse, sources))


def test_sbert_encodes_text_fields_with_a_local_model_and_checks_its_dimension(tmp_path, sbert_model):
    schema, sources = toy(tmp_path)
    text = dict(name='title', source=dict(file='items.parquet', column='title'), encoder='sbert', model=str(sbert_model),
                output_dimension=384, normalize=True)
    plan = check(declaration({'item': dict(fields=[text]), 'tag': dict(fields=[TAGS])}, reducer={'method': 'random_projection', 'seed': 2}),
                 schema, sources)
    bank = fit_signature_bank(plan)
    encoded = plan.models[str(sbert_model)].encode(pd.read_parquet(sources.root / 'items.parquet')['title'].tolist(),
                                                   normalize_embeddings=True)
    weight = np.random.default_rng(2).normal(size=(384, 8)) / np.sqrt(8)
    np.testing.assert_allclose(bank.coordinates['item'], project_rows(project_rows(encoded, weight), random_directions(8, 7)),
                               rtol=1e-4, atol=1e-4)
    for field, message in ((dict(text, output_dimension=768), 'outputs 384 dimensions, output_dimension says 768'),
                           (dict(text, model=str(tmp_path / 'no_such_model')), 'not available locally'),
                           (dict(text, source=dict(file='items.parquet', column='price')), 'numeric fields use lsh')):
        with pytest.raises(Stage1ConfigurationError, match=message):
            check(declaration({'item': dict(fields=[field]), 'tag': dict(fields=[TAGS])}, reducer={'method': 'random_projection', 'seed': 2}),
                  schema, sources)


def test_train_seed_overlap_with_evaluation_targets_is_reported(tmp_path):
    schema, sources = toy(tmp_path)
    plan = check(declaration({'item': dict(fields=[LSH]), 'tag': dict(fields=[TAGS])}), schema, sources,
                 evaluation=dict(dev=['i1', 'i4', 'unknown']))
    assert plan.isolation == dict(dev=dict(ids=3, unresolved=1, shared_with_train=1, shared_ids=['i1']))


def undeclared_config(resource_template, tmp_path, dataset):
    raw = yaml.safe_load(Path(resource_template['_path']).read_text())
    root = resolve_stage1(resource_template, dataset)[1].root
    if root is not None:
        raw['resources']['root'] = str(root)
    raw['outputs'] = dict(root='')
    raw['datasets'][dataset].pop('stage1')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config


def test_fit_signatures_without_a_declaration_fails_and_writes_a_template_once(tmp_path, resource_template):
    config = undeclared_config(resource_template, tmp_path, 'goodreads')
    template, bank = tmp_path / 'out' / 'goodreads.stage1.yaml', tmp_path / 'bank' / 'signature_bank.pt'
    command = [sys.executable, str(ROOT / 'scripts/fit_signatures.py'), '--config', str(config), '--dataset', 'goodreads',
               '--output', str(bank), '--template-output', str(template)]
    first = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert first.returncode != 0 and 'Stage1ConfigurationError' in first.stderr and str(template) in first.stderr
    assert 'Nothing was encoded or fitted' in first.stderr and not bank.exists()
    written = template.read_text()
    generated = yaml.safe_load(written)['datasets']['goodreads']['stage1']
    assert set(generated['node_types']) == {'book', 'review', 'user', 'author'}
    sources = {t: [f['source'] for f in v['fields']] for t, v in generated['node_types'].items()}
    assert {'file': 'book_features.npy'} in sources['book']
    assert all(f['encoder'] is None for v in generated['node_types'].values() for f in v['fields'])
    assert generated['train_nodes']['seed']['node_type'] is None
    second = subprocess.run(command, capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert second.returncode != 0 and 'left unchanged' in second.stderr and template.read_text() == written
    stage1, paths = resolve_stage1(load_template(config), 'goodreads')
    with pytest.raises(Stage1ConfigurationError) as error:
        stage1_plan(load_template(config), 'goodreads', dict(stage1, **generated), paths, template_output=template)
    assert error.value.missing and template.read_text() == written


def test_without_template_output_or_outputs_root_the_stage1_template_goes_to_a_new_temporary_directory(tmp_path, resource_template):
    config = undeclared_config(resource_template, tmp_path, 'goodreads')
    temporary = tmp_path / 'tmp'
    temporary.mkdir()
    command = [sys.executable, str(ROOT / 'scripts/fit_signatures.py'), '--config', str(config), '--dataset', 'goodreads']
    result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(temporary)))
    assert result.returncode != 0 and 'Stage1ConfigurationError' in result.stderr
    written = list(temporary.glob('comce-stage1-*/goodreads.stage1.yaml'))
    assert len(written) == 1 and str(written[0]) in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ['config.yaml', 'tmp']


def test_declared_dimensions_keep_the_768_contract_and_the_paper_record():
    assert TEMPLATE['paper_defaults']['stage1']['representation_dimension'] == 64
    for dataset, dimension in (('claim', 64), ('computer', 768), ('goodreads', 768)):
        stage1, _ = resolve_stage1(TEMPLATE, dataset)
        assert stage1['representation_dimension'] == dimension and stage1['reducer'] is None
        encoders = {f['encoder'] for v in stage1['node_types'].values() for f in v['fields']}
        assert encoders == ({'lsh'} if dataset == 'claim' else {'precomputed'} | ({'neighbor_mean'} if dataset == 'computer' else set()))
