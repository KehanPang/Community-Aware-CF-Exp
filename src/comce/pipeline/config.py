"""yaml.template resolution: command line > tuples.<dataset>.<backbone> > datasets.<dataset> > dataset_defaults > paper_defaults."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from comce.stage1 import signatures
from comce.stage2.domain import DomainParams
from comce.stage3.settings import SearchSettings

PARAMETERS = {
    'stage1': {'projection_dimensions', 'quantile_buckets', 'representation_dimension', 'fit_split', 'projection_seed',
               'reducer', 'node_types', 'train_nodes'},
    'stage2': {'relevance_threshold', 'query_slots', 'attention_heads', 'hidden_width', 'domain_edge_cap', 'admission_exponent',
               'fallback_min_domain_edges'},
    'stage3': {'explanation_budget', 'rollouts', 'failure_threshold', 'failures_before_expansion', 'expansion_budget',
               'certification_queries', 'lookahead', 'cleanup_queries', 'proposals_per_step', 'guidance_scale',
               'guidance_batch_size', 'expansion_communities_per_trigger', 'temperature', 'edit_cost', 'dedup_threshold'},
    'task': {'prediction', 'min_receptive_edges_exclusive', 'allowed_target_ids'},
    'execution': {'device', 'random_seed', 'threads', 'deterministic', 'allow_tf32'},
}
DATASET_KEYS = {'resources', 'adapter', 'relations', 'decision', 'training'} | set(PARAMETERS)
TUPLE_KEYS = {'model', 'gnn'} | (set(PARAMETERS) - {'stage1'})
SEARCH_KEYS = {'explanation_budget', 'rollouts', 'failure_threshold', 'failures_before_expansion', 'expansion_budget',
               'certification_queries', 'lookahead', 'cleanup_queries', 'proposals_per_step', 'guidance_scale',
               'guidance_batch_size', 'expansion_communities_per_trigger', 'temperature', 'dedup_threshold'}


class ResourceConfigurationError(Exception):
    """A task needs an external resource whose path the configuration leaves empty."""

    def __init__(self, keys, notes=()):
        self.keys, self.notes = sorted(set(keys)), list(dict.fromkeys(notes))
        super().__init__('\n'.join(['Required resource path is empty:', *self.keys, *self.notes,
                                    'Set this field in your YAML configuration.']))


def _value(document, key):
    for part in key.split('.'):
        if not isinstance(document, dict) or part not in document:
            return None
        document = document[part]
    return document


class ResourcePaths:
    """External resource paths of datasets.<dataset> and tuples.<dataset>.<backbone>. A path is absolute or relative to
    resources.root. An empty path has no default: resolving it raises ResourceConfigurationError naming its key."""

    def __init__(self, template, dataset, backbone=None):
        self.template, self.dataset, self.backbone = template, dataset, backbone
        self.dataset_config = template['datasets'][dataset]

    @property
    def root(self):
        return self.resolve('resources.root') if _value(self.template, 'resources.root') else None

    def resolve(self, key):
        """resources.root and outputs.root are absolute or relative to the configuration file."""
        value = _value(self.template, key)
        if not isinstance(value, str) or not value.strip():
            raise ResourceConfigurationError([key])
        path = Path(value)
        if path.is_absolute():
            return path
        if key in ('resources.root', 'outputs.root'):
            return Path(self.template['_path']).parent / path
        if self.root is None:
            raise ResourceConfigurationError(['resources.root'], [f'{key} is the relative path {value!r}'])
        return self.root / path

    def require(self, keys):
        """Resolve every key before anything is loaded; one error lists all empty keys."""
        missing, notes = [], []
        for key in keys:
            try:
                self.resolve(key)
            except ResourceConfigurationError as error:
                missing, notes = missing + error.keys, notes + error.notes
        if missing:
            raise ResourceConfigurationError(missing, notes)

    def key(self, name):
        return f'datasets.{self.dataset}.{name}'

    def tuple_key(self, name):
        return f'tuples.{self.dataset}.{self.backbone}.{name}'

    def resource(self, name):
        return self.resolve(self.key(f'resources.{name}'))

    def population(self, name):
        populations = self.dataset_config['resources'].get('populations') or {}
        if name not in populations:
            raise ValueError(f'Unknown population {name!r}; configured: {sorted(populations)}')
        return self.resolve(self.key(f'resources.populations.{name}'))

    def evaluation_keys(self):
        names = (self.dataset_config.get('training') or {}).get('evaluation_populations') or []
        return [self.key(f'resources.populations.{name}') for name in names]

    def inference_keys(self):
        return [self.tuple_key('model'), self.tuple_key('gnn.checkpoint'), self.key('resources.data'),
                self.key('resources.signature_bank')]

    def population_keys(self):
        """The TRAIN population and every population it must not intersect."""
        return [self.key('training.population_path'), *self.evaluation_keys()]

    def training_keys(self):
        """Files a bundle is trained from, the signature bank aside."""
        spec = self.dataset_config.get('training') or {}
        keys = [self.tuple_key('gnn.checkpoint'), self.key('resources.data'), *self.population_keys()]
        if spec.get('targets') is None and spec.get('labeled_population') is not None:
            keys.append(self.key(f'resources.populations.{spec["labeled_population"]}'))
        return keys


def load_template(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or not {'paper_defaults', 'dataset_defaults', 'resources', 'datasets', 'tuples'} <= set(value):
        raise ValueError(f'{path} is not a ComCE configuration')
    value['_path'] = str(path)
    return value


def _merge(base, changes, section):
    unknown = set(changes) - PARAMETERS[section]
    if unknown:
        raise ValueError(f'Unknown {section} key(s): {sorted(unknown)}')
    for key, value in changes.items():
        base[key] = deepcopy(value)
    return base


def parse_override(text):
    if '=' not in text:
        raise ValueError(f'Override must be section.key=value: {text}')
    path, raw = text.split('=', 1)
    parts = path.split('.')
    if len(parts) != 2 or parts[0] not in PARAMETERS or parts[0] == 'stage1':
        raise ValueError(f'Override must name stage2/stage3/task/execution.<key>: {path}')
    return parts[0], parts[1], yaml.safe_load(raw)


@dataclass(frozen=True)
class ResolvedConfig:
    dataset: str
    backbone: str
    parameters: dict[str, Any]
    dataset_config: dict[str, Any]
    tuple_config: dict[str, Any]
    paths: ResourcePaths
    verify_checksums: bool

    @property
    def domain(self):
        s = self.parameters['stage2']
        return DomainParams(float(s['relevance_threshold']), None if s['domain_edge_cap'] is None else int(s['domain_edge_cap']),
                            float(s['admission_exponent']), s['fallback_min_domain_edges'])

    @property
    def search(self):
        s = self.parameters['stage3']
        return SearchSettings(**{k: s[k] for k in SEARCH_KEYS})

    @property
    def task(self):
        return self.parameters['task']

    @property
    def execution(self):
        return self.parameters['execution']

    @property
    def stage1(self):
        return self.parameters['stage1']

    @property
    def resources(self):
        return dict(self.dataset_config['resources'], data=str(self.paths.resource('data')))

    @property
    def model_path(self):
        return self.paths.resolve(self.paths.tuple_key('model'))

    @property
    def gnn_path(self):
        return self.paths.resolve(self.paths.tuple_key('gnn.checkpoint'))

    @property
    def signature_bank_path(self):
        return self.paths.resource('signature_bank')

    def payload(self):
        root = self.paths.root
        return dict(dataset=self.dataset, backbone=self.backbone, parameters=self.parameters,
                    model=self.tuple_config['model'], gnn=self.tuple_config['gnn'],
                    resources_root=None if root is None else str(root),
                    adapter=self.dataset_config['adapter']['name'], relations=self.dataset_config['relations'],
                    decision=self.dataset_config['decision'])


def resolve_stage1(template, dataset):
    """Stage 1 parameters of one dataset, which need no model bundle: datasets.<d> > dataset_defaults > paper_defaults."""
    if dataset not in template['datasets']:
        raise ValueError(f'Unknown dataset {dataset!r}; configured: {sorted(template["datasets"])}')
    unknown = set(template['datasets'][dataset]) - DATASET_KEYS
    if unknown:
        raise ValueError(f'Unknown key(s) in datasets.{dataset}: {sorted(unknown)}')
    parameters = {}
    for layer in (template['paper_defaults'], template['dataset_defaults'], template['datasets'][dataset]):
        _merge(parameters, layer.get('stage1', {}), 'stage1')
    missing = sorted(PARAMETERS['stage1'] - set(parameters))
    if missing:
        raise ValueError(f'Unresolved configuration keys: {{"stage1": {missing}}}')
    _validate_stage1(parameters)
    return parameters, ResourcePaths(template, dataset)


def resolve(template, dataset, backbone, overrides=()):
    if dataset not in template['datasets']:
        raise ValueError(f'Unknown dataset {dataset!r}; configured: {sorted(template["datasets"])}')
    if backbone not in template['tuples'].get(dataset, {}):
        raise ValueError(f'No model configured for {dataset}/{backbone}')
    dataset_config = template['datasets'][dataset]
    tuple_config = template['tuples'][dataset][backbone]
    for name, value, allowed in ((f'datasets.{dataset}', dataset_config, DATASET_KEYS),
                                 (f'tuples.{dataset}.{backbone}', tuple_config, TUPLE_KEYS)):
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f'Unknown key(s) in {name}: {sorted(unknown)}')
    parameters = {section: {} for section in PARAMETERS}
    for section in ('stage1', 'stage2', 'stage3'):
        _merge(parameters[section], template['paper_defaults'].get(section, {}), section)
    for layer in (template['dataset_defaults'], dataset_config, tuple_config):
        for section in PARAMETERS:
            if section in layer:
                _merge(parameters[section], layer[section], section)
    for section, key, value in (parse_override(o) if isinstance(o, str) else o for o in overrides):
        _merge(parameters[section], {key: value}, section)
    missing = {section: sorted(PARAMETERS[section] - set(values)) for section, values in parameters.items()
               if PARAMETERS[section] - set(values)}
    if missing:
        raise ValueError(f'Unresolved configuration keys: {missing}')
    _validate(parameters)
    return ResolvedConfig(dataset, backbone, parameters, dataset_config, tuple_config, ResourcePaths(template, dataset, backbone),
                          bool(template['resources'].get('verify_checksums', True)))


def _validate_stage1(stage1):
    if stage1['projection_dimensions'] != signatures.DIMENSIONS or stage1['quantile_buckets'] != signatures.BUCKETS:
        raise ValueError('Signature projection dimensions and buckets are fixed by the signature banks')
    if type(stage1['representation_dimension']) is not int or stage1['representation_dimension'] < 1:
        raise ValueError('stage1.representation_dimension must be a positive integer')
    if stage1['fit_split'] != 'train' or type(stage1['projection_seed']) is not int:
        raise ValueError('stage1.fit_split must be train and stage1.projection_seed an integer')


def _validate(parameters):
    _validate_stage1(parameters['stage1'])
    parameters_search = SearchSettings(**{k: parameters['stage3'][k] for k in SEARCH_KEYS})
    if parameters_search.explanation_budget > 10:
        raise ValueError('explanation_budget must not exceed 10')
    stage2 = parameters['stage2']
    DomainParams(float(stage2['relevance_threshold']), stage2['domain_edge_cap'], float(stage2['admission_exponent']),
                 stage2['fallback_min_domain_edges'])
    task = parameters['task']
    if task['prediction'] not in {'checkpoint', 'positive_only'}:
        raise ValueError('task.prediction must be checkpoint or positive_only')
    if task['min_receptive_edges_exclusive'] is not None and (type(task['min_receptive_edges_exclusive']) is not int
                                                             or task['min_receptive_edges_exclusive'] < 0):
        raise ValueError('task.min_receptive_edges_exclusive must be a nonnegative integer or null')
    if task['allowed_target_ids'] is not None:
        task['allowed_target_ids'] = list(map(str, task['allowed_target_ids']))
    cost = parameters['stage3']['edit_cost']
    if not isinstance(cost, dict) or cost.get('mode') not in {'uniform', 'mapping'}:
        raise ValueError('stage3.edit_cost.mode must be uniform or mapping')
    execution = parameters['execution']
    if type(execution['threads']) is not int or execution['threads'] < 1:
        raise ValueError('execution.threads must be a positive integer')


def eligibility_reason(task, *, center_id, receptive_edges, original_prediction, positive_class, checkpoint_eligible):
    allowed = task['allowed_target_ids']
    if allowed is not None and str(center_id) not in allowed:
        return 'TARGET_NOT_IN_CONFIGURED_COHORT'
    minimum = task['min_receptive_edges_exclusive']
    if minimum is not None and receptive_edges <= minimum:
        return f'RECEPTIVE_EDGES_MUST_EXCEED_{minimum}'
    if task['prediction'] == 'positive_only' and original_prediction != positive_class:
        return 'ORIGINAL_PREDICTION_NOT_POSITIVE'
    if not checkpoint_eligible:
        return 'FROZEN_CHECKPOINT_DECISION_INELIGIBLE'
    return None
