"""One ComCE request: q0, Region record, Set Transformer, domain admission, search, certification and cleanup."""
import contextlib
import gc
import time

import torch

from comce.pipeline.config import eligibility_reason
from comce.pipeline.io import sha256
from comce.pipeline.models import load_models
from comce.pipeline.resources import load_explained_model
from comce.stage2.domain import select_domain
from comce.stage2.regions import build_record
from comce.stage2.set_transformer import selector_probabilities
from comce.stage3.candidates import candidate_evidence, candidate_order
from comce.stage3.certification import certify_candidates
from comce.stage3.edit_cost import EdgeCost
from comce.stage3.search import generate_candidates
from comce.datasets.oracle import ExactOracle
from comce.training.world_model_update import apply_update


class StageTimer:
    def __init__(self):
        self.spans, self.counts = {}, {}

    @contextlib.contextmanager
    def span(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def add(self, name, seconds):
        self.spans[name] = self.spans.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + 1

    def payload(self):
        return {**{key: float(value) for key, value in sorted(self.spans.items())}, 'span_counts': dict(sorted(self.counts.items()))}


def check_architecture(models, config):
    stage2 = config.parameters['stage2']
    architecture = models.metadata['architecture']['set_transformer']
    expected = dict(num_pattern_slots=stage2['query_slots'], num_heads=stage2['attention_heads'], d_model=stage2['hidden_width'])
    mismatched = {k: (v, architecture[k]) for k, v in expected.items() if architecture[k] != v}
    if mismatched:
        raise ValueError(f'Configured Set Transformer architecture disagrees with the model bundle: {mismatched}')


class Explainer:
    def __init__(self, config, *, allowed_ids, device=None, event_sink=None, world_model_update=None):
        config.paths.require(config.paths.inference_keys())
        self.config = config
        self.device = device or config.execution['device']
        self.models = load_models(config.model_path, bank_path=config.signature_bank_path, device=self.device,
                                  stage1=config.stage1)
        self.world_model_update = None if world_model_update is None else apply_update(
            self.models, world_model_update, bundle_sha256=sha256(config.model_path), target_ids=allowed_ids)
        check_architecture(self.models, config)
        self.io, self.gnn = load_explained_model(config, allowed_ids=allowed_ids, device=self.device,
                                                 explained_model_sha256=self.models.metadata['explained_model_sha256'])
        self.search = config.search
        self.domain_params = config.domain
        self.task = config.task
        self.random_seed = int(config.execution['random_seed'])
        self.edge_cost = EdgeCost(**config.parameters['stage3']['edit_cost'])
        self.relations = list(config.dataset_config['relations'])
        self.event_sink = event_sink
        self.request_identity = {}
        self.timer = StageTimer()

    def _intent(self, value):
        return self.event_sink.intent(value) if self.event_sink is not None else None

    def _complete(self, event_id, value):
        if self.event_sink is not None:
            self.event_sink.complete(event_id, value)

    def start(self, center_id, *, split):
        with self.timer.span('stage_context.prepare'):
            context = self.io.prepare_context(str(center_id), split=split)
        counted = ExactOracle(self.search.query_cap)
        identity = {**self.request_identity, 'center_id': str(center_id)}
        physical_id = self._intent({**identity, 'stage': 'q0', 'deleted': [], 'canonical_business_edges': 0})
        forward_started = time.perf_counter()
        original = counted.observe(self.gnn, context)
        forward_wall = time.perf_counter() - forward_started
        self.timer.add('oracle.q0', forward_wall)
        self._complete(physical_id, {**identity, 'stage': 'q0', 'deleted': [], 'wall_s': forward_wall, 'status': 'COMPLETE'})
        self.io.accept_observation(original)
        q0 = original.response()
        events = [{'logical_index': 1, 'physical_event_id': physical_id or f'local:{counted.physical}', 'stage': 'q0',
                   'deleted': [], 'response': q0, 'raw_logits': original.output.raw_logits[0].detach().cpu().tolist(),
                   'flip': False, 'canonical_business_edges': 0, 'computation_edge_columns_removed': 0,
                   'oracle_wall': forward_wall}]
        session = {'center_id': str(center_id), 'context': context, 'counted': counted, 'original': original, 'q0': q0,
                   'events': events, 'query_cap': self.search.query_cap}
        session['eligibility_reason'] = eligibility_reason(
            self.task, center_id=center_id, receptive_edges=len(context.columns), original_prediction=q0['pred'],
            positive_class=int(self.gnn.decision_spec.positive_class),
            checkpoint_eligible=bool(self.gnn.explanation_eligible(original)))
        session['task_eligible'] = session['eligibility_reason'] is None
        if session['task_eligible']:
            with self.timer.span('stage_context.record'):
                record = build_record(self.io, self.models.signature_bank, self.relations)
            if record.get('witness_edge_keys') or record.get('witness_indices'):
                raise ValueError('live record contains witness')
            session['record'] = record
            with self.timer.span('stage_context.selector'):
                session['selector_probabilities'] = selector_probabilities(self.models.set_transformer, record, self.device)

        def oracle(keys):
            if not keys:
                return q0
            stage = session.get('query_stage', 'certification_or_cleanup')
            columns_removed = sum(len(context.columns[str(key)]) for key in keys)
            event_id = self._intent({**identity, 'stage': stage, 'deleted': list(keys), 'canonical_business_edges': len(keys),
                                     'computation_edge_columns_removed': columns_removed})
            started = time.perf_counter()
            observation = counted.intervene(self.gnn, context, keys)
            wall = time.perf_counter() - started
            response = observation.response(original)
            self._complete(event_id, {**identity, 'stage': stage, 'deleted': list(keys), 'wall_s': wall, 'status': 'COMPLETE'})
            events.append({'logical_index': len(events) + 1, 'physical_event_id': event_id or f'local:{counted.physical}',
                           'stage': stage, 'deleted': list(keys), 'response': response,
                           'raw_logits': observation.output.raw_logits[0].detach().cpu().tolist(), 'flip': bool(response['flip']),
                           'canonical_business_edges': len(keys), 'computation_edge_columns_removed': columns_removed,
                           'oracle_wall': wall})
            return response
        session['oracle'] = oracle
        return session

    def task_contract(self):
        spec = self.gnn.decision_spec
        return {'kind': spec.kind, 'num_classes': int(spec.num_classes), 'positive_class': int(spec.positive_class),
                'threshold': float(spec.threshold), 'temperature': float(self.gnn.deployment_temperature),
                'class_vocabulary': list(self.gnn.class_vocabulary)}

    def decision_params(self):
        params = self.domain_params
        return {'relevance_threshold': params.relevance_threshold, 'domain_edge_cap': params.domain_edge_cap,
                'admission_exponent': params.admission_exponent}

    def request(self, center_id, *, split, request_identity=None):
        started = time.perf_counter()
        self.timer = StageTimer()
        self.request_identity = dict(request_identity or {})
        with self.timer.span('stage_context'):
            session = self.start(center_id, split=split)
        if not session['task_eligible']:
            row = {'center_id': session['center_id'], 'success': False, 'task_eligible': False, 'q0': session['q0'],
                   'returned_edge_keys': [], 'returned_size': None, 'total_queries': 1, 'candidate_queries': 0,
                   'cleanup_queries': 0, 'macro_domain_size': 0, 'macro_domain_edge_keys': [], 'pool_size': 0,
                   'events': session['events'], 'actual_blackbox_forwards': session['counted'].physical,
                   'decision_params': self.decision_params(), 'failure_reason': 'TASK_INELIGIBLE',
                   'task_contract': self.task_contract(), 'eligibility_reason': session['eligibility_reason'],
                   'search_executed': False, 'query_ledger': {'q0': 1, 'certification': 0, 'cleanup': 0, 'physical_total': 1},
                   'resolved_configuration': self.config.payload()}
            row['timing'] = {'pipeline_wall': time.perf_counter() - started, **self.timer.payload()}
            return row
        record = session['record']
        edit_cost = self.edge_cost.bind(record)
        with self.timer.span('stage2'):
            domain, selection = select_domain(record, session['selector_probabilities'], self.domain_params, edit_cost)
        with self.timer.span('generation'):
            pool, search = generate_candidates(record, session['selector_probabilities'], domain, self.models, self.search,
                                               random_seed=self.random_seed, center_id=session['center_id'])

        def oracle(keys, phase):
            session['query_stage'] = phase
            with self.timer.span('oracle.' + phase):
                return session['oracle'](keys)
        # Trade-off: real-GNN calls per target are capped at 1 + Q + Q_clean regardless of graph size; raising
        # stage3.certification_queries / stage3.cleanup_queries buys fidelity and smaller explanations for latency.
        with self.timer.span('certification_and_cleanup'):
            result = certify_candidates([[pool.keys[j] for j in indices] for indices in pool.sets()], pool.raw, oracle,
                                        search['context'].final_ranking_score, q_cert=self.search.certification_queries,
                                        q_clean=self.search.cleanup_queries, kappa=self.search.lookahead,
                                        max_edges=self.search.explanation_budget, edit_cost=edit_cost,
                                        dedup_threshold=self.search.dedup_threshold)
        result.update(candidate_evidence(pool, candidate_order(pool, edit_cost, self.search.dedup_threshold)))
        final_domain = search['domain']
        counted = session['counted']
        if counted.physical != result['total_queries']:
            raise RuntimeError('physical/logical query mismatch')
        if result['success'] and not any(e['flip'] and set(e['deleted']) == set(result['returned_edge_keys'])
                                         for e in session['events']):
            raise RuntimeError('success lacks exact certificate')
        row = {'center_id': session['center_id'], 'success': result['success'], 'task_eligible': True, 'q0': session['q0'],
               'returned_edge_keys': result['returned_edge_keys'], 'returned_size': result['returned_size'],
               'total_queries': result['total_queries'], 'candidate_queries': result['candidate_queries'],
               'cleanup_queries': result['cleanup_queries'], 'macro_domain_size': final_domain['macro_domain_size'],
               'macro_domain_edge_keys': list(final_domain['macro_domain_edge_keys']), 'pool_size': len(pool),
               'community_count': len(record['communities']), 'true_e123': int(record['u123_size']),
               'selection_trace': selection, 'attempts': result['attempts'], 'cleanup_attempts': result['cleanup_attempts'],
               'events': session['events'], 'actual_blackbox_forwards': counted.physical,
               'candidate_set_sha256': result['candidate_set_sha256'], 'candidate_score_sha256': result['candidate_score_sha256'],
               'candidate_schedule_sha256': result['candidate_schedule_sha256'], 'candidate_schedule_count': result['schedule_count'],
               'decision_params': self.decision_params(),
               'failure_reason': None if result['success'] else 'NO_CERTIFICATE_FOUND_WITHIN_PROTOCOL',
               'task_contract': self.task_contract(), 'query_ledger': result['query_ledger'],
               'weighted_edit_cost': result['weighted_edit_cost'], 'edge_count': result['returned_size'],
               'edit_cost_contract': edit_cost.payload(), 'exact_certified': result['success'], 'search_executed': True,
               'resolved_configuration': self.config.payload(), 'scoring_contract': search['context'].scoring_contract,
               'stage3_stats': search['stats'], 'initial_macro_domain_edge_keys': list(domain['macro_domain_edge_keys']),
               'first_hit_rank': result['first_hit_rank'], 'best_before_cleanup': result['best_before_cleanup'],
               'certified_sets': result['certified_sets'], 'pruning_guarantee': result['pruning_guarantee'],
               'strict_local_minimality': False}
        if result['success']:
            row['witness_bookkeeping'] = {'center_id': session['center_id'], 'split': split,
                                          'edge_keys': list(result['returned_edge_keys']), 'exact_certified': True}
        row['timing'] = {'pipeline_wall': time.perf_counter() - started, **self.timer.payload()}
        return row

    def close(self):
        del self.gnn, self.io, self.models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
