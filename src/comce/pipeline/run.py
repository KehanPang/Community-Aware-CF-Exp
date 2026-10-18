"""Population runs with durable per-request journals and an immutable run contract."""
from __future__ import annotations

import contextlib
import fcntl
import gc
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch

from comce.pipeline.explainer import Explainer
from comce.pipeline.io import PACKAGE_ROOT, append_jsonl, atomic_json, canonical_json, digest, read_jsonl_strict, sha256


class AppendJournal:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open('a', encoding='utf-8')

    def append(self, value: Any) -> float:
        started = time.perf_counter()
        self._stream.write(canonical_json(value) + '\n')
        self._stream.flush()
        os.fsync(self._stream.fileno())
        return time.perf_counter() - started

    def close(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


class PhysicalEventSink:
    """Write-ahead record of every real-GNN forward: an INTENT before the call and a COMPLETE after it."""

    def __init__(self, result_dir: Path):
        self.journal = AppendJournal(result_dir / 'PHYSICAL_EVENTS.jsonl')
        self.run_id = 'UNBOUND'
        self.request_id = 'UNBOUND'
        self.attempt_id = 0
        self.call_index = 0
        self.io_wall = 0.0
        self.oracle_walls: list[float] = []
        self.pending: set[str] = set()

    def begin_request(self, request_id: str, attempt_id: int) -> None:
        self.request_id, self.attempt_id = str(request_id), int(attempt_id)
        self.call_index, self.io_wall, self.oracle_walls, self.pending = 0, 0.0, [], set()

    def intent(self, value: dict[str, Any]) -> str:
        self.call_index += 1
        event_id = f'{self.run_id}:{self.request_id}:{self.attempt_id}:{self.call_index}'
        self.io_wall += self.journal.append({'event': 'INTENT', 'physical_event_id': event_id, 'run_id': self.run_id,
                                             'request_id': self.request_id, 'attempt_id': self.attempt_id,
                                             'call_index': self.call_index, 'time_ns': time.time_ns(), **value})
        self.pending.add(event_id)
        return event_id

    def complete(self, event_id: str | None, value: dict[str, Any]) -> None:
        if event_id is None or event_id not in self.pending:
            raise RuntimeError('physical completion lacks matching in-process intent')
        wall = float(value['wall_s'])
        if not math.isfinite(wall) or wall < 0:
            raise RuntimeError('invalid oracle wall time')
        self.io_wall += self.journal.append({'event': 'COMPLETE', 'physical_event_id': event_id, 'run_id': self.run_id,
                                             'request_id': self.request_id, 'attempt_id': self.attempt_id,
                                             'time_ns': time.time_ns(), **value})
        self.oracle_walls.append(wall)
        self.pending.remove(event_id)

    def close(self) -> None:
        self.journal.close()

    def request_summary(self) -> dict[str, Any]:
        values = sorted(self.oracle_walls)
        p95 = None if not values else values[min(len(values) - 1, math.ceil(.95 * len(values)) - 1)]
        return {'oracle_wall_total': float(sum(values)), 'oracle_call_count': len(values),
                'oracle_call_mean': None if not values else float(sum(values) / len(values)), 'oracle_call_p95': p95,
                'evidence_io_wall_physical': self.io_wall, 'dangling_physical_intents': sorted(self.pending)}


@contextlib.contextmanager
def runtime_environment(execution):
    previous = dict(threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                    warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(int(execution['threads']))
    torch.manual_seed(int(execution['random_seed']))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(execution['random_seed']))
    torch.use_deterministic_algorithms(bool(execution['deterministic']), warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = bool(execution['allow_tf32'])
    torch.backends.cudnn.allow_tf32 = bool(execution['allow_tf32'])
    try:
        yield
    finally:
        torch.set_num_threads(previous['threads'])
        torch.use_deterministic_algorithms(previous['deterministic'], warn_only=previous['warn_only'])
        torch.backends.cuda.matmul.allow_tf32 = previous['matmul_tf32']
        torch.backends.cudnn.allow_tf32 = previous['cudnn_tf32']


def source_digest():
    files = sorted(p for p in PACKAGE_ROOT.rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml'})
    return digest([[str(p.relative_to(PACKAGE_ROOT)), sha256(p)] for p in files])


def population_binding(ids: Iterable[str], *, split: str, source: str, selection_rule: str) -> dict[str, Any]:
    payload = {'ordered_ids': list(map(str, ids)), 'split': split, 'selection_rule': selection_rule, 'source': source}
    payload['population_id'] = digest(payload)
    return payload


def resource_state() -> dict[str, Any]:
    state = {'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        state.update(cuda_free_bytes=int(free), cuda_total_bytes=int(total), cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    return state


def _cost_and_sparsity(row: dict[str, Any]) -> tuple[int | None, float | None]:
    if not row['task_eligible']:
        return None, None
    deleted = [event['deleted'] for event in row['events'] if event.get('deleted')]
    cost = int(row['returned_size']) if row['success'] else max((len(keys) for keys in deleted), default=10)
    return cost, cost / int(row['true_e123'])


def run_population(config, *, ids, split, result_dir, role='adhoc', population_source='command line',
                   selection_rule='explicit target list', device=None, world_model_update=None) -> dict[str, Any]:
    from comce.pipeline.io import PROJECT_ROOT
    config.paths.require(config.paths.inference_keys())
    result_dir = Path(result_dir).resolve()
    if result_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError('Outputs must be outside the release repository')
    result_dir.mkdir(parents=True, exist_ok=True)
    lock = (result_dir / '.WRITER.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(f'another writer owns {result_dir}') from exc
    ordered_ids = list(map(str, ids))
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError('population contains duplicate centers')
    population = population_binding(ordered_ids, split=split, source=str(population_source), selection_rule=selection_rule)
    sink = PhysicalEventSink(result_dir)
    explainer = journal = commits = None
    init_started = time.perf_counter()
    try:
        execution = config.execution
        torch.set_num_threads(int(execution['threads']))
        torch.manual_seed(int(execution['random_seed']))
        torch.use_deterministic_algorithms(bool(execution['deterministic']), warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = bool(execution['allow_tf32'])
        torch.backends.cudnn.allow_tf32 = bool(execution['allow_tf32'])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        explainer = Explainer(config, allowed_ids={c: split for c in ordered_ids}, device=device, event_sink=sink,
                              world_model_update=world_model_update)
        init_wall = time.perf_counter() - init_started
        contract = dict(schema='comce-run-contract', dataset=config.dataset, backbone=config.backbone, role=role,
                        configuration=config.payload(), model_bundle_sha256=sha256(explainer.models.path),
                        signature_bank_sha256=explainer.models.metadata['signature_bank']['sha256'],
                        explained_model_sha256=explainer.models.metadata['explained_model_sha256'],
                        source_sha256=source_digest(), population=population)
        if explainer.world_model_update is not None:
            contract['world_model_update'] = explainer.world_model_update
        contract['run_contract_id'] = digest(contract)
        sink.run_id = contract['run_contract_id']
        contract_path = result_dir / 'CONTRACT.json'
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != json.loads(canonical_json(contract)):
                raise RuntimeError('immutable run contract mismatch; use a new output directory')
        else:
            atomic_json(contract_path, contract)
        journal_path = result_dir / 'REQUESTS.jsonl'
        rows = read_jsonl_strict(journal_path)
        if len(rows) > len(ordered_ids):
            raise RuntimeError('journal is longer than the population')
        committed = [str(row.get('center_id')) for row in rows]
        if committed != ordered_ids[:len(rows)]:
            raise RuntimeError('journal is not the exact ordered population prefix')
        journal = AppendJournal(journal_path)
        commits = AppendJournal(result_dir / 'COMMIT_EVENTS.jsonl')
        complete_path = result_dir / 'COMPLETE.json'
        if complete_path.exists():
            complete = json.loads(complete_path.read_text())
            if len(rows) != len(ordered_ids) or complete.get('run_contract_id') != contract['run_contract_id']:
                raise RuntimeError('invalid completed marker')
            return complete
        active = result_dir / 'CURRENT_REQUEST.json'
        attempt_counts: dict[str, int] = {}
        for item in read_jsonl_strict(result_dir / 'DANGLING_ATTEMPTS.jsonl'):
            attempt_counts[str(item['request_id'])] = max(attempt_counts.get(str(item['request_id']), 0), int(item['attempt_id']))
        if active.exists():
            dangling = json.loads(active.read_text())
            append_jsonl(result_dir / 'DANGLING_ATTEMPTS.jsonl', {
                **dangling, 'resolution': 'INCOMPLETE_ATTEMPT_RETRY_WITH_NEW_ATTEMPT_ID', 'physical_cost_lower_bound': 0,
                'physical_cost_upper_bound': config.search.query_cap, 'noticed_time_ns': time.time_ns()})
            attempt_counts[str(dangling['request_id'])] = max(attempt_counts.get(str(dangling['request_id']), 0),
                                                              int(dangling['attempt_id']))
        with runtime_environment(execution):
            for index, center in enumerate(ordered_ids[len(rows):], start=len(rows)):
                request_id = f'{index}:{center}'
                attempt_id = attempt_counts.get(request_id, 0) + 1
                intent = {'run_contract_id': contract['run_contract_id'], 'request_id': request_id, 'attempt_id': attempt_id,
                          'center_id': center, 'population_index': index, 'status': 'IN_PROGRESS', 'time_ns': time.time_ns()}
                intent_io = time.perf_counter()
                atomic_json(active, intent)
                intent_io = time.perf_counter() - intent_io
                sink.begin_request(request_id, attempt_id)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                request_started = time.perf_counter()
                row = explainer.request(center, split=split, request_identity={
                    'run_contract_id': contract['run_contract_id'], 'request_id': request_id, 'attempt_id': attempt_id})
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                precommit_wall = time.perf_counter() - request_started
                physical = sink.request_summary()
                if physical['dangling_physical_intents']:
                    raise RuntimeError('request ended with a dangling physical forward intent')
                if physical['oracle_call_count'] != int(row['actual_blackbox_forwards']):
                    raise RuntimeError('physical sink/oracle count mismatch')
                cost, sparsity = _cost_and_sparsity(row)
                row.update(dataset=config.dataset, backbone=config.backbone, role=role, population_index=index,
                           request_id=request_id, attempt_id=attempt_id, run_contract_id=contract['run_contract_id'],
                           population_deleted_cost=cost, population_deleted_edge_count=cost,
                           population_deleted_cost_unit='edge_count', population_sparsity_legal=sparsity,
                           scientific_status='COMPLETE' if row['task_eligible'] else 'TASK_INELIGIBLE',
                           timing={**row.get('timing', {}), **physical, 'request_wall_precommit': precommit_wall,
                                   'evidence_io_wall_intent': intent_io})
                if explainer.world_model_update is not None:
                    row['world_model_update'] = explainer.world_model_update
                row_io = journal.append(row)
                commits.append({'run_contract_id': contract['run_contract_id'], 'request_id': request_id,
                                'attempt_id': attempt_id, 'center_id': center, 'population_index': index,
                                'request_wall': time.perf_counter() - request_started, 'request_row_io_wall': row_io,
                                'request_intent_io_wall': intent_io, 'status': 'COMMITTED', 'time_ns': time.time_ns()})
                active.unlink(missing_ok=True)
                rows.append(row)
                if index < 2 or (index + 1) % 8 == 0 or index + 1 == len(ordered_ids):
                    atomic_json(result_dir / 'HEARTBEAT.json', {
                        'run_contract_id': contract['run_contract_id'], 'completed': index + 1, 'total': len(ordered_ids),
                        'dataset': config.dataset, 'backbone': config.backbone, 'role': role, 'resources': resource_state(),
                        'time_ns': time.time_ns()})
        complete = {'schema': 'comce-run-complete', 'run_contract_id': contract['run_contract_id'], 'n': len(rows),
                    'target': len(ordered_ids), 'init_wall': init_wall, 'resources': resource_state(),
                    'completed_time_ns': time.time_ns()}
        atomic_json(complete_path, complete)
        return complete
    finally:
        for closable in (journal, commits, sink):
            if closable is not None:
                closable.close()
        if explainer is not None:
            explainer.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
