"""Offline training of one (dataset, backbone) bundle on TRAIN targets only.

Stages: collect real-GNN flip labels -> Set Transformer -> WorldModel omega -> policy -> omega_bar and w_r on
real-GNN-labeled rollout candidates. Offline GNN queries are counted separately from the online budget.
"""
import copy
import hashlib
import pickle
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from comce.datasets.oracle import ExactOracle
from comce.pipeline.config import eligibility_reason
from comce.pipeline.features import FEATURE_SCHEMA_FILE, check_feature_schema, evaluation_populations
from comce.pipeline.io import sha256
from comce.pipeline.models import bundle_payload
from comce.pipeline.resources import load_explained_model
from comce.stage1.signatures import SignatureBank
from comce.stage2.domain import select_domain
from comce.stage2.regions import build_record
from comce.stage2.set_transformer import (SetTransformer, collate_center_records, model_forward, move_batch,
                                          selector_probabilities)
from comce.stage3.edit_cost import EdgeCost
from comce.stage3.policy import Policy, PolicyLogits, prepare_policy_inputs
from comce.stage3.ranking import RankingResidual, pooled_set_descriptors
from comce.stage3.rollout import CommunityExpansion, SearchContext, request_seed, run_rollouts
from comce.stage3.scoring import RequestScorer
from comce.stage3.set_context import SetContext
from comce.stage3.world_model import world_model_states
from comce.training.policy import RewardScorer, fit_policy, imitation_warm_start
from comce.training.selector import selector_loss


def training_targets(config):
    """TRAIN target IDs of datasets.<d>.training.population_path, in file order or shuffled with sampling_seed.

    Training supervision (real-GNN labels, witnesses, flips) is drawn from these targets only, so the population must
    share no canonical target ID with any evaluation population; otherwise training stops.
    """
    spec, paths = config.dataset_config['training'], config.paths
    population = paths.resolve(paths.key('training.population_path'))
    ordered = population.read_text().split()
    if len(set(ordered)) != len(ordered):
        raise ValueError(f'{spec["population_path"]} lists a TRAIN target twice')
    evaluation = evaluation_populations(paths)
    members = set(ordered)
    shared = {name: len(members & set(ids)) for name, ids in evaluation.items()}
    if any(shared.values()):
        raise ValueError(f'TRAIN population {spec["population_path"]} shares target IDs with evaluation populations '
                         f'{ {k: v for k, v in shared.items() if v} }; training stops')
    if spec.get('sampling_seed') is not None:
        ordered = [ordered[i] for i in np.random.default_rng(int(spec['sampling_seed'])).permutation(len(ordered))]
    isolation = dict(population_path=spec['population_path'], population_sha256=sha256(population),
                     sampling_seed=spec.get('sampling_seed'),
                     evaluation_populations={name: dict(ids=len(ids), shared_with_train=0) for name, ids in evaluation.items()})
    return ordered, isolation


def target_quota(config, template, available):
    """U_tr, highest precedence first: datasets.<d>.training.targets (an eligible count, or all), its target_fraction,
    training.target_fraction; a fraction inspects the first round(fraction * |labeled population|) TRAIN targets."""
    spec = config.dataset_config['training']
    targets, fraction = spec.get('targets'), spec.get('target_fraction')
    if targets is not None and fraction is not None:
        raise ValueError('datasets.<d>.training.targets and target_fraction are mutually exclusive')
    if targets == 'all':
        return dict(mode='all_eligible')
    if targets is not None:
        if type(targets) is not int or targets < 1:
            raise ValueError('datasets.<d>.training.targets is a positive integer, all, or null')
        return dict(mode='eligible_count', targets=targets)
    fraction = fraction if fraction is not None else template['training'].get('target_fraction')
    population = spec.get('labeled_population')
    if fraction is None or population is None:
        raise ValueError('training needs datasets.<d>.training.targets or a target_fraction with labeled_population')
    labeled = len(config.paths.population(population).read_text().split())
    quota = int(round(float(fraction) * labeled))
    if not 0 < quota <= available:
        raise ValueError(f'U_tr needs {quota} TRAIN targets ({fraction} x {labeled} labeled) but the TRAIN order has {available}')
    return dict(mode='labeled_fraction', fraction=float(fraction), labeled=labeled, inspected=quota)


class RecordCache:
    def __init__(self, root, memory_edge_limit=20000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory, self.memory_edge_limit = {}, memory_edge_limit

    def path(self, center):
        return self.root / f'{hashlib.sha256(str(center).encode()).hexdigest()[:24]}.pkl'

    def put(self, center, record):
        with self.path(center).open('wb') as stream:
            pickle.dump(record, stream, protocol=5)
        if len(record['edge_keys']) <= self.memory_edge_limit:
            self.memory[str(center)] = record

    def get(self, center):
        if str(center) in self.memory:
            return copy.copy(self.memory[str(center)])
        with self.path(center).open('rb') as stream:
            return pickle.load(stream)


def compact_record(record):
    record['edges'] = [dict(canonical_key=e['canonical_key'], hop=int(e['hop']), edge_type=e['edge_type']) for e in record['edges']]
    record.pop('edge_signatures', None)
    for community in record['communities']:
        community['member_indices'] = np.asarray(community['member_indices'], np.int32)
    return record


def supervision_candidates(record, rng, count, max_size=10):
    """Deletion sets with real-GNN labels: hop-1 singletons, all/leave-one-out hop-1, random mixtures and communities."""
    hops = np.asarray(record['hops'])
    keys = list(map(str, record['edge_keys']))
    near = np.flatnonzero(hops <= 2)
    first = np.flatnonzero(hops == 1)
    communities = [np.asarray(c['member_indices']) for c in record['communities'] if len(c['member_indices'])]
    out, seen = [], set()

    def add(indices):
        key = tuple(sorted({keys[int(i)] for i in indices}))
        if 0 < len(key) <= max_size and key not in seen:
            seen.add(key)
            out.append(key)
    for i in first[:max_size]:
        add([i])
    if 0 < len(first) <= max_size:
        add(first)
        for i in first:
            add(first[first != i])
    attempts = 0
    while len(out) < count and attempts < 40 * count:
        attempts += 1
        mode = rng.random()
        if mode < 0.35 and len(first):
            perm = rng.permutation(first)
            high = min(max_size, len(perm))
            low = 1 if rng.random() < 0.5 else max(1, (high + 1) // 2)
            add(perm[:int(rng.integers(low, high + 1))])
        elif mode < 0.55 and len(first) and len(near):
            perm = rng.permutation(first)[:int(rng.integers(1, min(max_size, len(first)) + 1))]
            extra = rng.choice(near, size=int(rng.integers(1, max(1, max_size - len(perm)) + 1)), replace=False)
            add(np.concatenate([perm, extra])[:max_size])
        elif mode < 0.8 and communities:
            members = communities[int(rng.integers(len(communities)))]
            add(members if len(members) <= max_size else rng.choice(members, size=int(rng.integers(1, max_size + 1)), replace=False))
        elif len(near):
            add(rng.choice(near, size=int(rng.integers(1, min(max_size, len(near)) + 1)), replace=False))
    return out[:count]


def wm_examples(record, candidates):
    index = {k: i for i, k in enumerate(map(str, record['edge_keys']))}
    x = np.asarray(record['edge_rows'], np.float32)
    universe = len(index)
    state = np.asarray(record['center_state'], np.float32)
    return [(x[[index[k] for k in candidate]],
             np.concatenate([state, np.asarray([len(candidate) / 10, len(candidate) / max(1, universe), np.log1p(universe)], np.float32)]))
            for candidate in candidates]


def to_tensors(examples, labels, edge_in, max_size=10):
    rows = np.zeros((len(examples), max_size, edge_in), np.float32)
    mask = np.zeros((len(examples), max_size), bool)
    for i, (r, _) in enumerate(examples):
        rows[i, :len(r)] = r
        mask[i, :len(r)] = True
    return (torch.from_numpy(rows), torch.from_numpy(mask), torch.from_numpy(np.stack([s for _, s in examples])),
            torch.as_tensor(np.asarray(labels, np.float32)))


def module_logits(module, rows, mask, set_features):
    return module(module.encode_edges(rows), set_features, mask)


def _bce(logits, labels):
    return float(F.binary_cross_entropy_with_logits(logits, labels).detach())


def _auc(scores, labels):
    scores, labels = np.asarray(scores, np.float64), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind='stable')
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_set_classifier(module, train, holdout, *, epochs, lr, device, seed, extra=None):
    """Binary cross-entropy on real-GNN flip labels; keeps the epoch with the lowest holdout loss."""
    torch.manual_seed(seed)
    module.to(device).train().requires_grad_(True)
    parameters = list(module.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    tensors = [t.to(device) for t in train]
    hold = [t.to(device) for t in holdout]
    best, best_state, history = float('inf'), None, []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        module.train()
        order = torch.randperm(len(tensors[0]), generator=generator).to(device)
        losses = []
        for begin in range(0, len(order), 256):
            batch = order[begin:begin + 256]
            rows, mask, setf, labels = (t[batch] for t in tensors)
            logits = extra(rows, mask, setf) if extra else module_logits(module, rows, mask, setf)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        module.eval()
        with torch.no_grad():
            rows, mask, setf, labels = hold
            logits = extra(rows, mask, setf) if extra else module_logits(module, rows, mask, setf)
            val = _bce(logits, labels)
            auc = _auc(logits.cpu().numpy(), labels.cpu().numpy())
        history.append(dict(epoch=epoch + 1, train_bce=float(np.mean(losses)), holdout_bce=val, holdout_auc=auc))
        if val < best:
            best = val
            best_state = copy.deepcopy([p.detach().cpu().clone() for p in parameters])
    if best_state is not None:
        with torch.no_grad():
            for p, saved in zip(parameters, best_state):
                p.copy_(saved.to(p.device))
    module.eval()
    return history


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class TupleTrainer:
    def __init__(self, config, template, *, device, work, signature_bank):
        config.paths.require(config.paths.training_keys())
        self.config = config
        self.dataset, self.backbone, self.device = config.dataset, config.backbone, device
        self.spec = template['training']
        self.ds = config.dataset_config['training']
        self.work = Path(work) / f'{self.dataset}_{self.backbone}'
        self.work.mkdir(parents=True, exist_ok=True)
        self.cache = RecordCache(self.work / 'records')
        self.bank_path = Path(signature_bank)
        self.bank = SignatureBank.load(self.bank_path)
        self.feature_schema = check_feature_schema(config.stage1, self.bank_path, self.bank)['fitted']
        self.centers, self.isolation = training_targets(config)
        self.quota = target_quota(config, template, len(self.centers))
        self.io, self.gnn = load_explained_model(config, allowed_ids={c: 'train' for c in self.centers}, device=device,
                                                 explained_model_sha256=config.tuple_config['gnn']['sha256'])
        self.relations = list(config.dataset_config['relations'])
        self.domain_params = config.domain
        self.search = config.search
        self.task = config.task
        self.random_seed = int(config.execution['random_seed'])
        self.offline_queries = dict(supervision=0, adaptation=0, q0=0)
        self.timing = {}

    def log(self, message):
        print(f'[{self.dataset}/{self.backbone} {time.strftime("%H:%M:%S")}] {message}', file=sys.stderr, flush=True)

    def _observe(self, center):
        context = self.io.prepare_context(center, split='train')
        oracle = ExactOracle(10 ** 9)
        original = oracle.observe(self.gnn, context)
        self.io.accept_observation(original)
        self.offline_queries['q0'] += 1
        reason = eligibility_reason(self.task, center_id=center, receptive_edges=len(context.columns),
                                    original_prediction=original.response()['pred'],
                                    positive_class=int(self.gnn.decision_spec.positive_class),
                                    checkpoint_eligible=bool(self.gnn.explanation_eligible(original)))
        return context, oracle, original, reason

    def label(self, context, oracle, original, sets, phase):
        out = []
        for keys in sets:
            response = oracle.intervene(self.gnn, context, list(keys)).response(original)
            out.append(bool(response['flip']))
            self.offline_queries[phase] += 1
        return out

    def collect(self):
        started = time.perf_counter()
        per = int(self.ds['supervision_sets'])
        target = self.quota.get('targets')
        centers = self.centers[:self.quota['inspected']] if self.quota['mode'] == 'labeled_fraction' else self.centers
        rng = np.random.default_rng(int(self.spec['seed']))
        rows, inspected, ineligible = [], 0, 0
        for center in centers:
            if target is not None and len(rows) >= target:
                break
            inspected += 1
            context, oracle, original, reason = self._observe(center)
            if reason is not None:
                ineligible += 1
                continue
            record = build_record(self.io, self.bank, self.relations)
            sets = supervision_candidates(record, rng, per)
            labels = self.label(context, oracle, original, sets, 'supervision')
            flips = [s for s, y in zip(sets, labels) if y]
            witness = min(flips, key=lambda s: (len(s), s)) if flips else None
            examples = wm_examples(record, sets)
            self.cache.put(center, compact_record(record))
            rows.append(dict(center=center, q0=original.response(), sets=sets, labels=labels, witness=witness, examples=examples))
            if len(rows) % 25 == 0:
                self.log(f'collected {len(rows)} eligible TRAIN targets ({inspected} inspected)')
        holdout = max(1, int(round(len(rows) * float(self.spec['holdout_fraction']))))
        for i, row in enumerate(sorted(rows, key=lambda r: hashlib.sha256(f"holdout|{r['center']}".encode()).hexdigest())):
            row['role'] = 'holdout' if i < holdout else 'train'
        self.rows = rows
        self.timing['collect'] = time.perf_counter() - started
        self.collection = dict(protocol=self.quota, isolation=self.isolation, targets=[r['center'] for r in rows],
                               inspected=inspected, ineligible=ineligible, eligible=len(rows),
                               with_witness=sum(r['witness'] is not None for r in rows),
                               labeled_sets=sum(len(r['sets']) for r in rows), positives=sum(sum(r['labels']) for r in rows))
        return self.collection

    def _record_with_witness(self, row):
        record = self.cache.get(row['center'])
        index = {k: i for i, k in enumerate(map(str, record['edge_keys']))}
        record['witness_edge_keys'] = list(row['witness'] or [])
        record['witness_indices'] = sorted(index[k] for k in record['witness_edge_keys'])
        return record

    def train_selector(self):
        started = time.perf_counter()
        cfg = self.spec['selector']
        stage2 = self.config.parameters['stage2']
        first = self.cache.get(self.rows[0]['center'])
        architecture = dict(signature_k=8, signature_m=16, signature_embed_dim=8,
                            center_state_dim=int(np.asarray(first['center_state']).shape[0]),
                            community_feature_dim=int(np.asarray(first['community_feats']).shape[1]),
                            num_pattern_slots=int(stage2['query_slots']), d_model=int(stage2['hidden_width']),
                            macro_hidden_dim=int(stage2['hidden_width']), num_heads=int(stage2['attention_heads']), dropout=0.1,
                            edge_feature_dim=int(np.asarray(first['edge_rows']).shape[1]), readout_dim=64,
                            locality_lambda_init=0.5, receptive_radius=3.0)
        torch.manual_seed(int(self.spec['seed']))
        model = SetTransformer(**architecture).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=0.)
        positive = [r for r in self.rows if r['witness'] is not None]
        train = [r for r in positive if r['role'] == 'train']
        hold = [r for r in positive if r['role'] == 'holdout']
        rng = np.random.default_rng(int(self.spec['seed']))
        history, best, best_state = [], float('inf'), None

        def loss_of(row):
            record = self._record_with_witness(row)
            batch = move_batch(collate_center_records([record], community_feature_dim=architecture['community_feature_dim'],
                                                      edge_feature_dim=architecture['edge_feature_dim']), torch.device(self.device))
            return selector_loss(model_forward(model, batch), batch, cfg)
        for epoch in range(int(cfg['epochs'])):
            model.train()
            losses = []
            for i in rng.permutation(len(train)):
                loss = loss_of(train[int(i)])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val = float(np.mean([float(loss_of(r)) for r in hold])) if hold else float(np.mean(losses))
            history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), holdout_loss=val))
            self.log(f'selector epoch {epoch + 1}: train {history[-1]["train_loss"]:.4f} holdout {val:.4f}')
            if val < best:
                best, best_state = val, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        model.load_state_dict(best_state)
        model.eval().requires_grad_(False)
        self.selector, self.selector_architecture, self.selector_history = model, architecture, history
        self.timing['selector'] = time.perf_counter() - started
        return history

    def train_world_model(self):
        started = time.perf_counter()
        cfg = self.spec['world_model']
        edge_in = int(self.rows[0]['examples'][0][0].shape[1])
        set_dim = int(self.rows[0]['examples'][0][1].shape[0])
        torch.manual_seed(int(self.spec['seed']))
        world_model = world_model_states(edge_in, set_dim, int(cfg['hidden']))
        residual = RankingResidual(edge_in)
        split = {role: [(e, y) for r in self.rows if r['role'] == role for e, y in zip(r['examples'], r['labels'])]
                 for role in ('train', 'holdout')}
        train = to_tensors([e for e, _ in split['train']], [y for _, y in split['train']], edge_in)
        hold = to_tensors([e for e, _ in split['holdout']], [y for _, y in split['holdout']], edge_in)
        history = fit_set_classifier(world_model['omega'], train, hold, epochs=int(cfg['epochs']), lr=float(cfg['lr']),
                                     device=self.device, seed=int(self.spec['seed']))
        world_model.to(self.device).eval()
        residual.to(self.device).eval()
        world_model['omega_bar'].load_state_dict(world_model['omega'].state_dict())
        with torch.no_grad():
            residual.linear.weight.zero_(); residual.linear.bias.zero_()
        self.world_model, self.residual, self.wm_history = world_model, residual, history
        self.wm_shape = dict(edge_in=edge_in, set_dim=set_dim, hidden=int(cfg['hidden']))
        self.timing['world_model'] = time.perf_counter() - started
        return history

    def _domain(self, record):
        probabilities = selector_probabilities(self.selector, record, self.device)
        domain, _ = select_domain(record, probabilities, self.domain_params, EdgeCost().bind(record))
        return probabilities, domain

    def _scorer(self, record, domain, signatures=None):
        return RequestScorer(self.world_model, self.residual, SetContext(record, domain, self.device), signatures)

    def train_policy(self):
        started = time.perf_counter()
        cfg = self.spec['policy']
        samples, scorers, self.domains = [], {}, {}
        for row in self.rows:
            record = self._record_with_witness(row)
            probabilities, domain = self._domain(record)
            self.domains[row['center']] = (probabilities, domain)
            if not domain['macro_domain_edge_keys']:
                continue
            samples.append(dict(center_id=str(row['center']), role='TRAIN' if row['role'] == 'train' else 'HOLDOUT',
                                inputs=prepare_policy_inputs(record, domain)))
            scorers[str(row['center'])] = self._scorer(record, domain)
        first = samples[0]['inputs']
        architecture = dict(edge_in_dim=int(first['edge_inputs'].shape[1]), hidden=64,
                            center_state_dim=int(first['center_state'].shape[0]), ctx_extra=4, budget=10)
        torch.manual_seed(int(self.spec['seed']))
        policy = Policy(**architecture)
        exp = SimpleNamespace(rl_lr=float(cfg['lr']), rl_seed=int(self.spec['seed']), rl_max_candidate_edges=10,
                              rl_entropy_weight=float(cfg['entropy_weight']), rl_lambda_flip=float(cfg['lambda_flip']),
                              rl_lambda_size=float(cfg['lambda_size']), rl_lambda_div=float(cfg['lambda_div']),
                              rl_bc_anchor_weight=float(cfg['bc_anchor_weight']), rl_stop_aux_weight=float(cfg['stop_aux_weight']),
                              rl_rollouts=int(cfg['rollouts']))
        train = [s for s in samples if s['role'] == 'TRAIN']
        holdout = [s for s in samples if s['role'] == 'HOLDOUT']
        imitation = imitation_warm_start(policy, train, exp=exp, device=self.device, epochs=int(cfg['imitation_epochs']),
                                         log_fn=lambda *_: None)
        reinforce = fit_policy(policy, train, holdout, scorer=RewardScorer(scorers), exp=exp,
                               n_rollouts=int(cfg['rollouts']), epochs=int(cfg['reinforce_epochs']), log_fn=lambda *_: None)
        policy.eval().requires_grad_(False)
        self.policy, self.policy_architecture = policy, architecture
        self.policy_history = dict(imitation=imitation['loss_curve'], reinforce=reinforce['curve'], best_val=reinforce['best_val'])
        self.timing['policy'] = time.perf_counter() - started
        return self.policy_history

    def _rollout_pool(self, row, record):
        """Rollouts of the trained policy ranked by z_omega (S is not trained yet)."""
        probabilities, domain = self.domains[row['center']]
        if not domain['macro_domain_edge_keys']:
            return []

        def context(current):
            scorer = self._scorer(record, current, getattr(record, 'signature_context', None))
            inputs = prepare_policy_inputs(record, current)
            inputs.pop('witness_keys', None)

            def rank(candidates):
                original = [tuple(sorted(c)) for c in candidates]
                unique = list(dict.fromkeys(original))
                if not unique:
                    return np.empty(0, np.float64)
                scores = scorer.world_model_scores(unique)
                position = {c: i for i, c in enumerate(unique)}
                return scores[[position[c] for c in original]]
            return SearchContext(scorer.keys, PolicyLogits(self.policy, inputs), scorer.world_model_scores, rank)
        expansion = CommunityExpansion(record, probabilities, domain, self.search.expansion_budget,
                                       self.search.expansion_communities_per_trigger)
        result = run_rollouts(domain, expansion, context, seed=request_seed(self.random_seed, str(row['center'])),
                              settings=self.search)
        return list(zip(result['sets'], result['scores']))

    def train_adaptation(self):
        started = time.perf_counter()
        cfg = self.spec['adaptation']
        rng = np.random.default_rng(int(self.spec['seed']) + 7)
        adapted = {'train': [], 'holdout': []}
        for position, row in enumerate(self.rows):
            if position % 25 == 0:
                self.log(f'adaptation rollouts {position}/{len(self.rows)}')
            record = self._record_with_witness(row)
            pool = self._rollout_pool(row, record)
            if not pool:
                continue
            ranked = [s for s, _ in sorted(pool, key=lambda item: (-item[1], item[0]))]
            top = ranked[:int(cfg['top_sets'])]
            rest = ranked[int(cfg['top_sets']):]
            extra = [rest[int(i)] for i in rng.permutation(len(rest))[:int(cfg['random_sets'])]] if rest else []
            sets = top + extra
            context, oracle, original, _ = self._observe(row['center'])
            labels = self.label(context, oracle, original, sets, 'adaptation')
            for example, y in zip(wm_examples(record, sets), labels):
                adapted[row['role']].append((example, y))
        edge_in = self.wm_shape['edge_in']
        train = to_tensors([e for e, _ in adapted['train']], [y for _, y in adapted['train']], edge_in)
        hold = to_tensors([e for e, _ in adapted['holdout']], [y for _, y in adapted['holdout']], edge_in)
        self.adaptation_counts = {k: dict(sets=len(v), positives=int(sum(y for _, y in v))) for k, v in adapted.items()}
        before = self._evaluate_scores(hold)
        omega_bar = self.world_model['omega_bar']
        adapted_history = fit_set_classifier(omega_bar, train, hold, epochs=int(cfg['epochs']), lr=float(cfg['lr']),
                                             device=self.device, seed=int(self.spec['seed']) + 1)
        omega_bar.requires_grad_(False)
        residual = self.residual.to(self.device)
        residual.requires_grad_(True)

        def ranking_score(rows, mask, setf):
            with torch.no_grad():
                base = module_logits(omega_bar, rows, mask, setf)
            return base + residual(rows, mask)
        weight_before = residual.linear.weight.detach().clone()
        residual_history = fit_set_classifier(residual, train, hold, epochs=int(cfg['residual_epochs']), lr=float(cfg['residual_lr']),
                                              device=self.device, seed=int(self.spec['seed']) + 2, extra=ranking_score)
        self.world_model.eval().requires_grad_(False)
        residual.eval().requires_grad_(False)
        self.adaptation_history = dict(adapted=adapted_history, residual=residual_history, holdout_before=before,
                                       holdout_after=self._evaluate_scores(hold),
                                       residual_weight_change=float((residual.linear.weight.detach() - weight_before).abs().sum()),
                                       residual_weight_l1=float(residual.linear.weight.detach().abs().sum()),
                                       residual_bias=float(residual.linear.bias.detach()))
        self.timing['adaptation'] = time.perf_counter() - started
        return self.adaptation_history

    def _evaluate_scores(self, tensors):
        rows, mask, setf, labels = (t.to(self.device) for t in tensors)
        with torch.no_grad():
            omega = module_logits(self.world_model['omega'], rows, mask, setf)
            omega_bar = module_logits(self.world_model['omega_bar'], rows, mask, setf)
            final = omega_bar + self.residual.linear(pooled_set_descriptors(rows, mask, self.residual.max_set_size)).squeeze(-1)
        y = labels.cpu().numpy()
        return {name: dict(bce=_bce(v, labels), auc=_auc(v.cpu().numpy(), y))
                for name, v in (('omega', omega), ('omega_bar', omega_bar), ('S', final))}

    def save(self, models_root):
        models_root = Path(models_root)
        bank_target = models_root / 'signatures' / self.dataset / 'signature_bank.pt'
        if not bank_target.exists():
            bank_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.bank_path, bank_target)
            if self.bank_path.with_name(FEATURE_SCHEMA_FILE).is_file():
                shutil.copy2(self.bank_path.with_name(FEATURE_SCHEMA_FILE), bank_target.with_name(FEATURE_SCHEMA_FILE))
        architecture = dict(set_transformer=self.selector_architecture, world_model=self.wm_shape,
                            ranking_residual=dict(edge_in=self.wm_shape['edge_in'], max_set_size=self.residual.max_set_size),
                            policy=self.policy_architecture)
        training = _plain(dict(seed=int(self.spec['seed']),
                               train_split='TRAIN only; disjoint from ' + ', '.join(self.ds['evaluation_populations']),
                               collection=self.collection, offline_gnn_queries=self.offline_queries, hidden_states_used=False,
                               selector_history=self.selector_history, world_model_history=self.wm_history,
                               policy_history=self.policy_history, adaptation=self.adaptation_history,
                               adaptation_counts=self.adaptation_counts, timing_seconds=self.timing))
        payload = bundle_payload(dataset=self.dataset, backbone=self.backbone, set_transformer=self.selector,
                                 world_model=self.world_model, ranking_residual=self.residual, policy=self.policy,
                                 architecture=architecture,
                                 signature_bank=dict(path=f'signatures/{self.dataset}/signature_bank.pt', sha256=sha256(bank_target)),
                                 explained_model_sha256=sha256(self.config.gnn_path),
                                 input_schema=dict(relations=self.relations, feature_schema=self.feature_schema),
                                 training=training)
        target = models_root / self.dataset / self.backbone / 'comce.pt'
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        return target


def train_tuple(config, template, *, device, work, signature_bank, models_root):
    trainer = TupleTrainer(config, template, device=device, work=work, signature_bank=signature_bank)
    trainer.log(f'collection: {trainer.collect()}')
    trainer.train_selector()
    trainer.log(f'world model holdout: {trainer.train_world_model()[-1]}')
    trainer.log(f'policy best holdout flip probability: {trainer.train_policy()["best_val"]}')
    adaptation = trainer.train_adaptation()
    trainer.log(f'adaptation holdout before {adaptation["holdout_before"]} after {adaptation["holdout_after"]}')
    return trainer.save(models_root)
