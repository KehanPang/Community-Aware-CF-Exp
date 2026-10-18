"""Optional update of the WorldModel state omega from real-GNN certification labels of TRAIN targets.

omega_eff = omega + gate * delta. delta has omega's parameter shapes, starts at zero and is fitted with omega frozen
(the gate is not part of the fit, so delta receives gradients); the gate is then 1 only when the BCE on held-out TRAIN
targets drops by more than min_improvement with at least min_validation_labels labels, otherwise 0, which is omega
exactly. The fit evaluates the one WorldModel forward with effective parameters (torch.func.functional_call); omega_bar
and the ranking residual are not touched. ComCE names periodic fine-tuning of F_omega on certification labels but gives
no schedule, loss weights or acceptance rule: training.world_model_update holds these engineering choices.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from comce.pipeline.features import evaluation_populations
from comce.pipeline.io import read_jsonl_strict, sha256
from comce.pipeline.resources import load_explained_model
from comce.stage2.regions import RegionConstructor, region_inputs
from comce.stage3.ranking import CandidateRanker
from comce.stage3.scoring import RequestScorer
from comce.stage3.set_context import SetContext
from comce.training.trainer import _auc, to_tensors, training_targets, wm_examples

UPDATE_FORMAT = 'comce-world-model-update'
LABEL_STAGES = ('certification', 'cleanup')


def world_model_logits(module, parameters, rows, mask, set_features):
    """z = F_omega(Pool(Enc_omega(x_e)), c_{u,Delta}) of the WorldModel evaluated with the given parameters."""
    encoder = {k[len('edge_enc.'):]: v for k, v in parameters.items() if k.startswith('edge_enc.')}
    embeddings = functional_call(module.edge_enc, encoder, (rows,))
    return functional_call(module, parameters, (embeddings, set_features, mask))


def _journal(run_dir):
    events = {}
    for entry in read_jsonl_strict(run_dir / 'PHYSICAL_EVENTS.jsonl'):
        events.setdefault(entry['physical_event_id'], {})[entry['event']] = entry
    return events


def certification_labels(config, models, run_dirs, *, bundle_sha256):
    """(center, deletion set, real-GNN flip) triples from run directories of TRAIN targets, with provenance checks."""
    train_ids, isolation = training_targets(config)
    train_ids = set(train_ids)
    evaluation = {name: set(ids) for name, ids in evaluation_populations(config.paths).items()}
    rows, logs = [], []
    for run_dir in map(Path, run_dirs):
        contract = json.loads((run_dir / 'CONTRACT.json').read_text())
        expected = dict(dataset=config.dataset, backbone=config.backbone, explained_model_sha256=config.tuple_config['gnn']['sha256'],
                        model_bundle_sha256=bundle_sha256, signature_bank_sha256=models.metadata['signature_bank']['sha256'])
        wrong = {k: (contract.get(k), v) for k, v in expected.items() if contract.get(k) != v}
        if wrong:
            raise ValueError(f'{run_dir}: certification log of another model or graph: {wrong}')
        if contract['population']['split'] != 'train':
            raise ValueError(f'{run_dir}: the certification log is not a run on TRAIN targets (split {contract["population"]["split"]!r})')
        journal = _journal(run_dir)
        requests = run_dir / 'REQUESTS.jsonl'
        for row in read_jsonl_strict(requests):
            center = str(row['center_id'])
            if row['run_contract_id'] != contract['run_contract_id']:
                raise ValueError(f'{run_dir}: row of another run')
            shared = [name for name, ids in evaluation.items() if center in ids]
            if shared or center not in train_ids:
                raise ValueError(f'{run_dir}: target {center} is not a TRAIN target'
                                 + (f' (it belongs to evaluation population(s) {shared})' if shared else ''))
            labels = []
            for event in row['events']:
                if event['stage'] not in LABEL_STAGES or not event['deleted']:
                    continue
                entry = journal.get(event['physical_event_id'], {})
                if set(entry) != {'INTENT', 'COMPLETE'} or entry['INTENT']['deleted'] != event['deleted']:
                    raise ValueError(f'{run_dir}: label {event["physical_event_id"]} has no journaled real-GNN call')
                labels.append((tuple(sorted(event['deleted'])), bool(event['flip'])))
            if labels:
                rows.append(dict(row=row, labels=labels))
        logs.append(dict(run_dir=str(run_dir), run_contract_id=contract['run_contract_id'], requests_sha256=sha256(requests),
                         population_id=contract['population']['population_id']))
    return rows, logs, isolation


def rebuild_examples(config, models, labeled, *, device):
    """Region records of the labeled targets rebuilt from the logged pre-intervention response (no GNN call), checked
    against the ranking scores S logged during certification."""
    io, _ = load_explained_model(config, allowed_ids={str(r['row']['center_id']): 'train' for r in labeled}, device=device,
                                 explained_model_sha256=models.metadata['explained_model_sha256'])
    relations = list(config.dataset_config['relations'])
    out, worst = [], 0.0
    for item in labeled:
        row = item['row']
        io.prepare_context(str(row['center_id']), split='train')
        edges, center_type, center_row = io.region_edges()
        record = RegionConstructor().construct(region_inputs(
            center_id=io.center, center_type=center_type, center_row=center_row, edges=edges, bank=models.signature_bank,
            relation_names=relations, probabilities=row['q0']['probs'], prediction=row['q0']['pred']))
        if int(record['u123_size']) != int(row['true_e123']):
            raise ValueError(f'target {row["center_id"]}: rebuilt receptive field differs from the certification log')
        attempts = [a for a in row['attempts'] if a.get('keys')]
        if attempts:
            scorer = RequestScorer(models.world_model, models.ranking_residual,
                                   SetContext(record, dict(macro_domain_edge_keys=row['macro_domain_edge_keys']), device),
                                   record.signature_context)
            scores = CandidateRanker(scorer)([tuple(a['keys']) for a in attempts])
            worst = max(worst, float(np.max(np.abs(scores - np.asarray([a['score'] for a in attempts], np.float64)))))
        sets = [s for s, _ in item['labels']]
        out.append(dict(center=str(row['center_id']), examples=wm_examples(record, sets), labels=[y for _, y in item['labels']]))
    return out, worst


def split_by_target(examples, fraction, seed):
    centers = sorted({e['center'] for e in examples}, key=lambda c: hashlib.sha256(f'{seed}|{c}'.encode()).hexdigest())
    held = set(centers[:int(round(len(centers) * float(fraction)))])
    return [e for e in examples if e['center'] not in held], [e for e in examples if e['center'] in held]


def _tensors(items, edge_in, device):
    examples = [x for item in items for x in item['examples']]
    labels = [y for item in items for y in item['labels']]
    if not examples:
        return None
    return [t.to(device) for t in to_tensors(examples, labels, edge_in)]


def fit_update(module, fit, validation, settings, *, device):
    """delta minimizing BCE + l2_weight ||delta||^2 + stability_weight BCE(p_eff, p_omega) on the fit labels; the gate is
    decided on the validation labels only."""
    module = module.to(device).eval().requires_grad_(False)
    base = {k: v.detach() for k, v in module.named_parameters()}
    delta = {k: torch.zeros_like(v, requires_grad=True) for k, v in base.items()}
    torch.manual_seed(int(settings['seed']))
    optimizer = torch.optim.Adam(list(delta.values()), lr=float(settings['lr']))
    rows, mask, set_features, labels = fit
    with torch.no_grad():
        anchor = torch.sigmoid(world_model_logits(module, base, rows, mask, set_features))
    history = []
    for epoch in range(int(settings['epochs'])):
        logits = world_model_logits(module, {k: base[k] + delta[k] for k in base}, rows, mask, set_features)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        stability = F.binary_cross_entropy_with_logits(logits, anchor)
        l2 = sum((d ** 2).sum() for d in delta.values())
        loss = bce + float(settings['stability_weight']) * stability + float(settings['l2_weight']) * l2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        history.append(dict(epoch=epoch + 1, fit_bce=float(bce.detach()), loss=float(loss.detach())))

    def metrics(tensors, parameters):
        if tensors is None:
            return None
        with torch.no_grad():
            z = world_model_logits(module, parameters, *tensors[:3])
        y = tensors[3]
        return dict(labels=int(len(y)), positives=int(y.sum()), bce=float(F.binary_cross_entropy_with_logits(z, y)),
                    auc=_auc(z.cpu().numpy(), y.cpu().numpy()))
    updated = {k: (base[k] + delta[k]).detach() for k in base}
    result = dict(fit=dict(omega=metrics(fit, base), updated=metrics(fit, updated)),
                  validation=dict(omega=metrics(validation, base), updated=metrics(validation, updated)), history=history)
    held = result['validation']
    enough = validation is not None and held['omega']['labels'] >= int(settings['min_validation_labels'])
    improved = enough and held['updated']['bce'] < held['omega']['bce'] - float(settings['min_improvement'])
    result['decision'] = dict(enough_validation_labels=bool(enough), improved=bool(improved),
                              rule=f'gate = 1 iff validation labels >= {settings["min_validation_labels"]} and validation BCE '
                                   f'drops by more than {settings["min_improvement"]}')
    return {k: v.detach().cpu() for k, v in delta.items()}, 1.0 if improved else 0.0, result


def apply_update(models, path, *, bundle_sha256, target_ids):
    """Load an accepted update into this process's copy of omega; the bundle file and omega_bar stay as they are."""
    import copy
    update = torch.load(path, map_location='cpu', weights_only=True)
    if update.get('format') != UPDATE_FORMAT or update.get('state') != 'omega':
        raise ValueError(f'{path} is not a ComCE WorldModel update')
    if update['base_bundle_sha256'] != bundle_sha256:
        raise ValueError(f'{path} was fitted for another model bundle')
    supervised = set(update['supervision']['fit_targets']) | set(update['supervision']['validation_targets'])
    shared = supervised & set(map(str, target_ids))
    if shared:
        raise ValueError(f'{len(shared)} requested target(s) supplied the real-GNN labels of {path}; results on them would '
                         'not measure the updated model on unseen targets')
    gate = float(update['gate'])
    info = dict(path=str(Path(path).resolve()), sha256=sha256(path), gate=gate, state='omega',
                supervision_targets=len(supervised))
    if gate:
        omega = copy.deepcopy(models.world_model['omega'])
        with torch.no_grad():
            for name, parameter in omega.named_parameters():
                parameter.add_(gate * update['delta'][name].to(parameter.device, parameter.dtype))
        models.world_model['omega'] = omega.eval().requires_grad_(False)
    return info
