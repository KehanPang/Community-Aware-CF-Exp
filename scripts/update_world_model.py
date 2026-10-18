import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch

from comce.pipeline.config import ResourceConfigurationError, load_template, resolve
from comce.pipeline.features import outside_release
from comce.pipeline.io import atomic_json
from comce.pipeline.models import load_models
from comce.training.world_model_update import (UPDATE_FORMAT, _tensors, certification_labels, fit_update, rebuild_examples,
                                               split_by_target)

# Largest |S_rebuilt - S_logged| accepted as the same ranking state (CPU/GPU float32 reductions differ in the last bits).
SCORE_TOLERANCE = 1e-3


def main():
    parser = argparse.ArgumentParser(description='Fit an optional update omega + gate * delta of the WorldModel state omega '
                                                 'from real-GNN certification labels of TRAIN targets')
    parser.add_argument('--config', type=Path, default=ROOT / 'yaml.template')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--certification-log', type=Path, action='append', required=True,
                        help='run directory of scripts/run_comce.py on TRAIN targets (--split train); repeat for several')
    parser.add_argument('--output', type=Path, required=True, help='update file outside the release; results go to <output>.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    template = load_template(args.config)
    config = resolve(template, args.dataset, args.backbone)
    config.paths.require(config.paths.inference_keys() + config.paths.population_keys())
    output = outside_release(args.output, 'the WorldModel update')
    report = output.with_suffix('.json')
    if output.exists() or report.exists():
        parser.error(f'{output} or {report} exists; write the update to a new location')
    settings = template['training']['world_model_update']
    models = load_models(config.model_path, bank_path=config.signature_bank_path, device=args.device)
    labeled, logs, isolation = certification_labels(config, args.certification_log)
    if not labeled:
        raise SystemExit('the certification logs hold no real-GNN labels of TRAIN targets')
    items, score_gap = rebuild_examples(config, models, labeled, device=args.device)
    if score_gap > SCORE_TOLERANCE:
        raise SystemExit(f'rebuilt ranking scores differ from the certification log by {score_gap:.3g}; the log was not '
                         f'produced by this bundle and configuration')
    fit_items, validation_items = split_by_target(items, settings['validation_fraction'], settings['seed'])
    edge_in = models.world_model['omega'].edge_in
    fit = _tensors(fit_items, edge_in, args.device)
    if fit is None:
        raise SystemExit('no labels are left for fitting after the validation split')
    delta, gate, result = fit_update(models.world_model['omega'], fit, _tensors(validation_items, edge_in, args.device),
                                     settings, device=args.device)
    supervision = dict(fit_targets=sorted(i['center'] for i in fit_items),
                       validation_targets=sorted(i['center'] for i in validation_items))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(format=UPDATE_FORMAT, state='omega', gate=gate, delta=delta, dataset=config.dataset,
                    backbone=config.backbone, settings=dict(settings), supervision=supervision, certification_logs=logs), output)
    summary = dict(format=UPDATE_FORMAT, update=str(output), dataset=config.dataset, backbone=config.backbone, state='omega', gate=gate,
                   accepted=bool(gate), delta_norm=float(sum((d ** 2).sum() for d in delta.values()) ** 0.5),
                   settings=dict(settings), certification_logs=logs, train_population=isolation,
                   label_source='journaled real-GNN certification and cleanup forwards of TRAIN targets',
                   rebuilt_score_max_abs_gap=score_gap, labels=dict(targets=len(items), fit_targets=len(fit_items),
                                                                    validation_targets=len(validation_items)),
                   supervision=supervision, **{k: v for k, v in result.items() if k != 'history'},
                   history=result['history'][::max(1, len(result['history']) // 20)] + result['history'][-1:])
    atomic_json(report, summary)
    print(json.dumps({k: summary[k] for k in ('gate', 'accepted', 'labels', 'validation', 'decision')}, indent=1))
    print(output, report)


if __name__ == '__main__':
    try:
        main()
    except ResourceConfigurationError as error:
        raise SystemExit(f'ResourceConfigurationError: {error}')
