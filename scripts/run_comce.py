import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from comce.pipeline.config import ResourceConfigurationError, load_template, resolve
from comce.pipeline.run import run_population


def target_ids(args, config):
    if args.target_id:
        return list(map(str, args.target_id)), 'command line'
    if args.population:
        path = Path(args.population).resolve()
        ids = path.read_text().split()
    else:
        path = config.paths.population(args.population_name)
        ids = path.read_text().split()
    if args.limit is not None:
        ids = ids[:args.limit]
    return ids, str(path)


def main():
    parser = argparse.ArgumentParser(description='ComCE counterfactual explanations for GNN vertex classification')
    parser.add_argument('--config', type=Path, default=ROOT / 'yaml.template')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', required=True)
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument('--target-id', action='append', help='target vertex id; repeat for several targets')
    targets.add_argument('--population', type=Path, help='file with whitespace-separated target ids')
    targets.add_argument('--population-name', help='a population declared under datasets.<dataset>.resources.populations')
    parser.add_argument('--limit', type=int, help='use the first N ids of the population')
    parser.add_argument('--split', default='val', help='split the targets belong to (train, val or test)')
    parser.add_argument('--output-dir', type=Path,
                        help='result directory outside the release; defaults to <outputs.root>/<dataset>-<backbone>-<time>')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='override a stage2/stage3/task/execution parameter, e.g. stage2.relevance_threshold=0.2')
    parser.add_argument('--role', default='adhoc')
    parser.add_argument('--world-model-update', type=Path,
                        help='opt-in omega update from scripts/update_world_model.py; off unless given')
    args = parser.parse_args()
    config = resolve(load_template(args.config), args.dataset, args.backbone, args.set)
    paths = config.paths
    if args.population_name and args.population_name not in (config.dataset_config['resources'].get('populations') or {}):
        parser.error(f'unknown population {args.population_name!r}; configured: '
                     f'{sorted(config.dataset_config["resources"].get("populations") or {})}')
    paths.require(paths.inference_keys() + ([] if args.output_dir else ['outputs.root'])
                  + ([paths.key(f'resources.populations.{args.population_name}')] if args.population_name else []))
    ids, source = target_ids(args, config)
    output = args.output_dir or paths.resolve('outputs.root') / f'{args.dataset}-{args.backbone}-{time.strftime("%Y%m%dT%H%M%S")}'
    complete = run_population(config, ids=ids, split=args.split, result_dir=output, role=args.role,
                              population_source=source,
                              selection_rule=f'first {args.limit}' if args.limit is not None else 'explicit target list',
                              world_model_update=args.world_model_update)
    for line in (output / 'REQUESTS.jsonl').read_text().splitlines():
        row = json.loads(line)
        print(json.dumps({k: row.get(k) for k in ('center_id', 'success', 'returned_edge_keys', 'actual_blackbox_forwards',
                                                  'failure_reason')}))
    print(json.dumps({'output_dir': str(output), **complete}, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except ResourceConfigurationError as error:
        raise SystemExit(f'ResourceConfigurationError: {error}')
