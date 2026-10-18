import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from comce.pipeline.config import ResourceConfigurationError, load_template, resolve_stage1
from comce.pipeline.features import outside_release, stage1_plan, stage1_resource_keys
from comce.stage1.encoders import fit_signature_bank
from comce.stage1.fields import Stage1ConfigurationError, declared


def main():
    parser = argparse.ArgumentParser(description='Fit a signature bank from the Stage 1 field declaration of a dataset: '
                                                 'field encoders, shared random projection, TRAIN-only buckets')
    parser.add_argument('--config', type=Path, default=ROOT / 'yaml.template')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=Path, help='signature_bank.pt outside the release; '
                                                    'defaults to <outputs.root>/signatures/<dataset>/signature_bank.pt')
    parser.add_argument('--template-output', type=Path, help='where to write the Stage 1 template when the declaration is '
                                                             'missing; defaults to <outputs.root>/stage1/<dataset>.stage1.yaml, '
                                                             'or a new system temporary directory when outputs.root is empty')
    args = parser.parse_args()
    template = load_template(args.config)
    stage1, paths = resolve_stage1(template, args.dataset)
    # Without a declaration only the Stage 1 template is written, so no output location is needed yet.
    paths.require(stage1_resource_keys(paths, stage1) + ([] if args.output or not declared(stage1) else ['outputs.root']))
    if declared(stage1):
        output = outside_release(args.output or paths.resolve('outputs.root') / 'signatures' / args.dataset / 'signature_bank.pt',
                                 'the fitted signature bank')
        if output.exists():
            parser.error(f'{output} exists; fit into a new location')
    try:
        plan = stage1_plan(template, args.dataset, stage1, paths, template_output=args.template_output)
    except Stage1ConfigurationError as error:
        raise SystemExit(f'Stage1ConfigurationError: {error}')
    for name, entry in plan.isolation.items():
        if entry['shared_with_train']:
            print(f'note: {entry["shared_with_train"]} TRAIN vertices of the bucketizer fit are listed in evaluation population '
                  f'{name!r} (feature values only; recorded in the bank provenance)', file=sys.stderr)
    bank = fit_signature_bank(plan)
    bank.save(output)
    print(args.dataset, {t: v.shape for t, v in bank.coordinates.items()}, bank.provenance['train_nodes'],
          'feature schema', plan.fingerprint, '->', output)


if __name__ == '__main__':
    try:
        main()
    except ResourceConfigurationError as error:
        raise SystemExit(f'ResourceConfigurationError: {error}')
