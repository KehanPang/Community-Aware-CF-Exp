import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from comce.pipeline.config import ResourceConfigurationError, load_template, resolve
from comce.pipeline.io import PROJECT_ROOT
from comce.training.trainer import train_tuple


def main():
    parser = argparse.ArgumentParser(description='Train one ComCE model bundle on TRAIN targets only')
    parser.add_argument('--config', type=Path, default=ROOT / 'yaml.template')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--signature-bank', type=Path, help='defaults to datasets.<dataset>.resources.signature_bank')
    parser.add_argument('--models-root', required=True, type=Path, help='output directory of the new bundle')
    parser.add_argument('--work', required=True, type=Path, help='scratch directory for record caches')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    for path in (args.models_root, args.work):
        if path.resolve().is_relative_to(PROJECT_ROOT):
            parser.error('training outputs must be outside the release')
    template = load_template(args.config)
    config = resolve(template, args.dataset, args.backbone)
    config.paths.require(config.paths.training_keys() + ([] if args.signature_bank else [config.paths.key('resources.signature_bank')]))
    bank = args.signature_bank or config.signature_bank_path
    import torch
    torch.set_num_threads(args.threads)
    print(train_tuple(config, template, device=args.device, work=args.work, signature_bank=bank, models_root=args.models_root))


if __name__ == '__main__':
    try:
        main()
    except ResourceConfigurationError as error:
        raise SystemExit(f'ResourceConfigurationError: {error}')
