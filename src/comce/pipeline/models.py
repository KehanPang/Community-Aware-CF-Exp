"""One ComCE model bundle per (dataset, backbone): F_Theta, the WorldModel states omega/omega_bar, w_r/b_r and pi_xi."""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from comce.stage1.signatures import SignatureBank
from comce.stage2.set_transformer import SetTransformer
from comce.stage3.policy import Policy
from comce.stage3.ranking import RankingResidual
from comce.stage3.world_model import STATES, world_model_states

BUNDLE_FORMAT = 'comce-model-bundle'


@dataclass
class ComCEModels:
    set_transformer: SetTransformer
    world_model: nn.ModuleDict
    ranking_residual: RankingResidual
    policy: Policy
    signature_bank: SignatureBank
    metadata: dict
    path: Path


def bundle_payload(*, dataset, backbone, set_transformer, world_model, ranking_residual, policy, architecture,
                   input_schema, training):
    return dict(format=BUNDLE_FORMAT, dataset=dataset, backbone=backbone, architecture=architecture, input_schema=input_schema,
                set_transformer={k: v.detach().cpu() for k, v in set_transformer.state_dict().items()},
                world_model={state: {k: v.detach().cpu() for k, v in world_model[state].state_dict().items()} for state in STATES},
                ranking_residual={k: v.detach().cpu() for k, v in ranking_residual.state_dict().items()},
                policy={k: v.detach().cpu() for k, v in policy.state_dict().items()},
                training=training)


def load_models(path, *, bank_path, device):
    """Strict load of one bundle and its signature bank."""
    path, bank_path = Path(path), Path(bank_path)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') != BUNDLE_FORMAT:
        raise ValueError(f'{path} is not a ComCE model bundle')
    architecture = payload['architecture']
    set_transformer = SetTransformer(**architecture['set_transformer'])
    set_transformer.load_state_dict(payload['set_transformer'], strict=True)
    world_model = world_model_states(**architecture['world_model'])
    for state in STATES:
        world_model[state].load_state_dict(payload['world_model'][state], strict=True)
    ranking_residual = RankingResidual(**architecture['ranking_residual'])
    ranking_residual.load_state_dict(payload['ranking_residual'], strict=True)
    policy = Policy(**architecture['policy'])
    policy.load_state_dict(payload['policy'], strict=True)
    bank = SignatureBank.load(bank_path)
    modules = [set_transformer, world_model, ranking_residual, policy]
    for module in modules:
        module.to(device).eval().requires_grad_(False)
    metadata = {k: v for k, v in payload.items() if k not in ('set_transformer', 'world_model', 'ranking_residual', 'policy')}
    return ComCEModels(set_transformer, world_model, ranking_residual, policy, bank, metadata, path)
