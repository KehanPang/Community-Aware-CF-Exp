"""Training objective of the Set Transformer (selector F_Theta) on TRAIN witnesses."""
import torch
import torch.nn.functional as F


def unique_edge_coverage(p_community_pad, community_mask, incidence):
    p = p_community_pad.clamp(0.0, 1.0 - 1e-06)
    log_survival = torch.log1p(-p).masked_fill(~community_mask, 0.0)
    batch = int(p_community_pad.shape[0])
    nmax = int(incidence['n_edges_max'])
    flat_log = torch.zeros(batch * nmax, dtype=log_survival.dtype, device=log_survival.device)
    community_index = incidence['community_index'].long()
    edge_index = incidence['edge_index'].long()
    if community_index.numel():
        flat_log.index_add_(0, edge_index, log_survival.reshape(-1)[community_index])
    edge_log_survival = flat_log.view(batch, nmax)
    return {'p_edge': 1.0 - torch.exp(edge_log_survival), 'edge_log_survival': edge_log_survival}


def alpha_slot_diversity(alpha_attention):
    losses = []
    for b in range(alpha_attention.shape[0]):
        alpha = alpha_attention[b]
        if alpha.shape[0] < 2:
            continue
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-06)
        gram = alpha @ alpha.transpose(0, 1)
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        losses.append(((gram - eye) ** 2).mean())
    if not losses:
        return torch.zeros((), device=alpha_attention.device)
    return torch.stack(losses).mean()


def selector_loss(output, batch, weights):
    """Witness coverage, domain size, community/membership supervision, membership entropy and slot diversity."""
    p_community_pad, q_membership = output['p_community'], output['q_membership']
    alpha_attention, macro_logits = output['alpha_attention'], output['macro_logits']
    community_mask = batch['community_mask']
    p_edge = unique_edge_coverage(p_community_pad, community_mask, batch['incidence'])['p_edge']
    witness_mask = batch['witness_mask'].to(p_edge.dtype)
    edge_mask = batch['edge_mask'].to(p_edge.dtype)
    l_cover = -(torch.log(p_edge + 1e-08) * witness_mask).sum() / witness_mask.sum().clamp_min(1.0)
    l_size = (p_edge * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
    l_domain = float(weights['lambda_cover']) * l_cover + float(weights['lambda_size']) * l_size
    community_pos = batch['community_pos'].to(macro_logits.dtype)
    community_bce = F.binary_cross_entropy_with_logits(macro_logits[community_mask], community_pos[community_mask])
    slot_target = community_pos.unsqueeze(1).expand_as(q_membership)
    slot_mask = community_mask.unsqueeze(1).expand_as(q_membership)
    membership_bce = F.binary_cross_entropy(q_membership[slot_mask], slot_target[slot_mask])
    q = q_membership.clamp(1e-06, 1.0 - 1e-06)
    q_entropy = -(q * torch.log(q) + (1.0 - q) * torch.log(1.0 - q)).mean()
    alpha_diversity = alpha_slot_diversity(alpha_attention)
    return (l_domain + float(weights['lambda_community']) * community_bce
            + float(weights['lambda_membership']) * membership_bce
            + float(weights['lambda_membership_entropy']) * q_entropy
            + float(weights['lambda_slot_diversity']) * alpha_diversity)
