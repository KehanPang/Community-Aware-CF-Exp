import torch

from comce.stage3.candidates import pack_sets
from comce.stage3.policy import PolicyLogits, prepare_policy_inputs
from comce.stage3.ranking import CandidateRanker
from comce.stage3.rollout import CommunityExpansion, SearchContext, request_seed, run_rollouts
from comce.stage3.scoring import RequestScorer
from comce.stage3.set_context import SetContext


@torch.no_grad()
def build_search_context(record, domain, models):
    """Policy logits, z_omega and S for one search domain."""
    inputs = prepare_policy_inputs(record, domain)
    inputs.pop('witness_keys', None)
    device = next(models.world_model.parameters()).device
    sets = SetContext(record, domain, device)
    signatures = getattr(record, 'signature_context', None)
    if signatures is None:
        raise ValueError('Live Region record must supply SignatureContext')
    scorer = RequestScorer(models.world_model, models.ranking_residual, sets, signatures)
    if list(inputs['edge_keys']) != sets.keys:
        raise RuntimeError('Policy/WorldModel canonical index mismatch')
    policy_logits = PolicyLogits(models.policy, inputs)
    ranker = CandidateRanker(scorer)
    contract = dict(world_model=scorer.contract(), ranking=ranker.contract(), cleanup_score='S')
    return SearchContext(sets.keys, policy_logits, scorer.world_model_scores, ranker, contract)


@torch.no_grad()
def generate_candidates(record, selector_probabilities, domain, models, settings, *, random_seed, center_id):
    """N_roll guided rollouts from the policy; every visited prefix becomes a candidate scored by S."""
    expansion = CommunityExpansion(record, selector_probabilities, domain, settings.expansion_budget,
                                   settings.expansion_communities_per_trigger)
    result = run_rollouts(domain, expansion, lambda current: build_search_context(record, current, models),
                          seed=request_seed(random_seed, center_id), settings=settings)
    pool = pack_sets(result['sets'], result['context'].keys, paths=result['paths'], scores=result['scores'],
                     stats=result['stats'])
    return pool, result
