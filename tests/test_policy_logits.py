import numpy as np
import torch

from comce.stage3.policy import Policy, PolicyLogits


def fixture(n=5, target=0):
    torch.manual_seed(15)
    policy = Policy(edge_in_dim=4, hidden=8, center_state_dim=3, ctx_extra=4, budget=10).eval()
    inputs = dict(edge_keys=[f'e{i}' for i in range(n)],
                  edge_inputs=np.arange(n * 4, dtype=np.float32).reshape(n, 4) / 17,
                  center_state=np.asarray([target, 0.3, -0.5], np.float32),
                  community_score_ctx=np.asarray([n / 10, 0.5], np.float32))
    return policy, inputs


@torch.no_grad()
def native(policy, inputs, selected):
    mask = torch.tensor([k in selected for k in inputs['edge_keys']])
    return policy(torch.as_tensor(inputs['edge_inputs']), torch.as_tensor(inputs['center_state']),
                  mask, torch.as_tensor(inputs['community_score_ctx'])).double().numpy()


def test_cached_logits_keep_native_bits_without_consuming_rng():
    policy, inputs = fixture()
    cache = PolicyLogits(policy, inputs)
    states = [(), ('e1',), ('e1', 'e3'), ('e3', 'e1'), (), ('e1',)]
    rng = torch.get_rng_state().clone()
    for state in states:
        assert cache(state).tobytes() == native(policy, inputs, state).tobytes()
        assert torch.equal(torch.get_rng_state(), rng)
    assert cache.policy_calls == 6 and cache.policy_cache_hits == 3


def test_logits_return_values_cannot_mutate_cached_output():
    policy, inputs = fixture()
    cache = PolicyLogits(policy, inputs)
    expected = native(policy, inputs, ('e1',))
    cache(('e1',))[:] = 999
    assert cache(('e1',)).tobytes() == expected.tobytes()
    cache(('e1',))[:] = -999
    assert cache(('e1',)).tobytes() == expected.tobytes()


def test_another_target_or_an_expanded_context_never_reuses_cached_logits():
    policy, inputs = fixture()
    first = PolicyLogits(policy, inputs)
    first(('e1',))
    for n, target in [(5, 2), (7, 0), (5, 0)]:
        policy, current = fixture(n, target)
        logits = PolicyLogits(policy, current)
        assert logits._logits_cache is not first._logits_cache
        assert logits(('e1',)).tobytes() == native(policy, current, ('e1',)).tobytes()
        assert logits.policy_calls == 1 and logits.policy_cache_hits == 0
