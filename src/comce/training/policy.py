"""Planner training: teacher-forced imitation of TRAIN witnesses, then REINFORCE with the WorldModel reward."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import Any, Callable

import numpy as np
import torch

from comce.stage3.policy import Policy


class RewardScorer:
    """p_omega(u, Delta) for planner rewards: z_omega of the WorldModel on each candidate set."""

    def __init__(self, scorers, chunk_size=4096):
        self.scorers = scorers
        self.chunk_size = chunk_size

    @torch.no_grad()
    def flip_probabilities(self, center_id, key_sets):
        scorer = self.scorers[str(center_id)]
        positions = [idx for idx, keys in enumerate(key_sets) if keys]
        sets = [sorted(scorer.index[key] for key in key_sets[idx]) for idx in positions]
        values = {}
        if sets:
            logits = torch.cat([scorer.state_logits('omega', *scorer.sets.pack(sets[begin:begin + self.chunk_size]))
                                for begin in range(0, len(sets), self.chunk_size)])
            values = dict(zip(positions, (float(p) for p in logits.sigmoid().cpu())))
        return [values.get(idx, 0.0) for idx in range(len(key_sets))]


def imitation_warm_start(policy: Policy, train_samples: list[dict[str, Any]], *, exp, device, epochs: int=10, log_fn: Callable[[str], None]=print) -> dict:
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(exp.rl_lr))
    rng = random.Random(int(exp.rl_seed))
    budget = int(exp.rl_max_candidate_edges)
    stop_weight = float(exp.rl_stop_aux_weight)
    loss_curve: list[float] = []
    skipped = 0
    for epoch in range(int(epochs)):
        policy.train()
        epoch_losses: list[float] = []
        for sample in train_samples:
            inputs = sample['inputs']
            target_actions = _teacher_target_actions(inputs, rng=rng, budget=budget)
            if target_actions is None:
                skipped += 1
                continue
            loss = _teacher_forced_loss(policy, inputs, target_actions, stop_weight=stop_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_curve.append(mean_loss)
        log_fn(f'[policy imitation] epoch={epoch + 1}/{epochs} mean_loss={mean_loss:.4f} centers={len(epoch_losses)}')
    return {'loss_curve': loss_curve, 'epochs': int(epochs), 'centers_skipped': skipped, 'stop_weight': float(stop_weight)}

def _rollout_trace(policy: Policy, inputs: dict[str, Any], *, budget: int, generator: torch.Generator, temperature: float, keep_graph: bool) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], int, bool]:
    device = next(policy.parameters()).device
    edge_inputs = torch.as_tensor(np.asarray(inputs['edge_inputs']), dtype=torch.float32, device=device)
    center_state = torch.as_tensor(np.asarray(inputs['center_state']), dtype=torch.float32, device=device)
    community_ctx = torch.as_tensor(np.asarray(inputs['community_score_ctx']), dtype=torch.float32, device=device)
    edge_keys = [str(key) for key in inputs['edge_keys']]
    m = len(edge_keys)
    selected_mask = torch.zeros(m, dtype=torch.bool, device=device)
    chosen: list[str] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    steps = 0
    learned_stop = False
    while True:
        if keep_graph:
            logits = policy(edge_inputs, center_state, selected_mask, community_ctx)
        else:
            with torch.no_grad():
                logits = policy(edge_inputs, center_state, selected_mask, community_ctx)
        dist = torch.distributions.Categorical(logits=logits / float(temperature))
        action = int(torch.multinomial(dist.probs.cpu(), 1, generator=generator).item())
        log_probs.append(dist.log_prob(torch.tensor(action, device=device)))
        entropies.append(dist.entropy())
        steps += 1
        if action == m:
            learned_stop = bool(bool((~selected_mask).any().item()) and len(chosen) < int(budget))
            break
        selected_mask = selected_mask.clone()
        selected_mask[action] = True
        chosen.append(edge_keys[action])
        if len(chosen) >= int(budget):
            break
    return (chosen, log_probs, entropies, steps, learned_stop)

def _teacher_target_actions(inputs: dict[str, Any], *, rng: random.Random, budget: int) -> list[int] | None:
    md_keys = [str(key) for key in inputs['edge_keys']]
    key_position = {key: idx for idx, key in enumerate(md_keys)}
    witness = [key for key in (str(k) for k in inputs.get('witness_keys', [])) if key in key_position]
    if not witness:
        return None
    rng.shuffle(witness)
    target_keys = witness[:int(budget)]
    return [key_position[key] for key in target_keys] + [len(md_keys)]

def _teacher_forced_loss(policy: Policy, inputs: dict[str, Any], target_actions: list[int], *, stop_weight: float=1.0) -> torch.Tensor:
    device = next(policy.parameters()).device
    edge_inputs = torch.as_tensor(np.asarray(inputs['edge_inputs']), dtype=torch.float32, device=device)
    center_state = torch.as_tensor(np.asarray(inputs['center_state']), dtype=torch.float32, device=device)
    community_ctx = torch.as_tensor(np.asarray(inputs['community_score_ctx']), dtype=torch.float32, device=device)
    stop_index = len(inputs['edge_keys'])
    selected_mask = torch.zeros(stop_index, dtype=torch.bool, device=device)
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    for action in target_actions:
        logits = policy(edge_inputs, center_state, selected_mask, community_ctx)
        target = torch.tensor(int(action), device=device)
        terms.append(torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0)))
        weights.append(float(stop_weight) if int(action) == stop_index else 1.0)
        if int(action) != stop_index:
            selected_mask = selected_mask.clone()
            selected_mask[int(action)] = True
    weight_sum = float(sum(weights)) or 1.0
    stacked = torch.stack(terms)
    weight_tensor = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device)
    return (stacked * weight_tensor).sum() / weight_sum

def _size_penalty(size: int, *, exp) -> float:
    horizon = max(1, int(exp.rl_max_candidate_edges))
    return float(exp.rl_lambda_size) * (int(size) / float(horizon))

def _novelty_sequence(candidate_sets: list[set[str]]) -> list[float]:
    novelties: list[float] = []
    for idx, current in enumerate(candidate_sets):
        if idx == 0:
            novelties.append(1.0)
            continue
        novelties.append(1.0 - max((jaccard(current, earlier) for earlier in candidate_sets[:idx])))
    return novelties

def jaccard(a: set, b: set) -> float:
    union = set(a) | set(b)
    if not union:
        return 1.0
    return float(len(set(a) & set(b)) / len(union))

def generate_candidates(policy: Policy, inputs: dict[str, Any], *, n_rollouts: int, budget: int, seed: int, temperature: float=1.0) -> dict:
    generator = torch.Generator().manual_seed(int(seed))
    raw: list[list[str]] = []
    empty_count = 0
    stop_types: list[str] = []
    stop_timesteps: list[int] = []
    for _ in range(int(n_rollouts)):
        result = rollout(policy, inputs, budget=budget, generator=generator, temperature=temperature)
        raw.append(sorted(result.edge_keys))
        stop_types.append(result.stop_type)
        stop_timesteps.append(int(result.stop_timestep))
        if not result.edge_keys:
            empty_count += 1
    seen: set[frozenset[str]] = set()
    candidates: list[list[str]] = []
    for keys in raw:
        key = frozenset(keys)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(list(keys))
    jaccards: list[float] = []
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            jaccards.append(jaccard(set(candidates[left]), set(candidates[right])))
    size_distribution: dict[str, int] = {}
    for keys in candidates:
        size_distribution[str(len(keys))] = size_distribution.get(str(len(keys)), 0) + 1
    unique_count = len(candidates)
    n = max(1, int(n_rollouts))
    learned = sum((1 for kind in stop_types if kind == 'learned'))
    sizes = [len(keys) for keys in candidates]
    return {'candidates': candidates, 'rollout_count': int(n_rollouts), 'unique_count': unique_count, 'unique_rate': float(unique_count / n), 'mean_pairwise_jaccard': float(np.mean(jaccards)) if jaccards else 0.0, 'max_pairwise_jaccard': float(np.max(jaccards)) if jaccards else 0.0, 'size_distribution': size_distribution, 'empty_count': empty_count, 'learned_stop_rate': float(learned / n), 'forced_stop_rate': float((int(n_rollouts) - learned) / n), 'mean_stop_timestep': float(np.mean(stop_timesteps)) if stop_timesteps else 0.0, 'mean_candidate_size': float(np.mean(sizes)) if sizes else 0.0, 'size_12_fraction': float(sum((1 for s in sizes if s >= int(budget))) / max(1, unique_count))}

def reinforce_train(policy: Policy, train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]], *, scorer, exp, device, n_rollouts: int, epochs: int, log_fn: Callable[[str], None]=print, edge_cost=None) -> dict:
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(exp.rl_lr))
    budget = int(exp.rl_max_candidate_edges)
    entropy_weight = float(exp.rl_entropy_weight)
    beta_bc = float(exp.rl_bc_anchor_weight)
    stop_weight = float(exp.rl_stop_aux_weight)
    curve: list[dict[str, Any]] = []
    best_val = float('-inf')
    best_state: dict[str, Any] | None = None
    for epoch in range(int(epochs)):
        policy.train()
        epoch_rewards: list[float] = []
        epoch_pg_losses: list[float] = []
        epoch_bc_losses: list[float] = []
        epoch_total_losses: list[float] = []
        epoch_sizes: list[int] = []
        epoch_learned_stops = 0
        epoch_rollouts = 0
        bc_rng = random.Random(int(exp.rl_seed) + 5000003 * epoch)
        for center_idx, sample in enumerate(train_samples):
            inputs = sample['inputs']
            generator = torch.Generator().manual_seed(int(exp.rl_seed) + 1000003 * epoch + center_idx)
            traces = [_rollout_trace(policy, inputs, budget=budget, generator=generator, temperature=1.0, keep_graph=True) for _ in range(int(n_rollouts))]
            candidate_sets = [set(trace[0]) for trace in traces]
            scores = scorer.flip_probabilities(sample['center_id'], candidate_sets)
            novelties = _novelty_sequence(candidate_sets)
            rewards = [candidate_reward(score, len(candidate_sets[idx]), novelties[idx], exp=exp,
                       edit_cost=edge_cost, edge_keys=candidate_sets[idx]) for idx, score in enumerate(scores)]
            baseline = float(np.mean(rewards)) if rewards else 0.0
            advantages = [reward - baseline for reward in rewards]
            policy_losses: list[torch.Tensor] = []
            entropy_terms: list[torch.Tensor] = []
            for idx, (_, log_probs, entropies, _steps, _learned) in enumerate(traces):
                sum_logp = torch.stack(log_probs).sum()
                policy_losses.append(-float(advantages[idx]) * sum_logp)
                entropy_terms.append(torch.stack(entropies).mean())
            pg_loss = torch.stack(policy_losses).mean() - entropy_weight * torch.stack(entropy_terms).mean()
            loss = pg_loss
            bc_loss_value = 0.0
            if beta_bc > 0.0:
                target_actions = _teacher_target_actions(inputs, rng=bc_rng, budget=budget)
                if target_actions is not None:
                    bc_loss = _teacher_forced_loss(policy, inputs, target_actions, stop_weight=stop_weight)
                    loss = pg_loss + beta_bc * bc_loss
                    bc_loss_value = float(bc_loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_rewards.extend(rewards)
            epoch_pg_losses.append(float(pg_loss.item()))
            epoch_bc_losses.append(bc_loss_value)
            epoch_total_losses.append(float(loss.item()))
            epoch_sizes.extend((len(trace[0]) for trace in traces))
            epoch_learned_stops += sum((1 for trace in traces if trace[4]))
            epoch_rollouts += len(traces)
        train_mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        val_flip_firsts: list[float] = []
        val_rewards: list[float] = []
        val_sizes: list[float] = []
        val_learned: list[float] = []
        val_forced: list[float] = []
        policy.eval()
        for center_idx, sample in enumerate(val_samples):
            inputs = sample['inputs']
            generated = generate_candidates(policy, inputs, n_rollouts=int(n_rollouts), budget=budget, seed=int(exp.rl_seed) + 77000000 + center_idx)
            candidate_sets = [set(keys) for keys in generated['candidates']]
            if not candidate_sets:
                continue
            scores = scorer.flip_probabilities(sample['center_id'], candidate_sets)
            novelties = _novelty_sequence(candidate_sets)
            rewards = [candidate_reward(score, len(candidate_sets[idx]), novelties[idx], exp=exp,
                       edit_cost=edge_cost, edge_keys=candidate_sets[idx]) for idx, score in enumerate(scores)]
            val_flip_firsts.append(max((float(score) for score in scores)))
            val_rewards.extend(rewards)
            val_sizes.append(float(generated['mean_candidate_size']))
            val_learned.append(float(generated['learned_stop_rate']))
            val_forced.append(float(generated['forced_stop_rate']))
        val_mean_best_flip = float(np.mean(val_flip_firsts)) if val_flip_firsts else 0.0
        val_mean_reward = float(np.mean(val_rewards)) if val_rewards else 0.0
        curve.append({'epoch': epoch + 1, 'train_mean_reward': train_mean_reward, 'train_mean_pg_loss': float(np.mean(epoch_pg_losses)) if epoch_pg_losses else 0.0, 'train_mean_bc_loss': float(np.mean(epoch_bc_losses)) if epoch_bc_losses else 0.0, 'train_mean_total_loss': float(np.mean(epoch_total_losses)) if epoch_total_losses else 0.0, 'train_mean_candidate_size': float(np.mean(epoch_sizes)) if epoch_sizes else 0.0, 'train_learned_stop_rate': float(epoch_learned_stops / epoch_rollouts) if epoch_rollouts else 0.0, 'train_forced_stop_rate': float(1.0 - epoch_learned_stops / epoch_rollouts) if epoch_rollouts else 0.0, 'val_mean_top1_flip_prob': val_mean_best_flip, 'val_mean_reward': val_mean_reward, 'val_mean_candidate_size': float(np.mean(val_sizes)) if val_sizes else 0.0, 'val_learned_stop_rate': float(np.mean(val_learned)) if val_learned else 0.0, 'val_forced_stop_rate': float(np.mean(val_forced)) if val_forced else 0.0})
        log_fn(f'[policy reinforce] epoch={epoch + 1}/{epochs} train_mean_reward={train_mean_reward:.4f} val_mean_top1_flip_prob={val_mean_best_flip:.4f} L_pg={curve[-1]['train_mean_pg_loss']:.4f} L_bc={curve[-1]['train_mean_bc_loss']:.4f} L_total={curve[-1]['train_mean_total_loss']:.4f} train_cand_size={curve[-1]['train_mean_candidate_size']:.2f} train_learned_stop={curve[-1]['train_learned_stop_rate']:.3f} val_cand_size={curve[-1]['val_mean_candidate_size']:.2f} val_learned_stop={curve[-1]['val_learned_stop_rate']:.3f}')
        if val_mean_best_flip > best_val:
            best_val = val_mean_best_flip
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in policy.state_dict().items()})
    if best_state is not None:
        policy.load_state_dict(best_state)
    return {'curve': curve, 'best_val': best_val, 'best_state_dict': best_state, 'bc_anchor_weight': beta_bc}


def candidate_reward(score: float, size: int, novelty: float, *, exp, edit_cost=None, edge_keys=None) -> float:
    if edit_cost is None:
        penalty = _size_penalty(size, exp=exp)
    else:
        if edge_keys is None or len(edge_keys) != size:
            raise ValueError('Weighted reward needs the same canonical edit set and edge count')
        value = float(edit_cost(edge_keys))
        if not np.isfinite(value) or value < 0:
            raise ValueError('Invalid weighted reward cost')
        divisor = max(1, int(exp.rl_max_candidate_edges))
        penalty = float(exp.rl_lambda_size) * (value / float(divisor))
    return float(float(exp.rl_lambda_flip) * float(score) - penalty - float(exp.rl_lambda_div) * (1.0 - float(novelty)))

@dataclass
class RolloutResult:
    edge_keys: list[str]
    log_probs: list[float]
    entropies: list[float]
    stopped_at: int
    stop_type: str = 'learned'
    stop_timestep: int = 0
    candidate_size: int = 0
    reached_budget: bool = False

def rollout(policy: Policy, inputs: dict[str, Any], *, budget: int, generator: torch.Generator, temperature: float=1.0) -> RolloutResult:
    chosen, log_probs, entropies, steps, learned_stop = _rollout_trace(policy, inputs, budget=budget, generator=generator, temperature=temperature, keep_graph=False)
    return RolloutResult(edge_keys=chosen, log_probs=[float(value.item()) for value in log_probs], entropies=[float(value.item()) for value in entropies], stopped_at=steps, stop_type='learned' if learned_stop else 'forced_budget', stop_timestep=steps, candidate_size=len(chosen), reached_budget=bool(len(chosen) >= int(budget)))


def fit_policy(policy, train_samples, holdout_samples, *, scorer, exp, n_rollouts, epochs, log_fn=print, edge_cost=None):
    if type(policy) is not Policy:
        raise ValueError('The shared policy definition is required')
    if not train_samples or any(s.get('role') != 'TRAIN' for s in train_samples):
        raise ValueError('Explicit TRAIN isolation is required')
    train = {str(s['center_id']) for s in train_samples}
    holdout = {str(s['center_id']) for s in holdout_samples}
    if train & holdout or any(s.get('role') != 'HOLDOUT' for s in holdout_samples):
        raise ValueError('TRAIN/holdout selection isolation')
    return reinforce_train(policy, train_samples, holdout_samples, scorer=scorer, exp=exp, device=next(policy.parameters()).device,
                           n_rollouts=n_rollouts, epochs=epochs, log_fn=log_fn, edge_cost=edge_cost)
