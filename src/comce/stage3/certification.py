from dataclasses import dataclass
import math

from comce.stage3.candidates import jaccard_greedy
from comce.stage3.rollout import canonical


@dataclass
class QueryLedger:
    q_cert: int = 9
    q_clean: int = 2
    q0: int = 1
    certification: int = 0
    cleanup: int = 0

    def charge(self, phase):
        limit = self.q_cert if phase == "certification" else self.q_clean if phase == "cleanup" else None
        if limit is None or getattr(self, phase) >= limit:
            raise RuntimeError("independent query budget exhausted")
        setattr(self, phase, getattr(self, phase) + 1)

    def payload(self):
        return {"q0": self.q0, "certification": self.certification, "cleanup": self.cleanup,
                "physical_total": self.q0 + self.certification + self.cleanup,
                "Q_cert": self.q_cert, "Q_clean": self.q_clean}


def certify_candidates(candidates, scores, oracle, singleton_scores, *,
                       q_cert=9, q_clean=2, kappa=2, max_edges=10, edit_cost=None, dedup_threshold=0.9):
    if min(q_cert, q_clean, kappa) < 0:
        raise ValueError("negative runtime budget")
    edit_cost = len if edit_cost is None else edit_cost
    def cost(key):
        value = float(edit_cost(key))
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid candidate edit cost')
        return value
    best_scores = {}
    for candidate, score in zip(candidates, scores, strict=True):
        key = canonical(candidate)
        if not key or len(key) > max_edges or len(key) != len(candidate) or not math.isfinite(float(score)):
            raise ValueError("illegal exact candidate")
        best_scores[key] = max(best_scores.get(key, -math.inf), float(score))
    costs = {key: cost(key) for key in best_scores}
    order = jaccard_greedy(sorted(best_scores, key=lambda k: (-best_scores[k], costs[k], len(k), k)), dedup_threshold)
    ledger = QueryLedger(q_cert, q_clean)
    attempts, prunes, certified = [], [], []
    first_hit_position = None
    for position, key in enumerate(order, 1):
        if ledger.certification >= q_cert:
            break
        if first_hit_position is not None and position > first_hit_position + kappa:
            break
        ledger.charge("certification")
        value = oracle(key, "certification")
        flip = bool(value["flip"])
        attempts.append({"static_rank": position, "keys": list(key), "flip": flip,
                         "source": "WM", "score": best_scores[key]})
        if flip:
            certified.append(key)
            if first_hit_position is None:
                first_hit_position = position
    chosen = min(certified, key=lambda k: (costs[k], -best_scores[k], len(k), k)) if certified else None
    before_cleanup = chosen
    if chosen is not None and len(chosen) > 1 and q_clean:
        singletons = [(edge,) for edge in chosen]
        singleton_raw = list(map(float, singleton_scores(singletons)))
        if len(singleton_raw) != len(singletons) or not all(map(math.isfinite, singleton_raw)):
            raise ValueError("invalid singleton final scores")
        restore_order = sorted(zip(chosen, singleton_raw), key=lambda pair: (pair[1], pair[0]))
        for edge, score in restore_order:
            if ledger.cleanup >= q_clean or len(chosen) <= 1:
                break
            proposed = tuple(k for k in chosen if k != edge)
            ledger.charge("cleanup")
            value = oracle(proposed, "cleanup")
            flip = bool(value["flip"])
            prunes.append({"keys": list(proposed), "restored_edge": edge,
                           "singleton_score": score, "flip": flip,
                           "before": list(chosen)})
            if flip and cost(proposed) <= cost(chosen):
                chosen = proposed
    return {"success": chosen is not None, "returned_edge_keys": list(chosen or ()),
            "returned_size": len(chosen) if chosen is not None else None,
            "weighted_edit_cost": cost(chosen) if chosen is not None else None,
            "candidate_queries": ledger.certification, "cleanup_queries": ledger.cleanup,
            "total_queries": ledger.payload()["physical_total"], "query_ledger": ledger.payload(),
            "attempts": attempts, "cleanup_attempts": prunes,
            "first_hit_rank": first_hit_position, "certified_sets": [list(k) for k in certified],
            "best_before_cleanup": list(before_cleanup or ()), "schedule_count": len(order),
            "pruning_guarantee": "BUDGET_BOUNDED_SINGLE_PASS", "strict_local_minimality": False}
