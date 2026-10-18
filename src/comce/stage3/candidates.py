from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def jaccard_greedy(ordered, threshold):
    """RankDedup: walk candidates in decreasing S and skip one whose Jaccard similarity with any retained candidate
    exceeds tau_dup (strict >). Returns the retained candidates in order."""
    tau = Fraction(str(threshold))
    if not 0 <= tau <= 1:
        raise ValueError('dedup_threshold must be in [0,1]')
    retained, by_size = [], defaultdict(list)
    for candidate in ordered:
        members = frozenset(candidate)
        size = len(members)
        duplicate = False
        for other_size, group in by_size.items():
            # J(A,B) <= min(|A|,|B|) / max(|A|,|B|), so a whole size class can be ruled out at once.
            if Fraction(min(size, other_size), max(size, other_size, 1)) <= tau:
                continue
            for other in group:
                shared = len(members & other)
                if shared * tau.denominator > tau.numerator * (size + other_size - shared):
                    duplicate = True
                    break
            if duplicate:
                break
        if not duplicate:
            retained.append(candidate)
            by_size[size].append(members)
    return retained


@dataclass
class PackedPool:
    canonical: np.ndarray
    paths: np.ndarray
    sizes: np.ndarray
    raw: np.ndarray
    priorities: np.ndarray
    keys: list[str]
    stats: dict

    def __len__(self):
        return len(self.raw)

    def sets(self):
        return [r[:int(n)].tolist() for r, n in zip(self.canonical, self.sizes, strict=True)]


def pack_sets(sets, keys, *, paths=None, scores=None, stats=None):
    index = {key: i for i, key in enumerate(keys)}
    canon = np.full((len(sets), 10), -1, np.int32)
    ordered = np.full_like(canon, -1)
    sizes = np.asarray([len(s) for s in sets], np.int32)
    for i, key in enumerate(sets):
        if not 0 < len(key) <= 10 or len(key) != len(set(key)):
            raise ValueError('invalid deletion set')
        canon[i, :len(key)] = sorted(index[k] for k in key)
        ordered[i, :len(key)] = [index[k] for k in (paths[i] if paths is not None else key)]
    raw = np.zeros(len(sets), np.float64) if scores is None else np.asarray(scores, np.float64)
    return PackedPool(canon, ordered, sizes, raw, raw.copy(), list(keys), stats or {})

