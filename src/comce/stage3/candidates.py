from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json

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

    def order(self):
        idx = np.arange(len(self))
        keys = [self.canonical[idx, i] for i in range(9, -1, -1)]
        keys.extend([self.sizes[idx], -self.raw[idx]])
        return idx[np.lexsort(tuple(keys))]


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


def _exact_key(pool, index):
    size = int(pool.sizes[int(index)])
    return tuple(int(x) for x in pool.canonical[int(index), :size])


def candidate_order(pool, edit_cost=None, dedup_threshold=1.0):
    """Sets by descending S, then edit cost, size and canonical key, after exact and Jaccard-greedy de-duplication."""
    sizes = np.asarray(pool.sizes)
    raw = np.asarray(pool.raw)
    if sizes.ndim != 1 or raw.ndim != 1 or len(sizes) != len(raw):
        raise ValueError('invalid packed pool shapes')
    if not np.issubdtype(sizes.dtype, np.integer) or np.any((sizes < 1) | (sizes > 10)):
        raise ValueError('packed candidate sizes must be integer values in [1,10]')
    if not np.isfinite(raw).all():
        raise ValueError('packed scores must be finite')
    for index in range(len(pool)):
        units = list(map(int, pool.canonical[index, :int(sizes[index])]))
        if len(units) != len(set(units)):
            raise ValueError('candidate contains duplicate canonical units')
        if any(unit < 0 or unit >= len(pool.keys) for unit in units):
            raise ValueError('candidate contains an out-of-domain unit')
    representatives = {}
    for index in range(len(pool)):
        key = _exact_key(pool, index)
        current = representatives.get(key)
        if current is None or float(pool.raw[index]) > float(pool.raw[current]):
            representatives[key] = index
    unique = set(representatives.values())
    result = [int(index) for index in pool.order() if int(index) in unique]
    if edit_cost is not None:
        result = sorted(result, key=lambda i: (-float(pool.raw[i]), edit_cost(pool.keys[j] for j in _exact_key(pool, i)),
                                               int(pool.sizes[i]), _exact_key(pool, i)))
    if len({_exact_key(pool, index) for index in result}) != len(result):
        raise RuntimeError('duplicate exact set in candidate schedule')
    survivors = set(jaccard_greedy([_exact_key(pool, index) for index in result], dedup_threshold))
    return [index for index in result if _exact_key(pool, index) in survivors]


def candidate_evidence(pool, schedule):
    exact = [_exact_key(pool, index) for index in range(len(pool))]
    set_hash = hashlib.sha256()
    for key in sorted(set(exact)):
        set_hash.update(json.dumps(key, separators=(',', ':')).encode() + b'\n')
    best = {}
    for index, key in enumerate(exact):
        score = float(pool.raw[index])
        if key not in best or score > best[key]:
            best[key] = score
    score_hash = hashlib.sha256()
    for key in sorted(best):
        score_hash.update(json.dumps([key, float(best[key])], separators=(',', ':'), allow_nan=False).encode() + b'\n')
    schedule_hash = hashlib.sha256()
    for index in schedule:
        schedule_hash.update(json.dumps(_exact_key(pool, index), separators=(',', ':')).encode() + b'\n')
    return {'candidate_set_sha256': set_hash.hexdigest(), 'candidate_score_sha256': score_hash.hexdigest(),
            'candidate_schedule_sha256': schedule_hash.hexdigest()}
