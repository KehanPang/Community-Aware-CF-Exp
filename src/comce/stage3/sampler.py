from __future__ import annotations

import math

import numpy as np

LOG_HALF = math.log(0.5)
LN2 = math.log(2.0)
TWO_M53 = 2.0 ** -53


class BitSource:

    def __init__(self, seed: int):
        self.generator = np.random.Generator(np.random.PCG64(int(seed)))
        self.raw = self.generator.bit_generator.random_raw
        self.words = 0

    def zero_run(self, count: int) -> bool:
        remaining = int(count)
        while remaining > 0:
            word = int(self.raw())
            self.words += 1
            take = min(64, remaining)
            if (word >> (64 - take)) != 0:
                return False
            remaining -= take
        return True

    def uniform53(self) -> float:
        self.words += 1
        return (int(self.raw()) >> 11) * TWO_M53


def _split(log_x: float) -> tuple[int, float]:
    log2x = log_x / LN2
    e = math.floor(log2x) + 1
    m = 2.0 ** (log2x - e)
    if m >= 1.0:
        e, m = e + 1, m / 2.0
    if m < 0.5:
        e, m = e - 1, m * 2.0
    return e, m


def exact_bernoulli(log_x: float, bits: BitSource) -> bool:
    if log_x >= 0.0:
        return True
    e, m = _split(log_x)
    if e > 0:
        raise ValueError("probability above one")
    return bits.zero_run(-e) and bits.uniform53() < m


def _prepare(log_w: np.ndarray):
    log_w = np.asarray(log_w, dtype=np.float64)
    if log_w.ndim != 1 or not len(log_w) or not np.isfinite(log_w).all():
        raise ValueError("finite one-dimensional log-weights required")
    order = np.lexsort((np.arange(len(log_w)), -log_w))
    lw = log_w[order]


    relative = np.zeros(len(lw), dtype=np.float64)
    log_q = np.zeros(len(lw), dtype=np.float64)
    log_r = np.full(len(lw), -np.inf, dtype=np.float64)
    for j in range(len(lw) - 2, -1, -1):
        tail = (lw[j + 1] - lw[j]) + relative[j + 1]
        if not math.isfinite(float(tail)):
            raise ValueError("log-weight span exceeds supported float64 arithmetic")
        relative[j] = np.logaddexp(0., tail)
        log_q[j] = -relative[j]
        log_r[j] = -np.logaddexp(0., -tail)
    return order, log_q, log_r


def sample(log_w: np.ndarray, bits: BitSource) -> int:
    order, log_q, log_r = _prepare(log_w)
    last = len(order) - 1
    for j in range(last):
        if log_q[j] <= LOG_HALF:
            accept = exact_bernoulli(float(log_q[j]), bits)
        else:
            accept = not exact_bernoulli(float(log_r[j]), bits)
        if accept:
            return int(order[j])
    return int(order[last])
