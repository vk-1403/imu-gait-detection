"""Four-state finite-state-machine baseline (STANCE, PRE_SWING, SWING, PRE_STANCE)."""
from dataclasses import asdict
import numpy as np

from .preprocessing import Params, precondition, adaptive_thresholds


STANCE, PRE_SWING, SWING, PRE_STANCE = 0, 1, 2, 3


def _run_fsm(g, thr_high, thr_low, p):
    rising_high = np.flatnonzero((g[1:] > thr_high[1:]) & (g[:-1] <= thr_high[:-1])) + 1
    falling_low = np.flatnonzero((g[1:] < thr_low[1:]) & (g[:-1] >= thr_low[:-1])) + 1
    local_max = np.flatnonzero((g[1:-1] > g[:-2]) & (g[1:-1] > g[2:])) + 1
    local_min = np.flatnonzero((g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])) + 1
    cand_idx = np.concatenate([rising_high, local_max, falling_low, local_min])
    cand_type = np.concatenate([
        np.zeros(len(rising_high), dtype=np.int8),
        np.full(len(local_max), 1, dtype=np.int8),
        np.full(len(falling_low), 2, dtype=np.int8),
        np.full(len(local_min), 3, dtype=np.int8),
    ])
    order = np.argsort(cand_idx, kind="mergesort")
    cand_idx, cand_type = cand_idx[order], cand_type[order]

    min_stance = int(p.min_stance_s * p.fs)
    min_swing = int(p.min_swing_s * p.fs)
    min_cycle = int(p.min_cycle_s * p.fs)

    state = STANCE
    last_change = 0
    last_to = -10**9
    hs_idx, to_idx = [], []
    for k in range(len(cand_idx)):
        i, kind = int(cand_idx[k]), int(cand_type[k])
        if state == STANCE and kind == 0:
            if (i - last_change) >= min_stance and (i - last_to) >= min_cycle:
                j = i
                while j > 0 and g[j - 1] > thr_high[j - 1] * 0.4 and g[j - 1] < g[j]:
                    j -= 1
                to_idx.append(j)
                last_to = j
                state = PRE_SWING
                last_change = i
        elif state == PRE_SWING and kind == 1:
            if g[i] > thr_high[i]:
                state = SWING
                last_change = i
        elif state == SWING and kind == 2:
            if (i - last_change) >= min_swing:
                state = PRE_STANCE
                last_change = i
        elif state == PRE_STANCE and kind == 3:
            if g[i] < thr_low[i] * 0.6:
                hs_idx.append(i)
                state = STANCE
                last_change = i
    return np.array(hs_idx, dtype=int), np.array(to_idx, dtype=int)


def detect_events_fsm(gyro_z, fs=200.0, params=None):
    p = params or Params(fs=fs)
    g = precondition(gyro_z, p)
    thr_high, thr_low = adaptive_thresholds(g, p)
    hs_idx, to_idx = _run_fsm(g, thr_high, thr_low, p)
    return {
        "hs_idx": hs_idx,
        "to_idx": to_idx,
        "gyro_filt": g,
        "thr_high": thr_high,
        "thr_low": thr_low,
        "params": asdict(p),
    }
