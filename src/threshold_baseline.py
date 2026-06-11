"""Greene-style fixed-threshold baseline detector."""
from dataclasses import asdict
import numpy as np

from .preprocessing import Params, precondition


def detect_events_fixed(gyro_z, fs=200.0, params=None,
                        scale_high=0.6, scale_low=0.6):
    p = params or Params(fs=fs)
    g = precondition(gyro_z, p)
    thr_high = scale_high * float(np.percentile(g, 95))
    thr_low = scale_low * float(np.percentile(g, 5))

    rising_high = np.flatnonzero((g[1:] >= thr_high) & (g[:-1] < thr_high)) + 1
    falling_high = np.flatnonzero((g[1:] < thr_high) & (g[:-1] >= thr_high)) + 1

    n = len(g)
    msw_min_dist = int(p.msw_min_dist_s * p.fs)
    to_back = int(0.35 * p.fs)
    hs_fwd = int(0.45 * p.fs)

    to_idx, hs_idx = [], []
    last_event = -10**9
    for r in rising_high:
        if r - last_event < msw_min_dist:
            continue
        last_event = r
        back_lim = max(0, r - to_back)
        if back_lim < r:
            j = int(np.argmin(g[back_lim:r])) + back_lim
            if g[j] < thr_low:
                to_idx.append(j)
        fall_after = falling_high[falling_high > r]
        if len(fall_after) == 0:
            continue
        f = int(fall_after[0])
        end = min(n, f + hs_fwd)
        if f < end:
            j = int(np.argmin(g[f:end])) + f
            if g[j] < thr_low:
                hs_idx.append(j)

    return {
        "hs_idx": np.array(hs_idx, dtype=int),
        "to_idx": np.array(to_idx, dtype=int),
        "gyro_filt": g,
        "params": asdict(p),
    }
