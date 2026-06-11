"""Peak-anchored heel-strike and toe-off detector.

Mid-swing peaks of the sagittal-plane angular velocity are used as per-stride
anchors. For each anchor, the latest preceding negative-prominence peak is
labelled toe-off and the earliest following negative-prominence peak is
labelled heel-strike.
"""
from dataclasses import asdict
import numpy as np
from scipy.signal import find_peaks

from .preprocessing import Params, precondition, adaptive_thresholds


def detect_events(gyro_z, fs=200.0, params=None):
    p = params or Params(fs=fs)
    g = precondition(gyro_z, p)

    p75 = float(np.percentile(g, 75))
    p25 = float(np.percentile(g, 25))
    amp = max(p75 - p25, 1e-6)

    msw_idx, _ = find_peaks(
        g,
        prominence=p.msw_prominence_frac * amp,
        distance=int(p.msw_min_dist_s * p.fs),
    )
    neg_idx, _ = find_peaks(
        -g,
        prominence=p.neg_prominence_frac * amp,
        distance=int(p.neg_min_dist_s * p.fs),
    )

    hs_list, to_list = [], []
    for m in msw_idx:
        before = neg_idx[neg_idx < m]
        after = neg_idx[neg_idx > m]
        if len(before):
            to_list.append(int(before[-1]))
        if len(after):
            hs_list.append(int(after[0]))

    hs_idx = np.array(sorted(set(hs_list)), dtype=int)
    to_idx = np.array(sorted(set(to_list)), dtype=int)

    thr_high, thr_low = adaptive_thresholds(g, p)
    return {
        "hs_idx": hs_idx,
        "to_idx": to_idx,
        "msw_idx": np.asarray(msw_idx, dtype=int),
        "gyro_filt": g,
        "thr_high": thr_high,
        "thr_low": thr_low,
        "params": asdict(p),
    }
