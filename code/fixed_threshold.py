"""Fixed-threshold comparison rule: thresholds at 0.6 x P95 and 0.6 x P5 of the filtered signal.
TO is the minimum in the 0.35 s before each rising crossing of the upper threshold and HS the minimum in the
0.45 s after the following falling crossing, both required to lie below the lower threshold."""
import numpy as np
from scipy.signal import butter, filtfilt


def detect(gyro_z, fs=200.0, scale_high=0.6, scale_low=0.6, fc=15.0, order=4, min_dist_s=0.45, to_back_s=0.35, hs_fwd_s=0.45):
    b, a = butter(order, fc / (0.5 * fs), btype='low'); g = filtfilt(b, a, gyro_z); g = g - np.median(g)
    thr_high = scale_high * float(np.percentile(g, 95)); thr_low = scale_low * float(np.percentile(g, 5))
    rising = np.flatnonzero((g[1:] >= thr_high) & (g[:-1] < thr_high)) + 1
    falling = np.flatnonzero((g[1:] < thr_high) & (g[:-1] >= thr_high)) + 1
    n = len(g); min_dist = int(min_dist_s * fs); to_back = int(to_back_s * fs); hs_fwd = int(hs_fwd_s * fs)
    to_idx, hs_idx = [], []; last = -10 ** 9
    for r in rising:
        if r - last < min_dist: continue
        last = r
        lo = max(0, r - to_back)
        if lo < r:
            j = int(np.argmin(g[lo:r])) + lo
            if g[j] < thr_low: to_idx.append(j)
        f_after = falling[falling > r]
        if len(f_after) == 0: continue
        f = int(f_after[0]); end = min(n, f + hs_fwd)
        if f < end:
            j = int(np.argmin(g[f:end])) + f
            if g[j] < thr_low: hs_idx.append(j)
    return np.array(hs_idx, dtype=int), np.array(to_idx, dtype=int)
