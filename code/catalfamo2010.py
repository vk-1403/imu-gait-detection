"""Re-implementation of the shank-gyroscope rule-based detector of
Catalfamo, Ghoussayni & Ewins, Sensors 2010, 10:5683-5702 (doi:10.3390/s100605683).

Rules as described in the paper (Sec. 2.3, Fig. 3):
  - 2nd-order Butterworth low-pass, 35 Hz, applied forwards and backwards.
  - Swing: signal above a threshold for at least 40 ms.
  - IC: first negative minimum after the swing wave.
  - Waiting time before FO search: 50 % of the first positive-wave duration (first step),
    then 50 % of the previous stance duration.
  - FO: after the waiting time, a sample that is the minimum of a 200 ms window
    (decreasing before, increasing after), before the next swing.
Only adaptation: the original swing threshold is 0.2 V (sensor-specific, no deg/s conversion
given), so it is expressed here as 0.20 x P99 of the trial signal. Sign: swing positive.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def detect(g, fs, thr_frac=0.20):
    b, a = butter(2, min(35.0, 0.45 * fs) / (fs / 2)); y = filtfilt(b, a, g)
    thr = thr_frac * np.percentile(y, 99)
    above = y > thr; n40 = max(int(round(0.040 * fs)), 1); half = int(round(0.100 * fs))
    # swing waves: runs above threshold lasting >= 40 ms
    edges = np.flatnonzero(np.diff(above.astype(int))); starts = edges[np.diff(np.r_[0, above[edges + 1]]) >= 0] if False else None
    runs, i, N = [], 0, len(y)
    while i < N:
        if above[i]:
            j = i
            while j < N and above[j]: j += 1
            if j - i >= n40: runs.append((i, j))
            i = j
        else:
            i += 1
    hs, to = [], []
    last_stance = None
    for k, (s, e) in enumerate(runs):
        nxt = runs[k + 1][0] if k + 1 < len(runs) else N
        # IC: first negative local minimum after the swing wave
        ic = None
        for m in range(e, nxt - 1):
            if y[m] < 0 and y[m] <= y[m - 1] and y[m] <= y[m + 1]:
                ic = m; break
        if ic is None: continue
        hs.append(ic)
        wait = int(0.5 * (e - s)) if last_stance is None else int(0.5 * last_stance)
        # FO: minimum of a 200 ms window after the waiting time, before the next swing
        fo = None
        for m in range(ic + wait, nxt):
            lo, hi = max(m - half, 0), min(m + half + 1, N)
            if y[m] == y[lo:hi].min() and m > 0 and m < N - 1 and y[m] < y[m - 1] + 1e-12 and y[m] <= y[m + 1]:
                fo = m; break
        if fo is not None:
            to.append(fo); last_stance = fo - ic
    return np.array(hs, int), np.array(to, int)
