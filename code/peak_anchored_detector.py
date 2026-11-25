"""Peak-anchored gait event detector for a shank-worn gyroscope (Section II of the paper).

Input: sagittal shank angular velocity g (rad/s), already sign-corrected so that the mid-swing peak is
positive (see `polarity`), sampled at fs (Hz). Each leg is processed independently.
Output: indices of heel-strike (HS), toe-off (TO) and mid-swing (MSW) samples.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, argrelmin

# every parameter of the detector (identical for all datasets, subjects and terrains)
P = dict(fc=15.0, order=4, msw_prom=0.30, msw_dist_s=0.45, msw_height=0.30, neg_prom=0.20, neg_dist_s=0.20)


def polarity(g):
    """+1 or -1 so that the mid-swing peak of the shank angular velocity is positive."""
    return 1.0 if np.percentile(g, 99) > -np.percentile(g, 1) else -1.0


def detect(g, fs=200.0, p=P, return_all=False):
    # zero-phase fourth-order Butterworth low-pass filter, recording median removed
    b, a = butter(p['order'], p['fc'] / (fs / 2)); y = filtfilt(b, a, g); y = y - np.median(y)
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    # mid-swing anchors: prominent, high and well-spaced positive peaks
    msw, _ = find_peaks(y, prominence=p['msw_prom'] * iqr, distance=int(p['msw_dist_s'] * fs),
                        height=p['msw_height'] * np.percentile(y, 99))
    neg, _ = find_peaks(-y, prominence=p['neg_prom'] * iqr, distance=int(p['neg_dist_s'] * fs))
    lmin = argrelmin(y)[0]; hs = []; to = []
    for k, m in enumerate(msw):
        nxt = msw[k + 1] if k + 1 < len(msw) else len(y); prv = msw[k - 1] if k > 0 else 0
        c = lmin[(lmin > m) & (lmin < nxt) & (y[lmin] < 0)]      # HS: first local minimum below zero after the anchor
        if len(c): hs.append(int(c[0]))
        c = neg[(neg < m) & (neg > prv)]                            # TO: last negative peak before the anchor
        if len(c): to.append(int(c[-1]))
    out = dict(hs=np.array(sorted(set(hs)), dtype=int), to=np.array(sorted(set(to)), dtype=int), msw=msw)
    if return_all: out['y'] = y
    return out
