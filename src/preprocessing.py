"""Signal preprocessing and shared parameters."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class Params:
    fs: float = 200.0
    lp_cutoff: float = 15.0
    lp_order: int = 4

    win_s: float = 1.5
    pct_high: float = 75.0
    pct_low: float = 15.0
    scale_high: float = 0.55
    scale_low: float = 0.55

    msw_min_dist_s: float = 0.45
    msw_prominence_frac: float = 0.30
    neg_prominence_frac: float = 0.20
    neg_min_dist_s: float = 0.20

    min_stance_s: float = 0.20
    min_swing_s: float = 0.25
    min_cycle_s: float = 0.45


def lowpass(x, fs, cutoff, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, x)


def rolling_percentile(x, fs, win_s, pct, stride_s=0.10):
    n = len(x)
    w = max(int(win_s * fs) | 1, 5)
    half = w // 2
    stride = max(int(round(stride_s * fs)), 1)
    xp = np.pad(x, half, mode="edge")
    anchors = np.arange(0, n, stride)
    if anchors[-1] != n - 1:
        anchors = np.append(anchors, n - 1)
    idx = anchors[:, None] + np.arange(w)[None, :]
    windows = xp[idx]
    vals = np.percentile(windows, pct, axis=1)
    return np.interp(np.arange(n), anchors, vals)


def adaptive_thresholds(x, params):
    high = params.scale_high * rolling_percentile(
        x, params.fs, params.win_s, params.pct_high)
    low = params.scale_low * rolling_percentile(
        x, params.fs, params.win_s, params.pct_low)
    return high, low


def precondition(gyro_z, params):
    g = lowpass(gyro_z, params.fs, params.lp_cutoff, params.lp_order)
    return g - np.median(g)
