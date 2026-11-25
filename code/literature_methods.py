"""Re-implementations of three established shank-gyroscope gait-event algorithms, from the original papers.
Input: sagittal shank angular velocity in rad/s (any sign), sampling rate fs. Output: (HS idx, TO idx).
Signals are converted to deg/s because the published thresholds are in deg/s.

Salarian A. et al., IEEE TBME 51(8):1434-1443, 2004 (doi 10.1109/TBME.2004.827933), Sec. II-C:
  midswing = local maxima > 50 deg/s; of adjacent peaks within 500 ms keep the highest.
  IC = nearest local minimum after midswing (search interval +/-1.5 s).
  TC: signal low-pass FIR, fc ~ 30 Hz; TC = minimum before midswing with amplitude < -20 deg/s.
  Not reproduced: the drift high-pass of Appendix A (cutoff not legible in our copy; drift negligible here).
Greene B.R. et al., Med Biol Eng Comput 48:1251-1260, 2010 (doi 10.1007/s11517-010-0692-0), Sec. 2.5:
  zero-phase 5th-order Butterworth low-pass 5 Hz; sign from skewness; adaptive thresholds th1-th6;
  0.5 s peak merging, +/-1.5 s search. Not reproduced: bilateral artefact rejection (needs both legs).
Aminian K. et al., J Biomech 35:689-699, 2002 (doi 10.1016/S0021-9290(02)00008-8), Sec. 2.4 / Fig. 4:
  coif5 multiresolution decomposition (signal at 200 Hz); sa = sum of details 1..9; HS-enhanced = A1(sa)-A9(sa);
  TO-enhanced = A3(sa)-A9(sa); midswing = peaks of each enhanced signal; HS = local minimum in [ms+0.25, ms+2] s;
  TO = local minimum in [ms-2, ms-0.05] s refined on the original signal in [to, to+0.075] s; keep pairs with
  0.1 < hs-to < 2.5 s. Unspecified in the paper and set here: peak detection = spacing 0.5 s, height 0.5 x P99;
  first minimum after ms+0.25 s (HS) and last minimum before ms-0.05 s (TO).
"""
import numpy as np
from scipy.signal import butter, filtfilt, firwin, argrelmin, argrelmax, find_peaks, resample_poly
from scipy.stats import skew

D = 180.0 / np.pi


def _merge_peaks(idx, vals, gap):
    """Among peaks closer than `gap` samples to each other, keep the highest."""
    out = []
    for i, v in zip(idx, vals):
        if out and i - out[-1][0] < gap:
            if v > out[-1][1]: out[-1] = (i, v)
        else:
            out.append((i, v))
    return np.array([o[0] for o in out], int)


def salarian2004(x, fs):
    s = np.asarray(x, float) * D
    if np.percentile(s, 99) < -np.percentile(s, 1): s = -s          # same sign handling as all methods
    mx = argrelmax(s)[0]; mx = mx[s[mx] > 50.0]
    ms = _merge_peaks(mx, s[mx], int(0.5 * fs))
    taps = firwin(int(0.15 * fs) | 1, 30.0 / (fs / 2)); sf = filtfilt(taps, [1.0], s)
    mn_raw = argrelmin(s)[0]; mn_f = argrelmin(sf)[0]; mn_f = mn_f[sf[mn_f] < -20.0]
    w = int(1.5 * fs); hs, to = [], []
    for m in ms:
        c = mn_raw[(mn_raw > m) & (mn_raw <= m + w)]
        if len(c): hs.append(c[0])
        c = mn_f[(mn_f < m) & (mn_f >= m - w)]
        if len(c): to.append(c[-1])
    return np.array(sorted(set(hs)), int), np.array(sorted(set(to)), int)


def greene2010(x, fs):
    b, a = butter(5, 5.0 / (fs / 2)); s = filtfilt(b, a, np.asarray(x, float) * D)
    if skew(s) < 0: s = -s
    mu = s.mean()
    th1 = 0.6 * s.max(); th2 = 0.8 * s[s > mu].mean(); th3 = 0.8 * abs(s[s < mu].mean())
    th4 = 0.8 * s[s < mu].mean(); th5 = mu; th6 = 2 * th3
    mx = argrelmax(s)[0]; mn = argrelmin(s)[0]
    def prev_min(i):
        p = mn[mn < i]; return s[p[-1]] if len(p) else s[:i].min() if i > 0 else s[0]
    def prev_max(i):
        p = mx[mx < i]; return s[p[-1]] if len(p) else s[:i].max() if i > 0 else s[0]
    cand = [i for i in mx if s[i] > th2 and s[i] - prev_min(i) >= th1]
    ms = _merge_peaks(np.array(cand, int), s[np.array(cand, int)] if cand else np.array([]), int(0.5 * fs))
    w = int(1.5 * fs); hs, to = [], []
    for m in ms:
        c = [i for i in mn[(mn > m) & (mn <= m + w)] if s[i] < th5 and prev_max(i) - s[i] >= th3]
        if c: hs.append(c[0])
        c = [i for i in mn[(mn < m) & (mn >= m - w)] if s[i] < th4 and prev_max(i) - s[i] >= th6]
        if c: to.append(c[-1])
    return np.array(sorted(set(hs)), int), np.array(sorted(set(to)), int)


def aminian2002(x, fs):
    import pywt
    s0 = np.asarray(x, float) * D
    if np.percentile(s0, 99) < -np.percentile(s0, 1): s0 = -s0
    # work at 200 Hz as in the paper (exact for 200 Hz and 1000/3.5 Hz inputs)
    if abs(fs - 200.0) > 1e-6:
        from fractions import Fraction
        fr = Fraction(200.0 / fs).limit_denominator(100); s = resample_poly(s0, fr.numerator, fr.denominator); scale = fs / 200.0
    else:
        s = s0; scale = 1.0
    F = 200.0; n = len(s)

    def band(sig, keep_details):          # reconstruct with only the listed detail levels (1 = finest)
        c = pywt.wavedec(sig, 'coif5', level=9); out = [np.zeros_like(c[0])]
        for lev_i, cc in enumerate(c[1:]):
            level = 9 - lev_i
            out.append(cc if level in keep_details else np.zeros_like(cc))
        return pywt.waverec(out, 'coif5')[:len(sig)]
    sa = band(s, set(range(1, 10)))
    hs_sig = band(sa, set(range(2, 10)))   # A1(sa) - A9(sa)
    to_sig = band(sa, set(range(4, 10)))   # A3(sa) - A9(sa)
    def mid(sig):
        p, _ = find_peaks(sig, distance=int(0.5 * F), height=0.5 * np.percentile(sig, 99)); return p
    ms_h, ms_t = mid(hs_sig), mid(to_sig)
    mn_h = argrelmin(hs_sig)[0]; mn_t = argrelmin(to_sig)[0]
    hs = []
    for m in ms_h:
        c = mn_h[(mn_h >= m + int(0.25 * F)) & (mn_h <= m + int(2 * F))]
        if len(c): hs.append(c[0])
    to = []
    for m in ms_t:
        c = mn_t[(mn_t >= m - int(2 * F)) & (mn_t <= m - int(0.05 * F))]
        if len(c):
            t0 = c[-1]; seg = s[t0:min(t0 + int(0.075 * F) + 1, n)]
            to.append(t0 + int(np.argmin(seg)))
    hs = np.array(sorted(set(hs)), int); to = np.array(sorted(set(to)), int)
    keep_hs, keep_to = [], []
    for h in hs:                           # pairing rule 0.1 s < hs - to < 2.5 s, nearest to
        d = (h - to) / F; ok = np.flatnonzero((d > 0.1) & (d < 2.5))
        if len(ok): keep_hs.append(h); keep_to.append(to[ok[np.argmin(d[ok])]])
    n0 = len(s0)
    hs = np.clip(np.round(np.array(keep_hs) * scale).astype(int), 0, n0 - 1)
    to = np.clip(np.round(np.array(sorted(set(keep_to))) * scale).astype(int), 0, n0 - 1)
    return hs, to
