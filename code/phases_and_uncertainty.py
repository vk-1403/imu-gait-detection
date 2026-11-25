"""(1) Per-event timing uncertainty sigma_t (Eq. (2) of the paper) and its relation to the observed error against
the force platforms. (2) Gait-phase durations against the force platforms: stance time (both datasets), stride and
swing time on the instrumented treadmill of Dataset 2 (consecutive right-belt contacts). (3) Median and quartiles
of the signed errors on Dataset 1 in the format of Pacini Panebianco et al. (2018)."""
import paths as PTH
import os, sys, glob, numpy as np, pandas as pd, scipy.io as sio
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt
sys.path.insert(0, f'{PTH.CODE}'); import peak_anchored_detector as D, metrics as M
PAC = f'{PTH.PAC}'; CAM = f'{PTH.CAM}'; W = 2


def noise_sigma(raw, fs):
    b, a = butter(4, min(40.0, 0.45 * fs) / (fs / 2)); return float(np.std(raw - filtfilt(b, a, raw), ddof=1))


def sig_t(y, k, dt, s):
    if k - W < 0 or k + W + 1 > len(y): return np.nan
    w = y[k - W:k + W + 1]; w = -w if (w[W] < w[0] and w[W] < w[-1]) else w
    x = np.arange(-W, W + 1, dtype=float); A = np.vstack([x * x, x, np.ones_like(x)]).T
    c, b, _ = np.linalg.lstsq(A, w, rcond=None)[0]
    if abs(c) < 1e-3: return np.nan
    cov = s * s * np.linalg.inv(A.T @ A); da, db_ = -1 / (2 * c), b / (2 * c * c)
    v = da * da * cov[1, 1] + db_ * db_ * cov[0, 0] + 2 * da * db_ * cov[0, 1]
    return np.sqrt(max(v, 0)) * dt * 1000


def polarity(g): return 1.0 if np.percentile(g, 99) > -np.percentile(g, 1) else -1.0


rows, phase, allev = [], [], []
# ---------- Pacini
G = pd.read_csv(f'{PTH.GT}'); G = G[(G.peakF >= 300) & (G.stance_s >= 0.3) & (G.stance_s <= 1.5)]
for t, g in G.groupby('trial'):
    imu = sio.loadmat(f'{PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
    for side, sn in [('R', 'rshank'), ('L', 'lshank')]:
        fs = 1000 / 3.5; raw = getattr(imu, sn).gyr[:, 2] * np.pi / 180; raw = polarity(raw) * raw
        d = D.detect(raw, fs, return_all=True); s = noise_sigma(raw, fs); y = d['y']
        for ev in ['hs', 'to', 'msw']:
            for k in d[ev]: allev.append(dict(ds='Pacini', subj=f'P{t:02d}', ev=ev.upper(), sigma_ms=sig_t(y, k, 1 / fs, s)))
        hs, to = d['hs'] / fs, d['to'] / fs; gs = g[g.side == side]
        mh = {i: (j, e) for i, j, e in M.match(hs, gs.hs_s.values, 0.15)}; mt = {i: (j, e) for i, j, e in M.match(to, gs.to_s.values, 0.15)}
        for i in range(len(gs)):
            r = gs.iloc[i]
            for ev, mm, idx in [('HS', mh, d['hs']), ('TO', mt, d['to'])]:
                if i in mm: rows.append(dict(ds='Pacini', subj=f'P{t:02d}', act='levelground', ev=ev, err_ms=mm[i][1] * 1000,
                                             sigma_ms=sig_t(y, idx[mm[i][0]], 1 / fs, s)))
            if i in mh and i in mt:
                st_imu = to[mt[i][0]] - hs[mh[i][0]]; phase.append(dict(ds='Pacini', subj=f'P{t:02d}', act='levelground', param='stance',
                                                                        ref_ms=r.stance_s * 1000, err_ms=(st_imu - r.stance_s) * 1000))
# ---------- Camargo
for f in sorted(glob.glob(CAM + '/*_events.csv')):
    sj = os.path.basename(f)[:4]; E = pd.read_csv(f); z = np.load(f'{CAM}/{sj}_signals.npz')
    tm = [k for k in z.files if k.startswith('treadmill')]; pol = polarity(np.concatenate([np.nan_to_num(z[k][:, 2].astype(float)) for k in tm]))
    for k in z.files:
        key = k.replace('__', '/'); act = key.split('/')[0]; S = z[k]; t = S[:, 0].astype(float); raw0 = np.nan_to_num(S[:, 2].astype(float)); raw = polarity(raw0) * raw0   # sign per recording (trial)
        d = D.detect(raw, 200.0, return_all=True); s = noise_sigma(raw, 200.0); y = d['y']
        for ev in ['hs', 'to', 'msw']:
            for kk in d[ev]: allev.append(dict(ds='Camargo', subj=sj, ev=ev.upper(), sigma_ms=sig_t(y, kk, 0.005, s)))
        e = E[E.key == key]; fp = e[(e.src == 'fp') & (e.plate != 'Combined')].copy(); fp['r'] = fp.rfoot_mid / fp.rfoot_all_med; fp['d'] = fp.t_off - fp.t
        fp = fp[(fp.r < 0.5) & (fp.peakF >= 300) & (fp.d >= 0.3) & (fp.d <= 1.5) & (fp.t > t[0] + 1) & (fp.t_off < t[-1] - 1)].sort_values('t')
        hs, to = t[d['hs']], t[d['to']]
        mh = {i: (j, er) for i, j, er in M.match(hs, fp.t.values, 0.15)}; mt = {i: (j, er) for i, j, er in M.match(to, fp.t_off.values, 0.15)}
        for i in range(len(fp)):
            for ev, mm, idx in [('HS', mh, d['hs']), ('TO', mt, d['to'])]:
                if i in mm: rows.append(dict(ds='Camargo', subj=sj, act=act, ev=ev, err_ms=mm[i][1] * 1000, sigma_ms=sig_t(y, idx[mm[i][0]], 0.005, s)))
            if i in mh and i in mt:
                phase.append(dict(ds='Camargo', subj=sj, act=act, param='stance', ref_ms=fp.d.values[i] * 1000,
                                  err_ms=((to[mt[i][0]] - hs[mh[i][0]]) - fp.d.values[i]) * 1000))
        if act == 'treadmill':   # consecutive right-belt contacts -> stride and swing
            fr = fp[fp.plate == 'Treadmill_R'].reset_index(drop=True)
            for i in range(len(fr) - 1):
                if i in mh and (i + 1) in mh and i in mt:
                    stride_ref = fr.t[i + 1] - fr.t[i]
                    if stride_ref > 2.0: continue
                    phase.append(dict(ds='Camargo', subj=sj, act=act, param='stride', ref_ms=stride_ref * 1000,
                                      err_ms=((hs[mh[i + 1][0]] - hs[mh[i][0]]) - stride_ref) * 1000))
                    phase.append(dict(ds='Camargo', subj=sj, act=act, param='swing', ref_ms=(fr.t[i + 1] - fr.t_off[i]) * 1000,
                                      err_ms=((hs[mh[i + 1][0]] - to[mt[i][0]]) - (fr.t[i + 1] - fr.t_off[i])) * 1000))
R = pd.DataFrame(rows); P = pd.DataFrame(phase); A = pd.DataFrame(allev)
os.makedirs(f'{PTH.RESULTS}', exist_ok=True)
R.to_csv(f'{PTH.RESULTS}/matched_errors_sigma.csv', index=False); P.to_csv(f'{PTH.RESULTS}/phase_durations.csv', index=False)
A.to_csv(f'{PTH.RESULTS}/sigma_all_events.csv', index=False)
print('=== per-event sigma_t (all detected events) ===')
print(A.groupby(['ds', 'ev']).sigma_ms.describe(percentiles=[.25, .5, .75, .95]).round(2).to_string())
print('=== sigma_t vs |error| (Spearman, matched within 150 ms) ===')
for (ds, ev), g in R.groupby(['ds', 'ev']):
    g = g.dropna(); rho, p = spearmanr(g.sigma_ms, g.err_ms.abs()); print(ds, ev, 'n', len(g), 'rho %.3f p %.2g' % (rho, p), 'median |err| %.1f ms' % g.err_ms.abs().median())
print('=== Pacini literature-format (signed error, ms) ===')
for ev, g in R[R.ds == 'Pacini'].groupby('ev'):
    q = np.percentile(g.err_ms, [25, 50, 75]); print(ev, 'median %.1f Q1 %.1f Q3 %.1f Dmed %.1f' % (q[1], q[0], q[2], q[2] - q[0]))
g = P[(P.ds == 'Pacini')]; q = np.percentile(g.err_ms, [25, 50, 75]); print('stance median %.1f Q1 %.1f Q3 %.1f Dmed %.1f' % (q[1], q[0], q[2], q[2] - q[0]))
print('=== phase-duration errors (ms): subject-mean bias, SD, MAE ===')
for (ds, act, prm), g in P.groupby(['ds', 'act', 'param']):
    S = M.summarise_errors(g.err_ms); print(ds, act, prm, 'n', S['n'], 'ref mean %.0f' % g.ref_ms.mean(), 'bias %.1f SD %.1f MAE %.1f' % (S['bias'], S['sd'], S['mae']))
