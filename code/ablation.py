"""Component ablation and parameter sweeps (Table IV). Each variant changes one component of the detector.
Evaluated at +/-100 ms: per-subject recall against the force platforms (both datasets) and F1 against gcRight (Dataset 2)."""
import paths as PTH
import os, sys, glob, numpy as np, pandas as pd, scipy.io as sio, json
from scipy.signal import butter, filtfilt, find_peaks, argrelmin
sys.path.insert(0, f'{PTH.CODE}'); import metrics as M
PAC = f'{PTH.PAC}'; CAM = f'{PTH.CAM}'
BASE = dict(fc=15.0, msw_prom=0.30, msw_dist=0.45, msw_height=0.30, neg_prom=0.20, polarity=True,
            adaptive=True, fixed_prom=1.0, hs_rule='first_min_below0', median=True)


def detect(g, fs, P):
    if P['polarity'] and np.percentile(g, 99) < -np.percentile(g, 1): g = -g
    b, a = butter(4, min(P['fc'], 0.45 * fs) / (fs / 2)); y = filtfilt(b, a, g)
    if P['median']: y = y - np.median(y)
    iqr = np.percentile(y, 75) - np.percentile(y, 25)
    prom = P['msw_prom'] * iqr if P['adaptive'] else P['fixed_prom']
    nprom = P['neg_prom'] * iqr if P['adaptive'] else P['fixed_prom'] * 2 / 3
    h = P['msw_height'] * np.percentile(y, 99) if P['msw_height'] > 0 else None
    msw, _ = find_peaks(y, prominence=prom, distance=max(int(P['msw_dist'] * fs), 1), height=h)
    neg, _ = find_peaks(-y, prominence=nprom, distance=int(0.20 * fs)); lmin = argrelmin(y)[0]; hs, to = [], []
    for k, m in enumerate(msw):
        nxt = msw[k + 1] if k + 1 < len(msw) else len(y); prv = msw[k - 1] if k > 0 else 0
        if P['hs_rule'] == 'first_min_below0': c = lmin[(lmin > m) & (lmin < nxt) & (y[lmin] < 0)]
        else: c = neg[(neg > m) & (neg < nxt)]
        if len(c): hs.append(c[0])
        c = neg[(neg < m) & (neg > prv)]
        if len(c): to.append(c[-1])
    return np.array(sorted(set(hs)), int), np.array(sorted(set(to)), int)


def load_all():
    G = pd.read_csv(f'{PTH.GT}')
    G = G[(G.peakF >= 300) & (G.stance_s >= 0.3) & (G.stance_s <= 1.5)]
    pac = []
    for t, g in G.groupby('trial'):
        imu = sio.loadmat(f'{PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
        for side, s in [('R', 'rshank'), ('L', 'lshank')]:
            gs = g[g.side == side]
            pac.append(dict(subj=f'P{t:02d}', x=getattr(imu, s).gyr[:, 2] * np.pi / 180, fs=1000 / 3.5,
                            hs=gs.hs_s.values, to=gs.to_s.values))
    cam = []
    for f in sorted(glob.glob(CAM + '/*_events.csv')):
        s = os.path.basename(f)[:4]; E = pd.read_csv(f); z = np.load(f'{CAM}/{s}_signals.npz')
        tm = [k for k in z.files if k.startswith('treadmill')]
        gall = np.concatenate([np.nan_to_num(z[k][:, 2].astype(float)) for k in tm])
        pol = 1.0 if np.percentile(gall, 99) > -np.percentile(gall, 1) else -1.0
        for k in z.files:
            key = k.replace('__', '/'); S = z[k]; t = S[:, 0].astype(float); e = E[E.key == key]
            fp = e[(e.src == 'fp') & (e.plate != 'Combined')].copy(); fp['r'] = fp.rfoot_mid / fp.rfoot_all_med; fp['d'] = fp.t_off - fp.t
            fp = fp[(fp.r < 0.5) & (fp.peakF >= 300) & (fp.d >= 0.3) & (fp.d <= 1.5) & (fp.t > t[0] + 1) & (fp.t_off < t[-1] - 1)]
            cam.append(dict(subj=s, act=key.split('/')[0], t=t, x=pol * np.nan_to_num(S[:, 2].astype(float)), raw_sign=pol,
                            fp_hs=fp.t.values, fp_to=fp.t_off.values,
                            gc_hs=np.sort(e[(e.src == 'gc') & (e.event == 'HS')].t.values),
                            gc_to=np.sort(e[(e.src == 'gc') & (e.event == 'TO')].t.values)))
    return pac, cam


def evaluate(P, pac, cam, tol=0.1):
    rows = []
    for r in pac:
        hs, to = detect(r['x'], r['fs'], P); hs = hs / r['fs']; to = to / r['fs']
        for ev, d, g in [('HS', hs, r['hs']), ('TO', to, r['to'])]:
            m = M.match(d, g, tol); rows.append(dict(ds='Pacini', act='levelground', ref='FP', ev=ev, subj=r['subj'], tp=len(m), ng=len(g), nd=np.nan,
                                                     ae=np.sum([abs(e) for _, _, e in m])))
    for r in cam:
        x = r['x'] if P['polarity'] else r['x'] * r['raw_sign']   # undo the subject-level sign fix for the no-polarity variant
        hi, ti = detect(x, 200.0, P); hs = r['t'][hi]; to = r['t'][ti]
        for ref, ev, d, g in [('FP', 'HS', hs, r['fp_hs']), ('FP', 'TO', to, r['fp_to']), ('gc', 'HS', hs, r['gc_hs']), ('gc', 'TO', to, r['gc_to'])]:
            m = M.match(d, g, tol)
            nd = np.sum((d >= g.min() - 0.3) & (d <= g.max() + 0.3)) if (ref == 'gc' and len(g)) else np.nan
            rows.append(dict(ds='Camargo', act=r['act'], ref=ref, ev=ev, subj=r['subj'], tp=len(m), ng=len(g), nd=nd, ae=np.sum([abs(e) for _, _, e in m])))
    R = pd.DataFrame(rows)
    out = {}
    # Same aggregation as statistics_tests.py: metric per subject and activity, mean over activities per subject, mean over subjects
    for (ds, ref, ev), g in R.groupby(['ds', 'ref', 'ev']):
        q = g.groupby(['subj', 'act'])[['tp', 'ng', 'nd', 'ae']].sum()
        q = q[q.ng > 0]
        rec = q.tp / q.ng; mae = (q.ae / q.tp.where(q.tp > 0)) * 1000
        out[f'{ds}_{ref}_{ev}_recall'] = rec.groupby(level=0).mean().mean()
        out[f'{ds}_{ref}_{ev}_mae'] = mae.groupby(level=0).mean().mean()
        if ref == 'gc':
            pr = q.tp / q.nd; f1 = (2 * pr * rec / (pr + rec)).fillna(0)
            out[f'{ds}_{ref}_{ev}_f1'] = f1.groupby(level=0).mean().mean()
    return out


VARIANTS = [('final', {})]
VARIANTS += [('no_polarity_check', dict(polarity=False)), ('fixed_prominence_1.0rad/s', dict(adaptive=False)),
             ('no_msw_height_rule', dict(msw_height=0.0)), ('prominent_peak_HS', dict(hs_rule='prominent_neg')),
             ('no_median_removal', dict(median=False))]
VARIANTS += [(f'fc_{v:g}Hz', dict(fc=v)) for v in [5, 10, 20, 30, 35]]
VARIANTS += [(f'msw_spacing_{v:g}s', dict(msw_dist=v)) for v in [0.30, 0.60, 0.80]]
VARIANTS += [(f'msw_prom_{v:g}xIQR', dict(msw_prom=v)) for v in [0.15, 0.45, 0.60]]
VARIANTS += [(f'msw_height_{v:g}xP99', dict(msw_height=v)) for v in [0.20, 0.40, 0.50]]

if __name__ == '__main__':
    pac, cam = load_all(); print('loaded', len(pac), len(cam), flush=True); res = []
    for name, ch in VARIANTS:
        P = dict(BASE); P.update(ch); o = evaluate(P, pac, cam); o['variant'] = name; res.append(o)
        print(name, {k: round(v, 3) for k, v in o.items() if k.endswith('recall') or k.endswith('f1')}, flush=True)
        pd.DataFrame(res).to_csv(f'{PTH.RESULTS}/ablation.csv', index=False)
