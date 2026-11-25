"""Recall and timing error of every method versus treadmill belt speed (Dataset 2, right-belt force-platform contacts).
The belt speed at each contact is read from the released speed records."""
import paths as PTH
import sys, glob, os, numpy as np, pandas as pd
sys.path.insert(0, f'{PTH.CODE}')
import evaluate as EA, metrics as M
CAM = EA.CAM
SP = pd.concat([pd.read_csv(f) for f in glob.glob(CAM + '/treadmill_speed_*.csv')])
DETS = ['peak_anchored', 'rf', 'hgb', 'cnn', 'salarian2004', 'catalfamo2010', 'greene2010', 'aminian2002', 'fixed_threshold']
rows = []
for f in sorted(glob.glob(CAM + '/*_events.csv')):
    s = os.path.basename(f)[:4]; E = pd.read_csv(f); z = np.load(f'{CAM}/{s}_signals.npz')
    for k in [k for k in z.files if k.startswith('treadmill')]:
        key = k.replace('__', '/'); S = z[k]; t = S[:, 0].astype(float)
        x0 = np.nan_to_num(S[:, 2].astype(float)); x = EA.polarity(x0) * x0; fs = 1 / np.median(np.diff(t))
        e = E[E.key == key]; fp = e[(e.src == 'fp') & (e.plate != 'Combined')].copy()
        fp['ratio'] = fp.rfoot_mid / fp.rfoot_all_med; fp['dur'] = fp.t_off - fp.t
        fp = fp[(fp.ratio < 0.5) & (fp.peakF >= 300) & (fp.dur >= 0.3) & (fp.dur <= 1.5) & (fp.t > t[0] + 1) & (fp.t_off < t[-1] - 1)]
        sp = SP[(SP.subject == s) & (SP.key == key)].sort_values('t')
        if len(sp) == 0 or len(fp) == 0: continue
        v = np.interp(fp.t.values, sp.t.values, sp.speed.values)
        for det in DETS:
            hi, ti = EA.DETECTORS[det](x, fs, ctx=('Camargo', s, key))
            hs = t[np.sort(np.asarray(hi, int))]; to = t[np.sort(np.asarray(ti, int))]
            for ev, dd, gt in [('HS', hs, fp.t.values), ('TO', to, fp.t_off.values)]:
                m100 = {i: er for i, j, er in M.match(dd, gt, 0.1)}; m150 = {i: er for i, j, er in M.match(dd, gt, 0.15)}
                for i in range(len(gt)):
                    rows.append(dict(detector=det, subject=s, event=ev, speed=v[i], matched=i in m100, err_ms=m150.get(i, np.nan) * 1000))
    print(s, flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{PTH.RESULTS}/speed_events.csv', index=False)
R['bin'] = pd.cut(R.speed, [0.45, 0.75, 1.05, 1.35, 1.65, 1.95], labels=['0.5-0.7', '0.8-1.0', '1.1-1.3', '1.4-1.6', '1.7-1.9'])
Sb = R.groupby(['detector', 'event', 'bin', 'subject'], observed=True).matched.mean().groupby(['detector', 'event', 'bin'], observed=True).agg(['mean', 'count'])
print(Sb.unstack('bin')['mean'].round(3).to_string())
print(R.groupby('bin', observed=True).size())
