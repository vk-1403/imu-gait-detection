"""Reliability of the polarity rule (P99 versus -P1) per recording in Dataset 2.
For every recording the detector is run with both signs, and the sign giving the higher gcRight HS F1 (+/-100 ms) is
taken as correct. This is used only to check the rule, never inside the detector."""
import paths as PTH
import numpy as np, pandas as pd, glob, os, sys
sys.path.insert(0, f'{PTH.CODE}'); import peak_anchored_detector as D, metrics as M
CAM = f'{PTH.CAM}'
pol = lambda g: 1.0 if np.percentile(g, 99) > -np.percentile(g, 1) else -1.0
rows = []
for f in sorted(glob.glob(CAM + '/*_signals.npz')):
    s = os.path.basename(f)[:4]; z = np.load(f); E = pd.read_csv(f'{CAM}/{s}_events.csv')
    tm = [k for k in z.files if k.startswith('treadmill')]
    Ps = pol(np.concatenate([np.nan_to_num(z[k][:, 2].astype(float)) for k in tm]))
    for k in z.files:
        key = k.replace('__', '/'); S = z[k]; t = S[:, 0].astype(float); g = np.nan_to_num(S[:, 2].astype(float))
        gt = np.sort(E[(E.key == key) & (E.src == 'gc') & (E.event == 'HS')].t.values)
        if len(gt) < 3: continue
        f1 = {}
        for sg in (1.0, -1.0):
            hs = t[D.detect(sg * g, 200.0)['hs'].astype(int)]; m = M.match(hs, gt, 0.1)
            nd = np.sum((hs >= gt.min() - .3) & (hs <= gt.max() + .3)); r = len(m) / len(gt); p = len(m) / max(nd, 1)
            f1[sg] = 2 * p * r / (p + r) if p + r > 0 else 0
        best = 1.0 if f1[1.0] >= f1[-1.0] else -1.0
        q99, q1 = np.percentile(g, 99), -np.percentile(g, 1)
        rows.append(dict(subject=s, key=key, act=key.split('/')[0], trial_rule=pol(g), subject_rule=Ps, best=best,
                         f1_pos=f1[1.0], f1_neg=f1[-1.0], ratio=max(q99, q1) / min(q99, q1)))
R = pd.DataFrame(rows); R.to_csv(f'{PTH.RESULTS}/polarity_check.csv', index=False)
print('trials', len(R))
print('per-trial rule correct', (R.trial_rule == R.best).sum(), 'subject rule correct', (R.subject_rule == R.best).sum())
print(R[R.trial_rule != R.best].to_string())
print(R.groupby('act').apply(lambda d: pd.Series(dict(n=len(d), trial_ok=(d.trial_rule == d.best).mean(), subj_ok=(d.subject_rule == d.best).mean()))))
print('ratio min', R.ratio.min(), 'P5', R.ratio.quantile(.05))
