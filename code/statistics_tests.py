"""Subject-level statistics.
Unit of analysis = subject (Dataset 1: 27 subjects, one recording each; Dataset 2: 22 subjects).
Per subject: recall at +/-100 ms against the force platforms (and F1 against gcRight for Dataset 2), MAE of matched events.
Friedman test across methods; two-sided Wilcoxon signed-rank tests of the proposed detector against each method;
Holm-Bonferroni correction over all Wilcoxon tests (one family). 95% CIs: percentile bootstrap over subjects,
10,000 resamples, seed 0."""
import paths as PTH
import numpy as np, pandas as pd, sys
from scipy.stats import friedmanchisquare, wilcoxon
sys.path.insert(0, f'{PTH.CODE}'); import metrics as M
DETS = ['peak_anchored', 'salarian2004', 'greene2010', 'aminian2002', 'catalfamo2010', 'fixed_threshold', 'hgb', 'rf', 'cnn']
P = pd.read_csv(f'{PTH.RESULTS}/per_subject_all.csv')
P = P[P.detector.isin(DETS) & (P.tol == 100)]
# For Camargo, pool activities per subject (one value per subject) and also keep per-activity tables
rows, tests = [], []
def per_subject(df, metric):
    return df.groupby(['detector', 'subject'])[metric].mean().unstack(0)
blocks = []
for ds, ref, metric in [('Pacini', 'FP', 'recall'), ('Camargo', 'FP', 'recall'), ('Camargo', 'gc', 'f1'), ('Pacini', 'FP', 'mae'), ('Camargo', 'FP', 'mae')]:
    for ev in ['HS', 'TO']:
        d = P[(P.dataset == ds) & (P.ref == ref) & (P.event == ev)]
        if metric == 'f1': d = d[d.f1.notna()]
        else: d = d[d.f1.isna()] if ref == 'gc' else d
        T = per_subject(d, metric).dropna(axis=1, how='all')
        dets = [x for x in DETS if x in T.columns]
        for det in dets:   # descriptive: every subject with a value for this detector
            v = T[det].dropna().values; lo, hi = M.boot_ci(v)
            rows.append(dict(dataset=ds, ref=ref, event=ev, metric=metric, detector=det, n_subj=len(v), mean=v.mean(), sd=v.std(ddof=1), ci_lo=lo, ci_hi=hi))
        Tc = T[dets].dropna()
        if len(dets) >= 3:
            fr = friedmanchisquare(*[Tc[x].values for x in dets])
            tests.append(dict(dataset=ds, ref=ref, event=ev, metric=metric, test='Friedman', comparison=' vs '.join(dets), n=len(Tc), stat=fr.statistic, p=fr.pvalue))
        for det in dets:   # paired: subjects with values for both detectors
            if det == 'peak_anchored': continue
            pr = T[['peak_anchored', det]].dropna(); diff = pr['peak_anchored'] - pr[det]
            p = wilcoxon(pr['peak_anchored'], pr[det], zero_method='zsplit').pvalue if np.any(diff != 0) else 1.0
            tests.append(dict(dataset=ds, ref=ref, event=ev, metric=metric, test='Wilcoxon', comparison=f'proposed vs {det}', n=len(pr), stat=diff.median(), p=p))
S = pd.DataFrame(rows); Tt = pd.DataFrame(tests)
w = Tt.test == 'Wilcoxon'; ps = Tt.loc[w, 'p'].values; order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
for r, i in enumerate(order):
    run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
Tt.loc[w, 'p_holm'] = adj
S.to_csv(f'{PTH.RESULTS}/stats_subject_level.csv', index=False); Tt.to_csv(f'{PTH.RESULTS}/stats_tests.csv', index=False)
pd.set_option('display.width', 250)
print(S.pivot_table(index=['dataset', 'ref', 'metric', 'event'], columns='detector', values='mean').round(3).to_string())
print(Tt.round(5).to_string(index=False))
