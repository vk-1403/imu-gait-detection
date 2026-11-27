"""Every number quoted in the paper and the supplementary material, computed from results/*.csv.
Writes results/numbers.json."""
import paths as PTH
import json, numpy as np, pandas as pd, sys
sys.path.insert(0, f'{PTH.CODE}'); import metrics as M
R = f'{PTH.RESULTS}'; N = {}
DETS = ['peak_anchored', 'rf', 'hgb', 'cnn', 'salarian2004', 'catalfamo2010', 'greene2010', 'aminian2002', 'fixed_threshold']
TF = ['peak_anchored', 'salarian2004', 'catalfamo2010', 'greene2010', 'aminian2002', 'fixed_threshold']

# ---- event counts
G = pd.read_csv(f'{PTH.GT}'); Gq = G[(G.peakF >= 300) & (G.stance_s >= 0.3) & (G.stance_s <= 1.5)]
N['PAC_NCONT_ALL'] = len(G); N['PAC_NCONT'] = len(Gq); N['PAC_NR'] = int((Gq.side == 'R').sum()); N['PAC_NL'] = int((Gq.side == 'L').sum())
C = pd.read_csv(f'{R}/peak_anchored/camargo_events.csv'); C100 = C[C.tol == 100]
N['CAM_NFP'] = int(len(C100[(C100.ref == 'FP') & (C100.event == 'HS')]))
N['CAM_NFP_ACT'] = C100[(C100.ref == 'FP') & (C100.event == 'HS')].groupby('activity').size().to_dict()
N['CAM_NGC_HS'] = int(len(C100[(C100.ref == 'gc') & (C100.event == 'HS')])); N['CAM_NGC_TO'] = int(len(C100[(C100.ref == 'gc') & (C100.event == 'TO')]))
N['CAM_NREC'] = int(C.groupby(['subject', 'activity']).size().shape[0])

# ---- subject-level metrics per dataset/terrain/tolerance from the per-event files
def subj_metrics(det, ds, ref, ev, tol, act=None):
    P = pd.read_csv(f'{R}/{det}/{"pacini" if ds == "Pacini" else "camargo"}_events.csv')
    g = P[(P.ref == ref) & (P.event == ev) & (P.tol == tol)]
    if act: g = g[g.activity == act]
    rec = g.groupby(['subject', 'activity']).matched.mean()
    mae = g[g.matched].assign(a=lambda d: d.err_ms.abs()).groupby(['subject', 'activity']).a.mean()
    rs = rec.groupby(level=0).mean(); ms = mae.groupby(level=0).mean()
    e = g[g.matched].err_ms; S = M.summarise_errors(e)
    lo, hi = M.boot_ci(rs.values)
    return dict(n=len(g), rec=rs.mean(), rec_lo=lo, rec_hi=hi, rec_min=rs.min(), mae=ms.mean(), bias=S['bias'], sd=S['sd'],
                loa_lo=S['loa_lo'], loa_hi=S['loa_hi'], med=np.median(e) if len(e) else np.nan)

def gc_f1(det, ev, tol, act=None):
    PR = pd.read_csv(f'{R}/{det}/camargo_precision.csv'); q = PR[(PR.event == ev) & (PR.tol == tol)]
    if act: q = q[q.activity == act]
    q = q.groupby(['subject', 'activity'])[['n_det', 'n_gt', 'tp']].sum(); q = q[q.n_gt > 0]
    pr = q.tp / q.n_det; rc = q.tp / q.n_gt; f1 = (2 * pr * rc / (pr + rc)).fillna(0)
    return dict(f1=f1.groupby(level=0).mean().mean(), prec=pr.groupby(level=0).mean().mean())

T = {}
for tol in [25, 50, 100, 150]:
    for ev in ['HS', 'TO']:
        T[f'Pacini_{ev}_{tol}'] = subj_metrics('peak_anchored', 'Pacini', 'FP', ev, tol)
for act in ['levelground', 'ramp', 'stair', 'treadmill', None]:
    for ev in ['HS', 'TO']:
        k = f'Camargo_{act or "all"}_{ev}_100'
        T[k] = subj_metrics('peak_anchored', 'Camargo', 'FP', ev, 100, act); T[k].update(gc_f1('peak_anchored', ev, 100, act))
N['T'] = T

# ---- Pacini literature-format medians (events matched within +/-150 ms)
E = pd.read_csv(f'{R}/matched_errors_sigma.csv')
for ev in ['HS', 'TO']:
    g = E[(E.ds == 'Pacini') & (E.ev == ev)].err_ms
    N[f'PAC_{ev}_MED'], N[f'PAC_{ev}_Q1'], N[f'PAC_{ev}_Q3'] = float(g.median()), float(g.quantile(.25)), float(g.quantile(.75))
for ev in ['HS', 'TO']:
    g = E[(E.ds == 'Camargo') & (E.ev == ev)].err_ms
    N[f'CAM_{ev}_MED'] = float(g.median())

# ---- baselines (subject level, +/-100 ms)
S = pd.read_csv(f'{R}/stats_subject_level.csv'); TT = pd.read_csv(f'{R}/stats_tests.csv')
B = {}
for det in DETS:
    b = {}
    for ds, ref, ev, met in [('Pacini', 'FP', 'HS', 'recall'), ('Pacini', 'FP', 'TO', 'recall'), ('Camargo', 'FP', 'HS', 'recall'),
                             ('Camargo', 'FP', 'TO', 'recall'), ('Camargo', 'gc', 'HS', 'f1'), ('Camargo', 'gc', 'TO', 'f1'),
                             ('Pacini', 'FP', 'HS', 'mae'), ('Pacini', 'FP', 'TO', 'mae'), ('Camargo', 'FP', 'HS', 'mae'), ('Camargo', 'FP', 'TO', 'mae')]:
        r = S[(S.dataset == ds) & (S.ref == ref) & (S.event == ev) & (S.metric == met) & (S.detector == det)].iloc[0]
        key = f'{ds}_{ref}_{ev}_{met}'; b[key] = dict(mean=r['mean'], lo=r.ci_lo, hi=r.ci_hi, n=int(r.n_subj))
        if det != 'peak_anchored':
            t = TT[(TT.dataset == ds) & (TT.ref == ref) & (TT.event == ev) & (TT.metric == met) & (TT.comparison == f'proposed vs {det}')].iloc[0]
            b[key].update(p=t.p, p_holm=t.p_holm, diff=t.stat)
    b['worst'] = min(b[k]['mean'] for k in b if k.endswith('_recall'))
    B[det] = b
N['B'] = B
N['N_WILCOXON'] = int((TT.test == 'Wilcoxon').sum())
N['FRIEDMAN_PMAX'] = float(TT[TT.test == 'Friedman'].p.max())
N['WC'] = {d: B[d]['worst'] for d in DETS}

# ---- phases
Ph = pd.read_csv(f'{R}/phase_durations.csv'); PH = {}
for (ds, act, prm), g in Ph.groupby(['ds', 'act', 'param']):
    S_ = M.summarise_errors(g.err_ms); PH[f'{ds}_{act}_{prm}'] = dict(ref=g.ref_ms.mean(), **S_)
N['PH'] = PH

# ---- sigma_t
A = pd.read_csv(f'{R}/sigma_all_events.csv'); SG = {}
for (ds, ev), g in A.groupby(['ds', 'ev']):
    SG[f'{ds}_{ev}'] = dict(n=len(g), med=g.sigma_ms.median(), p95=g.sigma_ms.quantile(.95))
from scipy.stats import spearmanr
for (ds, ev), g in E.dropna().groupby(['ds', 'ev']):
    SG[f'{ds}_{ev}'].update(rho=spearmanr(g.sigma_ms, g.err_ms.abs()).statistic, med_abs_err=g.err_ms.abs().median(), n_matched=len(g))
N['SG'] = SG; N['SG_NEV_PAC'] = int(A[A.ds == 'Pacini'].shape[0]); N['SG_NEV_CAM'] = int(A[A.ds == 'Camargo'].shape[0])

# ---- ablation
N['ABL'] = pd.read_csv(f'{R}/ablation.csv').set_index('variant').to_dict(orient='index')

# ---- sign check
Sg = pd.read_csv(f'{R}/polarity_check.csv'); Sg['tie'] = (Sg.f1_pos - Sg.f1_neg).abs() < 0.05; x = Sg[~Sg.tie]
N['SIGN_N'] = len(x); N['SIGN_TIES'] = int(Sg.tie.sum()); N['SIGN_OK'] = int((x.trial_rule == x.best).sum()); N['SIGN_SUBJ_OK'] = int((x.subject_rule == x.best).sum())
# ---- development (AB06-AB17) versus held-out (AB18-AB30) subjects of Dataset 2, proposed detector, +/-100 ms
_P = pd.read_csv(f'{R}/per_subject_all.csv')
_P = _P[(_P.detector == 'peak_anchored') & (_P.dataset == 'Camargo') & (_P.tol == 100)].copy()
_P['grp'] = np.where(_P.subject <= 'AB17', 'dev', 'held')
N['HO'] = {}
for (ref, ev), g in _P.groupby(['ref', 'event']):
    s_ = g.groupby(['grp', 'subject'])[['recall', 'f1']].mean().groupby('grp').mean()
    for grp in ('dev', 'held'):
        N['HO'][f'{grp}_{ref}_{ev}_recall'] = float(s_.loc[grp, 'recall'])
        if ref == 'gc': N['HO'][f'{grp}_{ref}_{ev}_f1'] = float(s_.loc[grp, 'f1'])
json.dump(N, open(f'{PTH.RESULTS}/numbers.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in N.items() if k not in ('T', 'B', 'PH', 'SG', 'ABL')}, indent=0, default=float))
