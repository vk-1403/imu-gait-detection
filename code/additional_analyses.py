"""Additional analyses of the Supplementary Material (Section S8, Tables S17-S23) and the checks quoted in
Sections II-B and IV-C of the paper. Run after run_all.sh, as it reads results/per_subject_all.csv,
results/stats_tests.csv, results/matched_errors_sigma.csv and results/phase_durations.csv.

    python additional_analyses.py            # all parts
    python additional_analyses.py variants   # one part

Parts: tables (S18, S19, S22, S23), loa (S21), variants (S17), sigma (S20), exclusions (Section S8),
references (two references on the same steps, HS landmark switching). Output: results/additional/*.csv
"""
import paths as PTH
import os, sys, glob, inspect, warnings, numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks, argrelmin, argrelmax, firwin
sys.path.insert(0, PTH.CODE)
import metrics as M, ablation as AB, literature_methods as LIT
warnings.filterwarnings('ignore')
OUT = os.path.join(PTH.RESULTS, 'additional'); os.makedirs(OUT, exist_ok=True)
HELD = ['AB18', 'AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']
METHODS = ['peak_anchored', 'rf', 'hgb', 'cnn', 'salarian2004', 'catalfamo2010', 'greene2010', 'aminian2002', 'fixed_threshold']
DEG = 180 / np.pi
_data = None


def data():
    global _data
    if _data is None: _data = AB.load_all()
    return _data


def sign(g): return g if np.percentile(g, 99) > -np.percentile(g, 1) else -g
def rate(r): return r['fs'] if 'fs' in r else 1 / np.median(np.diff(r['t']))   # Dataset 2: from the time stamps, as in evaluate.py
def lowpass(g, fs, fc=15.0): b, a = butter(4, fc / (fs / 2)); return filtfilt(b, a, g)


# ---------------------------------------------------------------- tables from the per-subject results
def tables():
    P = pd.read_csv(f'{PTH.RESULTS}/per_subject_all.csv')
    def per_subject(det, ds, ref, ev, met, tol):
        q = P[(P.detector == det) & (P.dataset == ds) & (P.ref == ref) & (P.event == ev) & (P.tol == tol)]
        if met == 'f1': q = q[q.f1.notna()]
        return q.groupby('subject')[met].mean()
    held, tol50, worst = [], [], []
    for d in METHODS:
        h, t, w = dict(method=d), dict(method=d), dict(method=d)
        for ev in ['HS', 'TO']:
            r = per_subject(d, 'Camargo', 'FP', ev, 'recall', 100); f = per_subject(d, 'Camargo', 'gc', ev, 'f1', 100)
            h[f'{ev}_recall_dev'] = r[~r.index.isin(HELD)].mean(); h[f'{ev}_recall_held'] = r[r.index.isin(HELD)].mean()
            h[f'{ev}_f1_dev'] = f[~f.index.isin(HELD)].mean(); h[f'{ev}_f1_held'] = f[f.index.isin(HELD)].mean()
            t[f'D2_{ev}_mae_gcRight_ms'] = per_subject(d, 'Camargo', 'gc', ev, 'mae', 100).mean()
            for ds, lab in [('Pacini', 'D1'), ('Camargo', 'D2')]:
                t[f'{lab}_{ev}_recall_50ms'] = per_subject(d, ds, 'FP', ev, 'recall', 50).mean()
                r = per_subject(d, ds, 'FP', ev, 'recall', 100); w[f'{lab}_{ev}_mean'] = r.mean(); w[f'{lab}_{ev}_min'] = r.min()
        held.append(h); tol50.append(t); worst.append(w)
    pd.DataFrame(held).to_csv(f'{OUT}/tableS18_heldout_all_methods.csv', index=False)
    pd.DataFrame(tol50).to_csv(f'{OUT}/tableS19_recall_50ms_mae_gcright.csv', index=False)
    pd.DataFrame(worst).to_csv(f'{OUT}/tableS22_mean_worst_subject.csv', index=False)
    # Table S23: median paired difference (proposed minus other) with a bootstrap 95% interval, for the 80 tests
    T = pd.read_csv(f'{PTH.RESULTS}/stats_tests.csv'); W = T[T.test == 'Wilcoxon']; rng = np.random.default_rng(0); out = []
    Q = P[P.tol == 100]
    for _, r in W.iterrows():
        other = r.comparison.split(' vs ')[1]; q = Q[(Q.dataset == r.dataset) & (Q.ref == r.ref) & (Q.event == r.event)]
        q = q[q.f1.notna()] if r.metric == 'f1' else (q[q.f1.isna()] if r.ref == 'gc' else q)
        g = q.groupby(['detector', 'subject'])[r.metric].mean().unstack(0)
        prop = [c for c in g.columns if c.startswith('peak_anchored')][0]
        d = (g[prop] - g[other]).dropna().values; bs = np.median(rng.choice(d, (10000, len(d))), axis=1)
        out.append(dict(dataset=r.dataset, ref=r.ref, event=r.event, metric=r.metric, comparator=other, n=len(d),
                        median_diff=np.median(d), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), p_holm=r.p_holm))
    O = pd.DataFrame(out); O.to_csv(f'{OUT}/tableS23_effect_sizes.csv', index=False)
    better = ((O.metric != 'mae') & (O.median_diff > 0)) | ((O.metric == 'mae') & (O.median_diff < 0)); s = O.p_holm < 0.05
    print('tests', len(O), 'significant', int(s.sum()), 'for proposed', int((s & better).sum()), 'against', int((s & ~better).sum()))


# ---------------------------------------------------------------- Table S21: limits of agreement with repeated measures
def loa():
    def ba(df, val='err_ms', subj='subj'):
        d = df[[subj, val]].dropna(); g = d.groupby(subj)[val]; m = g.size(); n = len(m); N = m.sum(); mu = g.mean(); gm = d[val].mean()
        msw = ((d[val] - d[subj].map(mu)) ** 2).sum() / (N - n); msb = (m * (mu - gm) ** 2).sum() / (n - 1)
        n0 = (N ** 2 - (m ** 2).sum()) / ((n - 1) * N); sb2 = max((msb - msw) / n0, 0); sd = np.sqrt(sb2 + msw); sp = d[val].std(ddof=1)
        return dict(subjects=n, n=N, bias=gm, loa_pooled_lo=gm - 1.96 * sp, loa_pooled_hi=gm + 1.96 * sp, loa_rm_lo=gm - 1.96 * sd,
                    loa_rm_hi=gm + 1.96 * sd, sd_between=np.sqrt(sb2), sd_within=np.sqrt(msw))
    R = pd.read_csv(f'{PTH.RESULTS}/matched_errors_sigma.csv'); Ph = pd.read_csv(f'{PTH.RESULTS}/phase_durations.csv'); rows = []
    for (ds, ev), g in R.groupby(['ds', 'ev']): rows.append(dict(quantity=f'{ds} {ev} error', **ba(g)))
    for (ds, act, prm), g in Ph.groupby(['ds', 'act', 'param']): rows.append(dict(quantity=f'{ds} {act} {prm}', **ba(g)))
    pd.DataFrame(rows).to_csv(f'{OUT}/tableS21_limits_of_agreement.csv', index=False); print(pd.DataFrame(rows).round(1).to_string(index=False))


# ---------------------------------------------------------------- Table S17: adapted variants
def aminian_window(start):
    """Aminian et al. with the HS search window starting `start` seconds after mid-swing (published: 0.25 s)."""
    src = inspect.getsource(LIT.aminian2002).replace('def aminian2002(x, fs):', 'def _am(x, fs):').replace('m + int(0.25 * F)', f'm + int({start} * F)')
    assert f'int({start} * F)' in src; ns = dict(LIT.__dict__); exec(src, ns); return ns['_am']


def generic(x, fs, msw_mode='adaptive', hs_mode='proposed', to_mode='proposed'):
    """Mid-swing anchored detector with exchangeable rules (adaptive or 50 deg/s mid-swing rule, proposed or Salarian HS and TO rules)."""
    g = sign(np.asarray(x, float)); y = lowpass(g, fs); y = y - np.median(y); iqr = np.subtract(*np.percentile(y, [75, 25]))
    if msw_mode == 'adaptive': ms = find_peaks(y, prominence=0.3 * iqr, distance=int(0.45 * fs), height=0.3 * np.percentile(y, 99))[0]
    else: s = y * DEG; mx = argrelmax(s)[0]; mx = mx[s[mx] > 50]; ms = LIT._merge_peaks(mx, s[mx], int(0.5 * fs))
    lmin = argrelmin(y)[0]; raw_min = argrelmin(g)[0]; neg = find_peaks(-y, prominence=0.2 * iqr, distance=int(0.2 * fs))[0]
    taps = firwin(int(0.15 * fs) | 1, 30 / (fs / 2)); sf = filtfilt(taps, [1.0], g * DEG); mnf = argrelmin(sf)[0]; mnf = mnf[sf[mnf] < -20]
    hs, to = [], []
    for k, m in enumerate(ms):
        nxt = ms[k + 1] if k + 1 < len(ms) else len(y); prv = ms[k - 1] if k > 0 else 0
        c = lmin[(lmin > m) & (lmin < nxt) & (y[lmin] < 0)] if hs_mode == 'proposed' else raw_min[(raw_min > m) & (raw_min <= m + int(1.5 * fs))]
        if len(c): hs.append(c[0])
        c = neg[(neg < m) & (neg > prv)] if to_mode == 'proposed' else mnf[(mnf < m) & (mnf >= m - int(1.5 * fs))]
        if len(c): to.append(c[-1])
    return np.array(sorted(set(hs)), int), np.array(sorted(set(to)), int)


def score(f, tol=0.1):
    """Recall against the force platforms (both datasets) and F1 against gcRight (Dataset 2), mean over subjects."""
    pac, cam = data(); rows = []
    for ds, recs in [('D1', pac), ('D2', cam)]:
        for r in recs:
            fs = rate(r); t = r['t'] if 't' in r else np.arange(len(r['x'])) / fs
            hs, to = f(r['x'], fs); hs = t[np.asarray(hs, int)]; to = t[np.asarray(to, int)]
            refs = [('FP', 'HS', hs, r['hs'] if ds == 'D1' else r['fp_hs']), ('FP', 'TO', to, r['to'] if ds == 'D1' else r['fp_to'])]
            if ds == 'D2': refs += [('gc', 'HS', hs, r['gc_hs']), ('gc', 'TO', to, r['gc_to'])]
            for ref, ev, d, g in refs:
                if not len(g): continue
                nd = np.sum((d >= g.min() - 0.3) & (d <= g.max() + 0.3)) if ref == 'gc' else np.nan
                rows.append(dict(k=f'{ds}_{ev}_{ref}', s=r['subj'], a=r.get('act', 'lg'), tp=len(M.match(d, g, tol)), ng=len(g), nd=nd))
    R = pd.DataFrame(rows); out = {}
    for k, g in R.groupby('k'):
        q = g.groupby(['s', 'a'])[['tp', 'ng', 'nd']].sum(); rec = q.tp / q.ng
        if k.endswith('FP'): out[k[:-3] + '_recall'] = rec.groupby(level=0).mean().mean() * 100
        else: pr = q.tp / q.nd; out[k[:-3] + '_f1'] = (2 * pr * rec / (pr + rec)).fillna(0).groupby(level=0).mean().mean()
    out['lowest_recall'] = min(v for k, v in out.items() if k.endswith('recall')); return out


def variants():
    pac, _ = data(); grid = []
    for st in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:           # window start chosen on Dataset 1 only, by HS recall
        f = aminian_window(st); R = []
        for r in pac:
            t = np.arange(len(r['x'])) / r['fs']; hs, _ = f(r['x'], r['fs']); R.append((r['subj'], len(M.match(t[hs], r['hs'], 0.1)), len(r['hs'])))
        D = pd.DataFrame(R, columns=['s', 'tp', 'n']).groupby('s').sum(); grid.append(dict(start_s=st, D1_HS_recall=(D.tp / D.n).mean() * 100))
    G = pd.DataFrame(grid); G.to_csv(f'{OUT}/aminian_window_grid_dataset1.csv', index=False); best = G.loc[G.D1_HS_recall.idxmax(), 'start_s']
    print(G.round(1).to_string(index=False), '\nchosen start', best)
    V = {'Proposed': lambda x, fs: generic(x, fs),
         'Proposed, fixed 50 deg/s MSW threshold': lambda x, fs: generic(x, fs, 'fixed50'),
         'Proposed, HS rule of Salarian': lambda x, fs: generic(x, fs, hs_mode='salarian'),
         'Salarian 2004, published': LIT.salarian2004,
         'Salarian 2004, on the 15 Hz signal': lambda x, fs: LIT.salarian2004(lowpass(sign(np.asarray(x, float)), fs), fs),
         'Salarian 2004, HS rule of the proposed detector': lambda x, fs: generic(x, fs, 'fixed50', 'proposed', 'salarian'),
         'Aminian 2002, published (HS window from 0.25 s)': LIT.aminian2002,
         f'Aminian 2002, HS window from {best} s': aminian_window(best)}
    T = pd.DataFrame([dict(variant=n, **score(f)) for n, f in V.items()]); T.to_csv(f'{OUT}/tableS17_adapted_variants.csv', index=False)
    print(T.round(3).to_string(index=False))


# ---------------------------------------------------------------- Table S20: noise-injection check of the precision index
def sigma(n_events=300, n_mc=200, seed=0):
    rng = np.random.default_rng(seed); W = 2; x5 = np.arange(-W, W + 1.0); A = np.vstack([x5 * x5, x5, np.ones(5)]).T; AtAi = np.linalg.inv(A.T @ A); H = AtAi @ A.T
    def kernel(fs):                                                  # autocovariance of unit white noise after the zero-phase filter
        imp = np.zeros(401); imp[200] = 1; g = lowpass(imp, fs); r = np.correlate(g, g, 'full'); c = len(r) // 2
        return np.array([[r[c + i - j] for j in range(5)] for i in range(5)])
    rows = []
    for ds, recs in zip(['D1', 'D2'], data()):
        picks = []
        for r in recs:
            fs = rate(r); raw = sign(r['x'].copy()); hi, ti = AB.detect(raw, fs, dict(AB.BASE))
            picks += [(r, fs, raw, ev, k) for ev, idx in [('HS', hi), ('TO', ti)] for k in idx]
        cache = {}
        for i in rng.choice(len(picks), size=min(n_events, len(picks)), replace=False):
            r, fs, raw, ev, k = picks[i]
            if id(raw) not in cache: b, a = butter(4, 40 / (fs / 2)); cache[id(raw)] = (lowpass(raw, fs), np.std(raw - filtfilt(b, a, raw), ddof=1))
            y, s = cache[id(raw)]
            if k < W + 5 or k > len(y) - W - 6: continue
            sw = s / np.sqrt((fs / 2 - 40) / (fs / 2)); dt = 1000 / fs     # white-noise SD of the raw signal from the residual above 40 Hz
            c, b_, _ = H @ y[k - W:k + W + 1]
            s_eq2 = s_corr = np.nan                                  # flat-extremum rule of Section II-B
            if abs(c) >= 1e-3:
                g = np.array([b_ / (2 * c * c), -1 / (2 * c), 0])
                s_eq2 = np.sqrt(g @ (s * s * AtAi) @ g) * dt; s_corr = np.sqrt(g @ (H @ (kernel(fs) * sw * sw) @ H.T) @ g) * dt
            lo, hi_ = max(0, k - 200), min(len(raw), k + 200); seg = raw[lo:hi_]; kk = k - lo; v = []
            for _ in range(n_mc):
                yn = lowpass(seg + rng.normal(0, sw, len(seg)), fs); j = kk - 3 + np.argmin(yn[kk - 3:kk + 4]); c2, b2, _ = H @ yn[j - W:j + W + 1]
                if abs(c2) > 1e-12: v.append((j - kk - b2 / (2 * c2)) * dt)
            rows.append(dict(dataset=ds, event=ev, sigma_eq2_ms=s_eq2, sigma_with_C_ms=s_corr, simulated_sd_ms=np.std(v, ddof=1)))
    R = pd.DataFrame(rows); R.round(4).to_csv(f'{OUT}/tableS20_sigma_noise_injection_events.csv', index=False)
    R['ratio_eq2'] = R.sigma_eq2_ms / R.simulated_sd_ms; R['ratio_with_C'] = R.sigma_with_C_ms / R.simulated_sd_ms
    S = R.groupby(['dataset', 'event']).agg(events=('simulated_sd_ms', 'size'), sigma_eq2_ms=('sigma_eq2_ms', 'median'), sigma_with_C_ms=('sigma_with_C_ms', 'median'),
                                             simulated_sd_ms=('simulated_sd_ms', 'median'), ratio_eq2=('ratio_eq2', 'median'), ratio_with_C=('ratio_with_C', 'median'))
    S.to_csv(f'{OUT}/tableS20_sigma_noise_injection.csv'); print(S.round(2))


# ---------------------------------------------------------------- Section S8: excluded contacts and foot assignment
def exclusions():
    G0 = pd.read_csv(PTH.GT); c1 = G0.peakF >= 300; c2 = G0.stance_s >= 0.3; c3 = G0.stance_s <= 1.5; G = G0[c1 & c2 & c3]
    out = dict(D1_contacts=len(G0), D1_below_300N=int((~c1).sum()), D1_shorter_0p3s=int((~c2).sum()), D1_longer_1p5s=int((~c3).sum()), D1_kept=len(G))
    FS = 1000 / 3.5; det = {}
    for t in G.trial.unique():
        imu = sio.loadmat(f'{PTH.PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
        for side, s in [('R', 'rshank'), ('L', 'lshank')]:
            hi, ti = AB.detect(getattr(imu, s).gyr[:, 2] * np.pi / 180, FS, dict(AB.BASE)); det[(t, side)] = (hi / FS, ti / FS)
    for mode in ['assigned', 'either_shank']:                         # recall with the foot-IMU assignment and without any assignment
        for ev, col, k in [('HS', 'hs_s', 0), ('TO', 'to_s', 1)]:
            v = []
            for t, g in G.groupby('trial'):
                if mode == 'assigned': tp = sum(len(M.match(det[(t, sd)][k], g[g.side == sd][col].values, 0.1)) for sd in 'RL')
                else: tp = len(M.match(np.sort(np.concatenate([det[(t, 'R')][k], det[(t, 'L')][k]])), g[col].values, 0.1))
                v.append(tp / len(g))
            out[f'D1_{ev}_recall_{mode}'] = np.mean(v) * 100
    cnt = dict(total=0, summed_channel=0, not_right_foot=0, below_300N=0, outside_duration=0, near_edge=0, kept=0)
    for f in sorted(glob.glob(PTH.CAM + '/*_events.csv')):
        E = pd.read_csv(f); z = np.load(f.replace('_events.csv', '_signals.npz'))
        for k in z.files:
            t = z[k][:, 0]; e = E[(E.key == k.replace('__', '/')) & (E.src == 'fp')]
            cnt['total'] += len(e); cnt['summed_channel'] += int((e.plate == 'Combined').sum()); e = e[e.plate != 'Combined']
            r = e.rfoot_mid / e.rfoot_all_med; cnt['not_right_foot'] += int((r >= 0.5).sum()); e = e[r < 0.5]
            cnt['below_300N'] += int((e.peakF < 300).sum()); e = e[e.peakF >= 300]; d = e.t_off - e.t
            cnt['outside_duration'] += int(((d < 0.3) | (d > 1.5)).sum()); e = e[(d >= 0.3) & (d <= 1.5)]
            ed = ~((e.t > t[0] + 1) & (e.t_off < t[-1] - 1)); cnt['near_edge'] += int(ed.sum()); cnt['kept'] += int((~ed).sum())
    out.update({f'D2_{k}': v for k, v in cnt.items()})
    pd.Series(out).to_csv(f'{OUT}/sectionS8_exclusions_assignment.csv', header=['value']); print(pd.Series(out).round(1).to_string())


# ---------------------------------------------------------------- Section IV-C: two references on the same steps, HS switching
def references():
    _, cam = data(); rows = []; sw = []
    for fc in [15.0, 30.0]:
        P = dict(AB.BASE); P['fc'] = fc
        for r in cam:
            hi, ti = AB.detect(r['x'], rate(r), P); d = r['t'][hi]
            mf = {j: e for i, j, e in M.match(d, r['fp_hs'], 0.15)}; mg = {j: e for i, j, e in M.match(d, r['gc_hs'], 0.15)}
            rows += [dict(fc=fc, subj=r['subj'], e_fp=mf[j] * 1000, e_gc=mg[j] * 1000) for j in set(mf) & set(mg)]
            if r['act'] == 'treadmill' and len(mf) >= 10:
                e = np.array(list(mf.values())) * 1000; sw.append(dict(fc=fc, subj=r['subj'], n=len(e), late_share=(e > 20).mean(), late_share_35ms=(e > 35).mean()))
    D = pd.DataFrame(rows); D['gc_minus_fp'] = D.e_fp - D.e_gc; out = []
    for fc, g in D.groupby('fc'):
        s = g.groupby('subj')[['e_fp', 'gc_minus_fp']].mean()
        out.append(dict(fc=fc, n=len(g), gc_minus_fp_mean=g.gc_minus_fp.mean(), gc_minus_fp_sd_between=s.gc_minus_fp.std(), error_fp_mean=g.e_fp.mean(),
                        error_fp_sd_between=s.e_fp.std(), error_fp_sd_within=(g.e_fp - g.subj.map(s.e_fp)).std()))
    S = pd.DataFrame(sw)
    for o in out:
        q = S[S.fc == o['fc']]; o['treadmill_recordings'] = len(q); o['recordings_with_both_minima'] = int(((q.late_share > 0.1) & (q.late_share < 0.9)).sum())
        o['treadmill_HS_at_later_minimum'] = (q.late_share_35ms * q.n).sum() / q.n.sum()
    pd.DataFrame(out).to_csv(f'{OUT}/sectionIVC_references_and_switching.csv', index=False); print(pd.DataFrame(out).round(3).T)


if __name__ == '__main__':
    parts = dict(tables=tables, loa=loa, variants=variants, sigma=sigma, exclusions=exclusions, references=references)
    for name in (sys.argv[1:] or list(parts)): print('==', name); parts[name]()
