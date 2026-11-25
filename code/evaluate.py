"""Evaluates one method on Dataset 1 (force platforms) and Dataset 2 (force platforms and gcRight).
Usage: python evaluate.py <method>   (names in METHODS)
Writes per-event errors and per-subject summaries to results/<method>/"""
import paths as PTH
import os, sys, glob, pickle, numpy as np, pandas as pd, scipy.io as sio
import peak_anchored_detector as PAD, fixed_threshold as FT, metrics as M
import catalfamo2010 as CAT
import literature_methods as LIT

PAC = f'{PTH.PAC}'; PAC_FS = 1000 / 3.5
CAM = f'{PTH.CAM}'
TOLS = [25, 50, 100, 150]


def det_proposed(x, fs, ctx=None):
    d = PAD.detect(x, fs); return d['hs'], d['to']

def det_fixed(x, fs, ctx=None): return FT.detect(x, fs)
def det_cat(x, fs, ctx=None): return CAT.detect(x, fs)
def det_sal(x, fs, ctx=None): return LIT.salarian2004(x, fs)
def det_gre(x, fs, ctx=None): return LIT.greene2010(x, fs)
def det_ami(x, fs, ctx=None): return LIT.aminian2002(x, fs)

_ML = {}
def _trained(name):            # events of the trained models, computed by train_loso.py
    def f(x, fs, ctx=None):
        if name not in _ML: _ML[name] = pickle.load(open(f'{PTH.CACHE}/ml_cache_{name}.pkl', 'rb'))
        c = _ML[name].get(ctx, {}); return c.get('HS', np.array([], int)), c.get('TO', np.array([], int))
    return f

METHODS = dict(peak_anchored=det_proposed, salarian2004=det_sal, catalfamo2010=det_cat, greene2010=det_gre, aminian2002=det_ami,
               fixed_threshold=det_fixed, rf=_trained('rf'), hgb=_trained('hgb'), cnn=_trained('cnn'))
DETECTORS = METHODS


def polarity(g):
    return 1.0 if np.percentile(g, 99) > -np.percentile(g, 1) else -1.0


def run_pacini(det):
    G = pd.read_csv(f'{PTH.GT}')
    G = G[(G.peakF >= 300) & (G.stance_s >= 0.3) & (G.stance_s <= 1.5)]
    rows = []
    for t, g in G.groupby('trial'):
        imu = sio.loadmat(f'{PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
        for side, s in [('R', 'rshank'), ('L', 'lshank')]:
            x = getattr(imu, s).gyr[:, 2] * np.pi / 180; x = polarity(x) * x
            hs, to = det(x, PAC_FS, ctx=('Pacini', f'P{t:02d}', side)); hs = np.sort(np.asarray(hs)) / PAC_FS; to = np.sort(np.asarray(to)) / PAC_FS
            gs = g[g.side == side]
            for ev, dd, gt in [('HS', hs, gs.hs_s.values), ('TO', to, gs.to_s.values)]:
                for tol in TOLS:
                    m = M.match(dd, gt, tol / 1000)
                    errs = {i: e for i, j, e in m}
                    for i in range(len(gt)):
                        rows.append(dict(dataset='Pacini', subject=f'P{t:02d}', activity='levelground', ref='FP', side=side,
                                         event=ev, tol=tol, matched=i in errs, err_ms=errs.get(i, np.nan) * 1000))
    return pd.DataFrame(rows)


def run_camargo(det):
    rows, prec = [], []
    for f in sorted(glob.glob(CAM + '/*_events.csv')):
        s = os.path.basename(f)[:4]; E = pd.read_csv(f); z = np.load(f'{CAM}/{s}_signals.npz')
        tm = [k for k in z.files if k.startswith('treadmill')] or z.files
        pol = polarity(np.concatenate([np.nan_to_num(z[k][:, 2].astype(float)) for k in tm]))
        for k in z.files:
            key = k.replace('__', '/'); act = key.split('/')[0]; S = z[k]; t = S[:, 0].astype(float)
            x0 = np.nan_to_num(S[:, 2].astype(float)); x = polarity(x0) * x0; fs = 1 / np.median(np.diff(t))   # sign per recording (trial)
            hi, ti = det(x, fs, ctx=('Camargo', s, key)); hs = t[np.sort(np.asarray(hi, int))]; to = t[np.sort(np.asarray(ti, int))]
            e = E[E.key == key]
            fp = e[(e.src == 'fp') & (e.plate != 'Combined')].copy()
            fp['ratio'] = fp.rfoot_mid / fp.rfoot_all_med; fp['dur'] = fp.t_off - fp.t
            fp = fp[(fp.ratio < 0.5) & (fp.peakF >= 300) & (fp.dur >= 0.3) & (fp.dur <= 1.5) & (fp.t > t[0] + 1) & (fp.t_off < t[-1] - 1)]
            gch = np.sort(e[(e.src == 'gc') & (e.event == 'HS')].t.values); gct = np.sort(e[(e.src == 'gc') & (e.event == 'TO')].t.values)
            for ref, ev, dd, gt in [('FP', 'HS', hs, fp.t.values), ('FP', 'TO', to, fp.t_off.values), ('gc', 'HS', hs, gch), ('gc', 'TO', to, gct)]:
                for tol in TOLS:
                    m = M.match(dd, gt, tol / 1000); errs = {i: er for i, j, er in m}
                    for i in range(len(gt)):
                        rows.append(dict(dataset='Camargo', subject=s, activity=act, ref=ref, side='R', event=ev, tol=tol,
                                         matched=i in errs, err_ms=errs.get(i, np.nan) * 1000))
                    if ref == 'gc' and len(gt) > 1:   # precision against the continuous dataset annotation
                        lo, hi_ = gt.min() - 0.3, gt.max() + 0.3
                        nd = int(np.sum((dd >= lo) & (dd <= hi_)))
                        prec.append(dict(subject=s, activity=act, event=ev, tol=tol, n_det=nd, n_gt=len(gt), tp=len(m)))
    return pd.DataFrame(rows), pd.DataFrame(prec)


if __name__ == '__main__':
    name = sys.argv[1]; det = DETECTORS[name]
    out = f'{PTH.RESULTS}/{name}'; os.makedirs(out, exist_ok=True)
    P = run_pacini(det); P.to_csv(f'{out}/pacini_events.csv', index=False)
    if '--pacini-only' not in sys.argv:
        C, PR = run_camargo(det); C.to_csv(f'{out}/camargo_events.csv', index=False); PR.to_csv(f'{out}/camargo_precision.csv', index=False)
    print(name, 'done', len(P))
