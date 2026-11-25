"""Random forest (RF) and histogram-based gradient boosting (HGB) for HS/TO detection from the sagittal shank gyro.
Training labels: gcRight events of Dataset 2, leave-one-subject-out (22 folds). Dataset 1: model trained on all
22 subjects of Dataset 2, applied unchanged. Input: polarity-corrected sagittal gyro, low-pass 15 Hz, resampled to
100 Hz, divided by the trial IQR; features = 51-sample window (+/-0.25 s) of the signal and of its first difference.
Labels: positive within +/-20 ms of an event; negatives subsampled 5:1. Events: peaks of the predicted probability,
height 0.5, distance 0.40 s."""
import paths as PTH
import os, sys, glob, pickle, time, numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
CAM = f'{PTH.CAM}'; PAC = f'{PTH.PAC}'
FS2, W = 100.0, 25; rng = np.random.default_rng(0)


def prep(g, fs):
    b, a = butter(4, 15 / (fs / 2)); y = filtfilt(b, a, g)
    t_new = np.arange(0, len(y) / fs, 1 / FS2); y2 = np.interp(t_new, np.arange(len(y)) / fs, y)
    iqr = np.percentile(y2, 75) - np.percentile(y2, 25); return (y2 - np.median(y2)) / max(iqr, 1e-6)


def feats(y):
    p = np.pad(y, W, mode='edge'); d = np.diff(p, prepend=p[0])
    idx = np.arange(len(y))[:, None] + np.arange(2 * W + 1)[None, :]
    return np.hstack([p[idx], d[idx]]).astype(np.float32)


def polarity(g):
    return 1.0 if np.percentile(g, 99) > -np.percentile(g, 1) else -1.0


def load_camargo():
    data = {}
    for f in sorted(glob.glob(CAM + '/*_events.csv')):
        s = os.path.basename(f)[:4]; E = pd.read_csv(f); z = np.load(f'{CAM}/{s}_signals.npz')
        tm = [k for k in z.files if k.startswith('treadmill')]
        pol = polarity(np.concatenate([np.nan_to_num(z[k][:, 2].astype(float)) for k in tm]))
        trials = []
        for k in z.files:
            key = k.replace('__', '/'); S = z[k]; t = S[:, 0].astype(float); fs = 1 / np.median(np.diff(t))
            g0 = np.nan_to_num(S[:, 2].astype(float)); y = prep(polarity(g0) * g0, fs); t2 = t[0] + np.arange(len(y)) / FS2   # sign per recording (trial)
            e = E[(E.key == key) & (E.src == 'gc')]
            lab = {}
            for ev in ['HS', 'TO']:
                tt = e[e.event == ev].t.values; L = np.zeros(len(y), np.int8)
                for x in tt:
                    i = int(round((x - t2[0]) * FS2)); L[max(i - 2, 0):i + 3] = 1
                lab[ev] = L
            gt = e.t.values
            valid = (t2 >= gt.min() - 0.3) & (t2 <= gt.max() + 0.3) if len(gt) else np.zeros(len(y), bool)
            trials.append(dict(key=key, y=y, t0=t[0], fs=fs, n_orig=len(t), lab=lab, valid=valid))
        data[s] = trials; print('loaded', s, flush=True)
    return data


def train(model, X, y):
    if model == 'hgb':
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0)
    else:
        m = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=0)
    return m.fit(X, y)


def build(trials_list, ev, cap):
    Xs, ys = [], []
    for tr in trials_list:
        F = feats(tr['y']); L = tr['lab'][ev]; v = tr['valid']
        pos = np.flatnonzero(L & v); neg = np.flatnonzero((~L.astype(bool)) & v)
        neg = rng.choice(neg, size=min(len(neg), 5 * max(len(pos), 1)), replace=False) if len(neg) else neg
        idx = np.r_[pos, neg]; Xs.append(F[idx]); ys.append(L[idx])
    X = np.vstack(Xs); y = np.concatenate(ys)
    if len(y) > cap:
        k = rng.choice(len(y), cap, replace=False); X, y = X[k], y[k]
    return X, y


def to_events(prob):
    p, _ = find_peaks(prob, height=0.5, distance=int(0.40 * FS2)); return p


if __name__ == '__main__':
    model = sys.argv[1]; cap = 600000 if model == 'hgb' else 100000
    D = load_camargo(); subs = sorted(D); cache = {}
    for s in subs:
        t0 = time.time()
        tr_list = [tr for o in subs if o != s for tr in D[o]]
        for ev in ['HS', 'TO']:
            X, y = build(tr_list, ev, cap); m = train(model, X, y)
            for tr in D[s]:
                prob = m.predict_proba(feats(tr['y']))[:, 1]; p = to_events(prob)
                idx = np.clip(np.round(p / FS2 * tr['fs']).astype(int), 0, tr['n_orig'] - 1)
                cache.setdefault(('Camargo', s, tr['key']), {})[ev] = idx
        print(model, 'fold', s, f'{time.time()-t0:.0f}s', flush=True)
    # final model on all Camargo -> Pacini
    allt = [tr for o in subs for tr in D[o]]
    for ev in ['HS', 'TO']:
        X, y = build(allt, ev, cap); m = train(model, X, y)
        for f in sorted(glob.glob(PAC + '/Synchronized Force Platform Data/*.mat')):
            t = int(os.path.basename(f).split('_')[-1][:-4])
            imu = sio.loadmat(f'{PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
            for side, sname in [('R', 'rshank'), ('L', 'lshank')]:
                g = getattr(imu, sname).gyr[:, 2] * np.pi / 180; fs = 1000 / 3.5; g = polarity(g) * g
                y = prep(g, fs); prob = m.predict_proba(feats(y))[:, 1]; p = to_events(prob)
                idx = np.clip(np.round(p / FS2 * fs).astype(int), 0, len(g) - 1)
                cache.setdefault(('Pacini', f'P{t:02d}', side), {})[ev] = idx
    pickle.dump(cache, open(f'{PTH.CACHE}/ml_cache_{model}.pkl', 'wb')); print('done', model)
