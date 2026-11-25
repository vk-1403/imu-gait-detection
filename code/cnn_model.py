"""Temporal convolutional network (CNN) for HS/TO detection from the sagittal shank gyro.
Same input, labels, validation and event extraction as tree_models.py, so the three trained models differ only in
the learner.
  input  : polarity-corrected sagittal gyro, low-pass 15 Hz, resampled to 100 Hz, (x - median)/IQR per trial
  labels : gcRight HS and TO of Dataset 2, positive within +/-20 ms of an event (two output channels)
  network: fully convolutional, non-causal; Conv1d(1->C, k=7) then 6 residual blocks of dilated Conv1d
           (k=5, dilation 1,2,4,8,16,32, ReLU, batch norm), Conv1d(C->2, k=1); receptive field +/-1.3 s
  train  : random 8 s windows, batch 32, Adam lr 1e-3, BCE with positive weight 4, masked to the annotated span,
           fixed number of steps; settings fixed in advance (no tuning on any test subject)
  events : peaks of the predicted probability, height 0.5, distance 0.40 s (as for RF and HGB)"""
import paths as PTH
import os, sys, glob, pickle, time, numpy as np, scipy.io as sio
import torch, torch.nn as nn
sys.path.insert(0, f'{PTH.CODE}')
from tree_models import load_camargo, prep, polarity, to_events, FS2, PAC

C, STEPS, WIN, BATCH = int(os.environ.get('CNN_C', 24)), int(os.environ.get('CNN_STEPS', 1500)), 800, 32
torch.set_num_threads(2)


class Block(nn.Module):
    def __init__(s, c, d):
        super().__init__(); s.conv = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d); s.bn = nn.BatchNorm1d(c)
    def forward(s, x): return x + torch.relu(s.bn(s.conv(x)))


class TCN(nn.Module):
    def __init__(s):
        super().__init__(); s.inp = nn.Conv1d(1, C, 7, padding=3)
        s.blocks = nn.Sequential(*[Block(C, d) for d in (1, 2, 4, 8, 16, 32)]); s.out = nn.Conv1d(C, 2, 1)
    def forward(s, x): return s.out(s.blocks(torch.relu(s.inp(x))))


def make_pool(trials):
    P = []
    for tr in trials:
        Y = np.stack([tr['lab']['HS'], tr['lab']['TO']]).astype(np.float32)
        if tr['valid'].sum() > WIN: P.append((tr['y'].astype(np.float32), Y, tr['valid'].astype(np.float32)))
    w = np.array([len(p[0]) for p in P], float); return P, w / w.sum()


def train(trials, seed):
    g = np.random.default_rng(seed); torch.manual_seed(seed)
    P, w = make_pool(trials); net = TCN(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0), reduction='none'); net.train()
    for step in range(STEPS):
        xb, yb, mb = [], [], []
        for k in g.choice(len(P), BATCH, p=w):
            y, Y, v = P[k]; i = g.integers(0, len(y) - WIN + 1)
            xb.append(y[i:i + WIN]); yb.append(Y[:, i:i + WIN]); mb.append(v[i:i + WIN])
        x = torch.from_numpy(np.stack(xb))[:, None]; Yt = torch.from_numpy(np.stack(yb)); M = torch.from_numpy(np.stack(mb))[:, None]
        l = (lossf(net(x), Yt) * M).sum() / (2 * M.sum() + 1e-6)
        opt.zero_grad(); l.backward(); opt.step()
    net.eval(); return net


def predict(net, y):
    with torch.no_grad(): return torch.sigmoid(net(torch.from_numpy(y.astype(np.float32))[None, None]))[0].numpy()


if __name__ == '__main__':
    D = load_camargo(); subs = sorted(D); cache = {}
    only = sys.argv[1:]                       # optional: subset of folds (for timing tests)
    for fi, s in enumerate(subs):
        if only and s not in only: continue
        t0 = time.time(); net = train([tr for o in subs if o != s for tr in D[o]], seed=fi)
        for tr in D[s]:
            pr = predict(net, tr['y'])
            for j, ev in enumerate(['HS', 'TO']):
                p = to_events(pr[j]); idx = np.clip(np.round(p / FS2 * tr['fs']).astype(int), 0, tr['n_orig'] - 1)
                cache.setdefault(('Camargo', s, tr['key']), {})[ev] = idx
        print('cnn fold', s, f'{time.time()-t0:.0f}s', flush=True)
    if only: pickle.dump(cache, open(f'{PTH.CACHE}/ml_cache_cnn_test.pkl', 'wb')); sys.exit()
    net = train([tr for o in subs for tr in D[o]], seed=99)
    for f in sorted(glob.glob(PAC + '/Synchronized Force Platform Data/*.mat')):
        t = int(os.path.basename(f).split('_')[-1][:-4])
        imu = sio.loadmat(f'{PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
        for side, sname in [('R', 'rshank'), ('L', 'lshank')]:
            g = getattr(imu, sname).gyr[:, 2] * np.pi / 180; fs = 1000 / 3.5; g = polarity(g) * g
            pr = predict(net, prep(g, fs))
            for j, ev in enumerate(['HS', 'TO']):
                p = to_events(pr[j]); idx = np.clip(np.round(p / FS2 * fs).astype(int), 0, len(g) - 1)
                cache.setdefault(('Pacini', f'P{t:02d}', side), {})[ev] = idx
    pickle.dump(cache, open(f'{PTH.CACHE}/ml_cache_cnn.pkl', 'wb')); print('done cnn', flush=True)
