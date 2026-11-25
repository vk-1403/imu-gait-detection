"""Leave-one-subject-out training of the trained models, one fold per call.
Usage: python train_loso.py <rf|hgb|cnn> <subject|final|merge>
  <subject>  train on the other 21 subjects of Dataset 2 and detect events of this subject
  final      train on all 22 subjects of Dataset 2 and detect events of Dataset 1
  merge      collect all folds into cache/ml_cache_<model>.pkl (read by evaluate.py)"""
import paths as PTH
import os, sys, glob, pickle, time, numpy as np, scipy.io as sio
sys.path.insert(0, f'{PTH.CODE}'); os.makedirs(f'{PTH.CACHE}/parts', exist_ok=True)
import tree_models as MB
model, what = sys.argv[1], sys.argv[2]
if what == 'merge':
    cache = {}
    for f in sorted(glob.glob(f'{PTH.CACHE}/parts/{model}_*.pkl')):
        for k, v in pickle.load(open(f, 'rb')).items(): cache.setdefault(k, {}).update(v)
    pickle.dump(cache, open(f'{PTH.CACHE}/ml_cache_{model}.pkl', 'wb')); print('merged', model, len(cache)); sys.exit()
D = MB.load_camargo(); subs = sorted(D)
if model == 'cnn': import cnn_model as CB
def pacini(fn):
    out = {}
    for f in sorted(glob.glob(MB.PAC + '/Synchronized Force Platform Data/*.mat')):
        t = int(os.path.basename(f).split('_')[-1][:-4])
        imu = sio.loadmat(f'{MB.PAC}/IMU data/Walking_{t}.mat', squeeze_me=True, struct_as_record=False)['Walking']
        for side, sname in [('R', 'rshank'), ('L', 'lshank')]:
            g = getattr(imu, sname).gyr[:, 2] * np.pi / 180; fs = 1000 / 3.5; g = MB.polarity(g) * g
            for ev, p in fn(MB.prep(g, fs)).items():
                idx = np.clip(np.round(p / MB.FS2 * fs).astype(int), 0, len(g) - 1)
                out.setdefault(('Pacini', f'P{t:02d}', side), {})[ev] = idx
    return out
if what == 'final':
    allt = [tr for o in subs for tr in D[o]]; cache = {}
    if model == 'cnn':
        net = CB.train(allt, seed=99)
        cache = pacini(lambda y: dict(zip(['HS', 'TO'], [MB.to_events(r) for r in CB.predict(net, y)])))
    else:
        cap = 600000 if model == 'hgb' else 100000
        for ev in ['HS', 'TO']:
            X, y = MB.build(allt, ev, cap); m = MB.train(model, X, y)
            for k, v in pacini(lambda yy: {ev: MB.to_events(m.predict_proba(MB.feats(yy))[:, 1])}).items(): cache.setdefault(k, {}).update(v)
    pickle.dump(cache, open(f'{PTH.CACHE}/parts/{model}_final.pkl', 'wb')); print(model, 'final done', flush=True); sys.exit()
for s in what.split(','):
    t0 = time.time(); cache = {}; tr_list = [tr for o in subs if o != s for tr in D[o]]
    if model == 'cnn':
        net = CB.train(tr_list, seed=subs.index(s))
        for tr in D[s]:
            pr = CB.predict(net, tr['y'])
            for j, ev in enumerate(['HS', 'TO']):
                p = MB.to_events(pr[j]); cache.setdefault(('Camargo', s, tr['key']), {})[ev] = np.clip(np.round(p / MB.FS2 * tr['fs']).astype(int), 0, tr['n_orig'] - 1)
    else:
        cap = 600000 if model == 'hgb' else 100000
        for ev in ['HS', 'TO']:
            X, y = MB.build(tr_list, ev, cap); m = MB.train(model, X, y)
            for tr in D[s]:
                p = MB.to_events(m.predict_proba(MB.feats(tr['y']))[:, 1])
                cache.setdefault(('Camargo', s, tr['key']), {})[ev] = np.clip(np.round(p / MB.FS2 * tr['fs']).astype(int), 0, tr['n_orig'] - 1)
    pickle.dump(cache, open(f'{PTH.CACHE}/parts/{model}_{s}.pkl', 'wb')); print(model, 'fold', s, f'{time.time()-t0:.0f}s', flush=True)
