"""Run time of every method (median of 20 runs on one 141 s recording of Dataset 1, one CPU thread)
and size of the trained models. Run the stages in order:
  free | train_hgb_HS | train_hgb_TO | train_rf_HS | train_rf_TO | time_hgb | time_rf | train_cnn | time_cnn | versions
Writes results/cost.json."""
import os, sys, json, time, glob, pickle, platform
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as PTH
os.environ.setdefault('PAC', PTH.PAC); os.environ.setdefault('CAM', PTH.CAM)
MODELS = PTH.CACHE + '/timing_models'; os.makedirs(MODELS, exist_ok=True)
OUT = PTH.RESULTS + '/cost.json'
stage = sys.argv[1]
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
FS = 1000 / 3.5
def signal():
    import evaluate as EA
    for f in sorted(glob.glob(os.environ['PAC'] + '/IMU data/Walking_*.mat')):
        imu = sio.loadmat(f, squeeze_me=True, struct_as_record=False)['Walking']; g = imu.rshank.gyr[:, 2] * np.pi / 180
        if len(g) == 40345: return EA.polarity(g) * g, os.path.basename(f)
    raise SystemExit('recording not found')
def tmed(fn, n=20):
    fn(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))
def save():
    json.dump(out, open(OUT, 'w'), indent=1)
if stage == 'free':
    import evaluate as EA
    x, rec = signal(); out.update(recording=rec, n_samples=len(x))
    for name in ['peak_anchored', 'salarian2004', 'catalfamo2010', 'greene2010', 'aminian2002', 'fixed_threshold']:
        out[name] = dict(ms=tmed(lambda: EA.DETECTORS[name](x, FS)), params=0); print(name, out[name], flush=True)
elif stage.startswith('train_') and stage != 'train_cnn':
    import tree_models as MB
    _, model, ev = stage.split('_'); cap = 600000 if model == 'hgb' else 100000
    D = MB.load_camargo(); allt = [tr for s in D for tr in D[s]]
    X, y = MB.build(allt, ev, cap); t0 = time.perf_counter(); m = MB.train(model, X, y)
    print(model, ev, 'trained in %.1f s' % (time.perf_counter() - t0), flush=True)
    if hasattr(m, 'n_jobs'): m.n_jobs = 1
    pickle.dump(m, open(f'{MODELS}/{model}_{ev}.pkl', 'wb'))
elif stage in ('time_hgb', 'time_rf'):
    import tree_models as MB
    model = stage[5:]; ms = {ev: pickle.load(open(f'{MODELS}/{model}_{ev}.pkl', 'rb')) for ev in ['HS', 'TO']}
    for ev in ms:
        if hasattr(ms[ev], 'n_jobs'): ms[ev].n_jobs = 1
    x, _ = signal()
    def run():
        y2 = MB.prep(x, FS); F = MB.feats(y2)
        for ev in ['HS', 'TO']: MB.to_events(ms[ev].predict_proba(F)[:, 1])
    if model == 'rf': nodes = int(sum(e.tree_.node_count for ev in ms for e in ms[ev].estimators_))
    else: nodes = int(sum(p.nodes.shape[0] for ev in ms for it in ms[ev]._predictors for p in it))
    out[model] = dict(ms=tmed(run, 10), params=nodes); print(model, out[model], flush=True)
elif stage == 'train_cnn':
    import torch; torch.set_num_threads(2)
    import tree_models as MB, cnn_model as CB
    D = MB.load_camargo(); allt = [tr for s in D for tr in D[s]]
    CB.STEPS = 300; t0 = time.perf_counter(); net = CB.train(allt, seed=0)   # inference cost does not depend on training length
    print('cnn trained in %.1f s' % (time.perf_counter() - t0), flush=True)
    torch.save(net, MODELS + '/cnn.pt')
elif stage == 'time_cnn':
    import torch; torch.set_num_threads(1)
    import tree_models as MB, cnn_model as CB
    net = torch.load(MODELS + '/cnn.pt', weights_only=False); net.eval(); x, _ = signal()
    def runc():
        pr = CB.predict(net, MB.prep(x, FS))
        for j in range(2): MB.to_events(pr[j])
    out['cnn'] = dict(ms=tmed(runc), params=int(sum(p.numel() for p in net.parameters()))); print('cnn', out['cnn'], flush=True)
elif stage == 'versions':
    import scipy, sklearn, torch
    cpu = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
    out['machine'] = dict(cpu=cpu, os=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                          scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__)
    print(out['machine'])
save()
