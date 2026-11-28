"""Builds the compact Dataset 2 files read by the evaluation scripts, directly from the public release of
Camargo et al. (the unpacked ABxx subject folders with one .mat table per sensor and trial).

Usage: python build_compact_files.py <release folder> <output folder> [ABxx ...]

The release folder is searched recursively for ABxx/<session>/<activity>/imu/*.mat, so the three downloaded
parts can stay in their own folders. For every subject it writes
  ABxx_signals.npz          one float32 array per recording, named <activity>__<trial>, with the columns
                            time, right-shank gyroscope X, Y, Z (rad/s) and right-shank accelerometer X, Y, Z
  ABxx_events.csv           reference events of every recording (columns key, src, event, t, t_off, plate, peakF,
                            rfoot_mid, rfoot_all_med)
  treadmill_speed_ABxx.csv  belt speed of the treadmill trials, every 0.1 s

Reference events
  src = gc    HS and TO of the released gcRight annotation: the samples where the HeelStrike or ToeOff phase
              (0-100 %) restarts.
  src = fp    force-platform contacts from the vertical force (_vy) of every plate at 1000 Hz: onset (t) and offset
              (t_off) at a 20 N threshold, contacts of at least 0.2 s. peakF is the peak vertical force.
              rfoot_mid is the median right-foot angular velocity (norm of the 15 Hz low-pass filtered gyroscope)
              over the middle 40 % of the contact and rfoot_all_med its median over the recording. Their ratio
              assigns a contact to the right foot in the evaluation (ratio < 0.5), which also applies the
              300 N, 0.3-1.5 s and recording-edge criteria and leaves out the summed 'Combined' channel.
  src = cond  median belt speed of a treadmill trial (column t), kept for reference.
"""
import sys, os, io, glob, re
import numpy as np, pandas as pd, scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.signal import butter, filtfilt

FS, THR, MINC = 200.0, 20.0, 0.2
ACTIVITIES = ['levelground', 'ramp', 'stair', 'treadmill']
SHANK = ['Header', 'shank_Gyro_X', 'shank_Gyro_Y', 'shank_Gyro_Z', 'shank_Accel_X', 'shank_Accel_Y', 'shank_Accel_Z']
FOOT = ['foot_Gyro_X', 'foot_Gyro_Y', 'foot_Gyro_Z']
COLS = ['key', 'src', 'event', 't', 't_off', 'plate', 'peakF', 'rfoot_mid', 'rfoot_all_med']


def read_mat_table(path):
    """Reads a MATLAB table stored in a v7 .mat file into a DataFrame."""
    fws = sio.loadmat(path)['__function_workspace__'].tobytes()
    r = MatFile5Reader(io.BytesIO(b' ' * 116 + b'\x00' * 8 + fws[:4] + fws[8:]), squeeze_me=False)
    r.initialize_read(); r.mat_stream.seek(128); h, _ = r.read_var_header(); a = r.read_var_array(h, process=True)
    names, data = [], []
    for c in a[0, 0]['MCOS'][0]['arr'].ravel():
        if isinstance(c, np.ndarray) and c.dtype == object and c.size > 1:
            el = list(c.ravel())
            if all(isinstance(x, np.ndarray) and x.dtype.kind == 'U' and x.size for x in el): names.append([str(x.ravel()[0]) for x in el])
            elif all(isinstance(x, np.ndarray) and x.dtype.kind == 'f' for x in el): data.append(el)
    for d in data:
        for n in names:
            if len(n) == len(d): return pd.DataFrame({k: v.ravel() for k, v in zip(n, d)})
    raise ValueError('no table found in ' + path)


def phase_restarts(t, phase): return t[np.flatnonzero(np.diff(phase) < -50) + 1]


def trial(act_dir, tr, key):
    imu = read_mat_table(f'{act_dir}/imu/{tr}.mat'); ti = imu.Header.values; rows = []
    f = f'{act_dir}/gcRight/{tr}.mat'
    if os.path.exists(f):
        gc = read_mat_table(f); h = gc.Header.values
        rows += [dict(key=key, src='gc', event='HS', t=x) for x in phase_restarts(h, gc.HeelStrike.values)]
        rows += [dict(key=key, src='gc', event='TO', t=x) for x in phase_restarts(h, gc.ToeOff.values)]
    f = f'{act_dir}/fp/{tr}.mat'
    if os.path.exists(f):
        fp = read_mat_table(f); tf = fp.Header.values; b, a = butter(4, 15 / (FS / 2))
        foot = np.linalg.norm(filtfilt(b, a, imu[FOOT].values, axis=0), axis=1); foot_med = np.median(foot)
        for c in [c for c in fp.columns if c.endswith('_vy')]:
            fv = fp[c].values; on = (fv > THR).astype(int); rise = np.flatnonzero(np.diff(on) == 1) + 1; fall = np.flatnonzero(np.diff(on) == -1) + 1
            for r in rise:
                if not np.any(fall > r): continue
                e = fall[fall > r][0]; hs, to = tf[r], tf[e]
                if to - hs < MINC: continue
                mid = (ti >= hs + 0.3 * (to - hs)) & (ti <= hs + 0.7 * (to - hs))
                rows.append(dict(key=key, src='fp', event='contact', t=hs, t_off=to, plate=c[:-3], peakF=fv[r:e].max(),
                                 rfoot_mid=np.median(foot[mid]) if mid.any() else np.nan, rfoot_all_med=foot_med))
    speed = None; f = f'{act_dir}/conditions/{tr}.mat'
    if act_dir.endswith('treadmill') and os.path.exists(f):
        c = read_mat_table(f); rows.append(dict(key=key, src='cond', event='speed', t=np.median(c.Speed.values)))
        speed = pd.DataFrame(dict(key=key, t=np.round(c.Header.values[::100], 1), speed=c.Speed.values[::100]))
    return imu[SHANK].values.astype('float32'), rows, speed


def subject(sess, subj, out):
    sig, rows, speed = {}, [], []
    for act in ACTIVITIES:
        for f in sorted(glob.glob(glob.escape(f'{sess}/{act}/imu') + '/*.mat')):
            tr = os.path.basename(f)[:-4]
            try: S, r, sp = trial(f'{sess}/{act}', tr, f'{act}/{tr}')
            except Exception as e: print(subj, act, tr, 'skipped:', e); continue
            sig[f'{act}__{tr}'] = S; rows += r
            if sp is not None: sp.insert(0, 'subject', subj); speed.append(sp)
    np.savez_compressed(f'{out}/{subj}_signals.npz', **sig); pd.DataFrame(rows, columns=COLS).to_csv(f'{out}/{subj}_events.csv', index=False)
    if speed: pd.concat(speed).to_csv(f'{out}/treadmill_speed_{subj}.csv', index=False)
    print(subj, len(sig), 'recordings', len(rows), 'events', flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 3: sys.exit(__doc__)
    root, out, want = sys.argv[1], sys.argv[2], set(sys.argv[3:]); os.makedirs(out, exist_ok=True); done = set()
    for d, sub, _ in os.walk(root):
        subj = os.path.basename(os.path.dirname(d))
        if re.fullmatch(r'AB\d\d', subj) and any(a in sub for a in ACTIVITIES) and subj not in done and (not want or subj in want):
            done.add(subj); subject(d, subj, out)
