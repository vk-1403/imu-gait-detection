"""
Extracts the channels of Dataset 2 (Camargo et al.) used in the paper from the released 200 Hz signals.

Usage: python extract_camargo_subset.py <folder 01_signals_200Hz> <output folder>
Reads only these columns from 01_signals_200Hz/ABxx/ABxx_<activity>_200Hz.parquet:
  ids         : subject, activity, trial, trial_name, time, mode
  IMUs        : imu_shank_* and imu_foot_* (right side, as recorded, no virtual IMUs)
  force plates: fp_* force-vector columns of every plate and treadmill plate
  gait cycle  : gc_* (HeelStrike / ToeOff phase of the release)
Writes one parquet file per subject to the output folder, plus columns_used.txt.
"""
import os, glob, sys
import pandas as pd
import pyarrow.parquet as pq

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

IDS = ['subject', 'activity', 'trial', 'trial_name', 'time', 'mode']


def wanted(cols):
    keep = [c for c in IDS if c in cols]
    keep += [c for c in cols if c.startswith('imu_shank_') or c.startswith('imu_foot_')]
    keep += [c for c in cols if c.startswith('gc_')]
    fp = [c for c in cols if c.startswith('fp_')]
    # force vectors only (drop CoP / moments) when the names make that possible
    fpv = [c for c in fp if ('_v' in c.lower() or 'force' in c.lower())]
    keep += fpv if fpv else fp
    return keep


files = sorted(glob.glob(os.path.join(SRC, 'AB*', 'AB*_200Hz.parquet')))
print(len(files), 'source files found under', SRC)
if not files:
    sys.exit('No files found under ' + SRC)

inventory = None
by_subject = {}
for f in files:
    cols = pq.ParquetFile(f).schema_arrow.names
    keep = wanted(cols)
    if inventory is None:
        inventory = keep
        with open(os.path.join(OUT, 'columns_used.txt'), 'w') as fh:
            fh.write('\n'.join(keep) + '\n\nALL fp_ columns in source:\n' +
                     '\n'.join(c for c in cols if c.startswith('fp_')) + '\n')
    d = pd.read_parquet(f, columns=keep)
    for c in d.columns:
        if d[c].dtype == 'float64':
            d[c] = d[c].astype('float32')
    subj = os.path.basename(f).split('_')[0]
    by_subject.setdefault(subj, []).append(d)
    print(f'{os.path.basename(f)}: {len(d):,} rows, {len(keep)} columns')

for subj, parts in by_subject.items():
    out = os.path.join(OUT, f'{subj}_shank_fp_gc.parquet')
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False, compression='zstd')
    print('wrote', out, f'{os.path.getsize(out)/1e6:.1f} MB')
print('Done:', OUT)
