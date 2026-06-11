"""Run the recommended detector on a single trial and print event indices."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import detect_events, Params


def load_trial(data_dir, trial):
    rsh = data_dir / f"pacini_T{trial:02d}_rshank.csv"
    lsh = data_dir / f"pacini_T{trial:02d}_lshank.csv"
    if not (rsh.exists() and lsh.exists()):
        raise FileNotFoundError(f"Trial {trial} CSVs not found in {data_dir}")
    return pd.read_csv(rsh), pd.read_csv(lsh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "data" / "pacini_real")
    ap.add_argument("--fs", type=float, default=200.0)
    args = ap.parse_args()

    rsh, lsh = load_trial(args.data_dir, args.trial)
    params = Params(fs=args.fs)
    out_r = detect_events(rsh["gyro_z_rad_s"].values, fs=args.fs, params=params)
    out_l = detect_events(lsh["gyro_z_rad_s"].values, fs=args.fs, params=params)

    print(f"Trial {args.trial}")
    print(f"  Right shank: HS={len(out_r['hs_idx'])}  TO={len(out_r['to_idx'])}")
    print(f"  Left shank : HS={len(out_l['hs_idx'])}  TO={len(out_l['to_idx'])}")
    print(f"  Right HS indices: {out_r['hs_idx'][:10]} ...")
    print(f"  Left  HS indices: {out_l['hs_idx'][:10]} ...")


if __name__ == "__main__":
    main()
