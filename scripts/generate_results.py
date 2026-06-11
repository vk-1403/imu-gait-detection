"""End-to-end evaluation of all detectors at four matching tolerances."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (
    Params,
    detect_events,
    detect_events_fsm,
    detect_events_fixed,
    gt_driven_match,
    metrics,
)

DATA_DIR = ROOT / "data" / "pacini_real"
TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FS = 200.0
TOLS_S = [0.05, 0.10, 0.15, 0.20]
ALIGN_TOL_S = 0.20

DETECTORS = {
    "peak_anchored": detect_events,
    "fsm": detect_events_fsm,
    "fixed_threshold": detect_events_fixed,
}


def best_match_offset(det, gt, search_s=4.0, tol_s=ALIGN_TOL_S, fs=FS):
    if len(det) == 0 or len(gt) == 0:
        return 0
    half = int(search_s * fs)
    tol = int(tol_s * fs)
    span = int(max(np.max(det), np.max(gt) + half) + half + tol + 1)
    a = np.zeros(span, dtype=np.float32)
    b = np.zeros(span, dtype=np.float32)
    a[np.clip(det.astype(int), 0, span - 1)] = 1.0
    b[np.clip(gt.astype(int), 0, span - 1)] = 1.0
    if tol > 0:
        kernel = np.ones(2 * tol + 1, dtype=np.float32)
        b = np.convolve(b, kernel, mode="same")
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-(len(b) - 1), len(a))
    mask = (lags >= -half) & (lags <= half)
    if not mask.any():
        return 0
    return int(lags[mask][int(np.argmax(corr[mask]))])


def trial_numbers():
    nums = set()
    for p in DATA_DIR.glob("pacini_T*_rshank.csv"):
        nums.add(int(p.stem.replace("pacini_T", "").split("_")[0]))
    return sorted(nums)


def main():
    trials = trial_numbers()
    if not trials:
        raise SystemExit(f"No trial CSVs in {DATA_DIR}. See data/instructions.txt.")

    results = {m: {t: {ev: {"tp": 0, "fp": 0, "fn": 0, "errors": []}
                       for ev in ("HS", "TO")}
                   for t in TOLS_S} for m in DETECTORS}
    err_rows, trial_rows = [], []

    for T in trials:
        rsh = pd.read_csv(DATA_DIR / f"pacini_T{T:02d}_rshank.csv")
        lsh = pd.read_csv(DATA_DIR / f"pacini_T{T:02d}_lshank.csv")
        gt_hs = np.union1d(np.flatnonzero(rsh["gt_HS"].values),
                           np.flatnonzero(lsh["gt_HS"].values))
        gt_to = np.union1d(np.flatnonzero(rsh["gt_TO"].values),
                           np.flatnonzero(lsh["gt_TO"].values))

        for name, fn in DETECTORS.items():
            out_r = fn(rsh["gyro_z_rad_s"].values, fs=FS, params=Params(fs=FS))
            out_l = fn(lsh["gyro_z_rad_s"].values, fs=FS, params=Params(fs=FS))
            det_hs = np.sort(np.concatenate([out_r["hs_idx"], out_l["hs_idx"]]))
            det_to = np.sort(np.concatenate([out_r["to_idx"], out_l["to_idx"]]))

            off_hs = best_match_offset(det_hs, gt_hs)
            off_to = best_match_offset(det_to, gt_to)
            gt_hs_a = gt_hs + off_hs
            gt_to_a = gt_to + off_to

            for tol_s in TOLS_S:
                m_hs = gt_driven_match(det_hs, gt_hs_a, FS, tol_s=tol_s)
                m_to = gt_driven_match(det_to, gt_to_a, FS, tol_s=tol_s)
                for key, m in (("HS", m_hs), ("TO", m_to)):
                    bucket = results[name][tol_s][key]
                    bucket["tp"] += m["tp"]
                    bucket["fp"] += m["fp"]
                    bucket["fn"] += m["fn"]
                    bucket["errors"].extend(m["errors_ms"].tolist())

            if name == "peak_anchored":
                m_hs100 = gt_driven_match(det_hs, gt_hs_a, FS, tol_s=0.10)
                m_to100 = gt_driven_match(det_to, gt_to_a, FS, tol_s=0.10)
                for e in m_hs100["errors_ms"]:
                    err_rows.append({"trial": T, "event": "HS",
                                     "error_ms": float(e),
                                     "sync_offset_ms": off_hs / FS * 1000})
                for e in m_to100["errors_ms"]:
                    err_rows.append({"trial": T, "event": "TO",
                                     "error_ms": float(e),
                                     "sync_offset_ms": off_to / FS * 1000})
                trial_rows.append({"trial": T,
                                   "sync_offset_ms": off_hs / FS * 1000,
                                   "n_gt_HS": len(gt_hs),
                                   "n_gt_TO": len(gt_to),
                                   "tp_HS": m_hs100["tp"],
                                   "tp_TO": m_to100["tp"]})

    pd.DataFrame(err_rows).to_csv(TABLES_DIR / "event_errors.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(TABLES_DIR / "trial_metrics.csv", index=False)

    summary = {}
    flat_rows = []
    for name in DETECTORS:
        summary[name] = {}
        for tol_s in TOLS_S:
            key = str(int(tol_s * 1000))
            summary[name][key] = {}
            for ev in ("HS", "TO"):
                b = results[name][tol_s][ev]
                m = metrics({"tp": b["tp"], "fp": b["fp"], "fn": b["fn"],
                             "errors_ms": np.asarray(b["errors"], dtype=float)})
                summary[name][key][ev] = m
                flat_rows.append({"method": name, "tol_ms": int(tol_s * 1000),
                                  "event": ev, **m})

    with open(TABLES_DIR / "multi_tolerance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(flat_rows).to_csv(TABLES_DIR / "method_comparison.csv", index=False)

    rows = []
    for ev in ("HS", "TO"):
        m = summary["peak_anchored"]["100"][ev]
        rows.append({"event": ev, "n_gt": m["tp"] + m["fn"], **m})
    pd.DataFrame(rows).to_csv(TABLES_DIR / "detection_summary.csv", index=False)

    print(f"Trials processed: {len(trials)}")
    print(f"Tables written to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
