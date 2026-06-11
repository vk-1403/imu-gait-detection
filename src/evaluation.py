"""Event matching and timing-error metrics."""
import numpy as np


def gt_driven_match(detected, ground_truth, fs, tol_s=0.10, fp_window_s=0.30):
    """Match detected events to ground truth within +/- tol_s.

    A detected event is counted as a false positive only if it falls within
    +/- fp_window_s of some ground-truth event (windowed false-positive
    convention for sparsely annotated reference data).
    """
    detected = np.asarray(detected, dtype=int)
    ground_truth = np.asarray(ground_truth, dtype=int)
    tol = int(tol_s * fs)
    used = np.zeros(len(detected), dtype=bool)
    errors, matched = [], []
    for gt in ground_truth:
        if len(detected) == 0:
            continue
        diffs = np.abs(detected - gt).astype(float)
        diffs[used] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= tol:
            errors.append((detected[j] - gt) / fs * 1000.0)
            matched.append(j)
            used[j] = True
    tp = len(matched)
    fp = 0
    if len(detected) and len(ground_truth):
        fp_w = int(fp_window_s * fs)
        for j in np.flatnonzero(~used):
            if int(np.min(np.abs(ground_truth - detected[j]))) <= fp_w:
                fp += 1
    fn = len(ground_truth) - tp
    return {"tp": tp, "fp": fp, "fn": fn, "errors_ms": np.array(errors)}


def metrics(match):
    e = match["errors_ms"]
    tp, fp, fn = match["tp"], match["fp"], match["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)
          if (precision + recall) > 0 else 0.0)
    sd = float(np.std(e, ddof=1)) if e.size > 1 else 0.0
    bias = float(np.mean(e)) if e.size else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mae_ms": float(np.mean(np.abs(e))) if e.size else float("nan"),
        "bias_ms": bias, "sd_ms": sd,
        "loa_lo": bias - 1.96 * sd if e.size > 1 else float("nan"),
        "loa_hi": bias + 1.96 * sd if e.size > 1 else float("nan"),
    }
