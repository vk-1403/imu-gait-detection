"""Shared evaluation utilities.

match(det, gt, tol): one-to-one matching, closest pairs first -> list of (gt_idx, det_idx, err_s)
Subject-level statistics: metrics are computed per subject first, then summarised across subjects
(mean, SD, 95% percentile bootstrap CI over subjects, 10,000 resamples, seed 0).
"""
import numpy as np, pandas as pd


def match(det, gt, tol):
    det = np.asarray(det, float); gt = np.asarray(gt, float)
    if len(det) == 0 or len(gt) == 0:
        return []
    pairs = []
    for i, g in enumerate(gt):
        lo, hi = np.searchsorted(det, [g - tol, g + tol])
        for j in range(lo, min(hi + 1, len(det))):
            e = det[j] - g
            if abs(e) <= tol:
                pairs.append((abs(e), i, j, e))
    pairs.sort()
    used_g, used_d, out = set(), set(), []
    for _, i, j, e in pairs:
        if i in used_g or j in used_d:
            continue
        used_g.add(i); used_d.add(j); out.append((i, j, e))
    return out


def summarise_errors(err_ms):
    e = np.asarray(err_ms, float)
    if len(e) == 0:
        return dict(n=0, bias=np.nan, sd=np.nan, mae=np.nan, loa_lo=np.nan, loa_hi=np.nan)
    b, s = e.mean(), e.std(ddof=1) if len(e) > 1 else 0.0
    return dict(n=len(e), bias=b, sd=s, mae=np.abs(e).mean(), loa_lo=b - 1.96 * s, loa_hi=b + 1.96 * s)


def boot_ci(values, n=10000, seed=0):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def subject_table(df, value_cols, by=('subject',)):
    """df has one row per subject (already aggregated). Returns mean, SD, bootstrap CI per column."""
    rows = []
    for c in value_cols:
        v = df[c].values.astype(float)
        lo, hi = boot_ci(v)
        rows.append(dict(metric=c, mean=np.nanmean(v), sd=np.nanstd(v, ddof=1), ci_lo=lo, ci_hi=hi, n_subjects=int(np.sum(~np.isnan(v)))))
    return pd.DataFrame(rows)
