# IMU-Based Gait Event Detection: Peak-Anchored Baseline Pipeline

Reference implementation of a peak-anchored heel-strike and toe-off detector
for shank-mounted inertial measurement units, with fixed-threshold and
finite-state-machine baselines for comparison.

## Dataset

Pacini Panebianco et al. (2020), *Foot platform and inertial wearable sensor
data for gait*, Mendeley Data v3.
DOI: [10.17632/92xd6g4xpk.3](https://doi.org/10.17632/92xd6g4xpk.3) (CC BY 4.0).

See `data/instructions.txt` for download steps. Raw data files are not
included in this repository.

## Setup

```
pip install -r requirements.txt
```

Python 3.9 or newer.

## Run

End-to-end evaluation on all trials (recommended detector and three
baselines, four matching tolerances):

```
python scripts/generate_results.py
```

Single-trial demonstration:

```
python scripts/run_pipeline.py --trial 1
```

## Outputs

`generate_results.py` writes to `results/`:

- `tables/method_comparison.csv` -- per-method per-tolerance metrics
- `tables/multi_tolerance_summary.json` -- detailed summary
- `tables/detection_summary.csv` -- recommended detector at +/-100 ms
- `tables/event_errors.csv` -- per-event timing errors
- `tables/trial_metrics.csv` -- per-trial metrics

`run_pipeline.py` prints detected event indices for the requested trial.
