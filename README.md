# An Interpretable Peak-Anchored Pipeline for Shank-IMU Gait Event Detection: Uncertainty Quantification and Cross-Dataset Benchmarking

Code and results for

> V. Kumar, D. K. Pratihar and M. J. Piran, "An Interpretable Peak-Anchored Pipeline for Shank-IMU Gait Event Detection: Uncertainty Quantification and Cross-Dataset Benchmarking," *IEEE Sensors Journal*, manuscript Sensors-113675-2026.

The detector is training-free. It anchors each gait cycle on the mid-swing (MSW) peak of the sagittal shank angular velocity and takes toe-off (TO) and heel-strike (HS) from the troughs before and after it. Thresholds are scaled to each recording by the interquartile range, and every event carries a Cramér-Rao timing-uncertainty bound. Each leg is processed independently, so the detector works with a sensor on one or both shanks.

The detector itself is `code/peak_anchored_detector.py`. Everything else reproduces the evaluation in the paper.

## Repository layout

| Folder | Content |
|---|---|
| `code/` | detector, re-implemented comparison methods, trained models and all evaluation scripts |
| `data/derived/` | force-platform reference events of Dataset 1 |
| `data/prep/` | extraction of the required channels of Dataset 2 |
| `results/` | summary results used in the paper (`numbers.json` holds every number quoted in the text) |
| `run_all.sh` | runs the whole pipeline in order |

## Environment

All results were computed on one PC (Intel Core i5-12400F, 16 GB RAM, Ubuntu 22.04) with Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2 and PyTorch 2.14.1 (CPU).

```
pip install -r requirements.txt
```

## Data

Both datasets are public and are not redistributed here.

* **Dataset 1**, Pacini Panebianco et al., *Foot platform and inertial wearable sensor data gait*, Mendeley Data v3, DOI [10.17632/92xd6g4xpk.3](https://doi.org/10.17632/92xd6g4xpk.3) (CC BY 4.0). Unzip it and point `DATASET1_DIR` to the folder that contains `IMU data` and `Synchronized Force Platform Data`.
* **Dataset 2**, Camargo et al., *J. Biomech.* 119 (2021) 110320, [epic.gatech.edu](https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/). Only the right-shank and right-foot IMUs, the force plates and the `gcRight` gait-cycle annotation are used. See `data/README.md` for the compact files the scripts read, and point `DATASET2_DIR` to them.

```
export DATASET1_DIR=/path/to/dataset1
export DATASET2_DIR=/path/to/dataset2_compact
```

## Reproducing the paper

`run_all.sh` runs every step. The table lists what each script produces.

| Script | Output | Used for |
|---|---|---|
| `dataset1_references.py` | `data/derived/dataset1_reference_events.csv` | Dataset 1 reference events (Sec. III-A) |
| `tree_models.py`, `cnn_model.py`, `train_loso.py` | `cache/ml_cache_{rf,hgb,cnn}.pkl` | random forest, gradient boosting and temporal CNN, leave-one-subject-out on Dataset 2 and transferred to Dataset 1 (Sec. III-B) |
| `evaluate.py <method>` | `results/<method>/` | detected and matched events of each method |
| `summarise.py` | `summary_all.csv`, `per_subject_all.csv` | Tables II and III, Figs. 2, 4 and 5 |
| `statistics_tests.py` | `stats_tests.csv`, `stats_subject_level.csv` | Friedman and Holm-corrected Wilcoxon tests (Table III) |
| `speed_analysis.py` | `speed_events.csv` | walking-speed analysis (Fig. 3) |
| `ablation.py` | `ablation.csv` | ablation (Table IV) |
| `msw_height_rule.py`, `heldout.py` | `heldout.json` | MSW height rule on development and held-out subjects (Sec. IV-B) |
| `phases_and_uncertainty.py` | `phase_durations.csv`, `sigma_all_events.csv` | gait-phase durations and timing uncertainty (Table V, Fig. 6) |
| `polarity_check.py` | `polarity_check.csv` | per-recording polarity rule (Sec. III-A) |
| `paper_numbers.py` | `numbers.json` | bootstrap confidence intervals and all numbers in the text |
| `run_time.py <stage>` | `cost.json` | run time on one CPU thread (Table III) |
| `make_figures.py` | `results/figures/` | Figs. 1–6 |

The leave-one-subject-out training runs one fold per call (`train_loso.py <rf|hgb|cnn> <subject>`), followed by `final` and `merge`. Method names accepted by `evaluate.py` are `peak_anchored`, `salarian2004`, `catalfamo2010`, `greene2010`, `aminian2002`, `fixed_threshold`, `rf`, `hgb` and `cnn`.

The comparison methods (Aminian et al. 2002, Salarian et al. 2004, Greene et al. 2010, Catalfamo et al. 2010) were re-implemented from their publications with their published parameters (`literature_methods.py`, `catalfamo2010.py`). Steps that could not be reproduced are listed in the Supplementary Material of the paper. `fixed_threshold.py` is the fixed-threshold rule, and `metrics.py` holds the event matching and bootstrap code shared by all scripts.

## Contact

Dilip Kumar Pratihar (dkpra@mech.iitkgp.ac.in), Vidyapati Kumar
