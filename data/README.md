# Data

## Dataset 1 (Pacini Panebianco et al.)

Download the Mendeley Data release (DOI 10.17632/92xd6g4xpk.3) and set `DATASET1_DIR` to the folder with `IMU data` and `Synchronized Force Platform Data`. `code/dataset1_references.py` builds the force-platform reference events (`derived/dataset1_reference_events.csv`, included here).

## Dataset 2 (Camargo et al.)

`prep/extract_camargo_subset.py` reads the released 200 Hz parquet files and keeps only the columns used in the paper (right-shank and right-foot IMUs, force plates, `gcRight`). The evaluation scripts read one compact pair of files per subject from `DATASET2_DIR`.

* `ABxx_signals.npz` has one array per recording, named `<mode>__<trial>`, whose column 0 is time and column 2 the sagittal shank angular velocity.
* `ABxx_events.csv` has one row per reference event, with columns `key, src, event, t, t_off, plate, peakF, rfoot_mid, rfoot_all_med`.
  * `src = gc` rows are `gcRight` HS and TO.
  * `src = fp` rows are force-platform contacts: 20 N threshold above the median baseline of the plate, peak force of at least 300 N, duration 0.3–1.5 s, with `t` the onset (HS) and `t_off` the offset (TO). `rfoot_mid` and `rfoot_all_med` are the right-foot angular velocity during the contact, used to assign contacts to the right foot.
* `treadmill_speed_*.csv` holds the released belt-speed records of the treadmill trials.
