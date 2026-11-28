# Data

## Dataset 1 (Pacini Panebianco et al.)

Download the Mendeley Data release (DOI 10.17632/92xd6g4xpk.3) and set `DATASET1_DIR` to the folder with `IMU data` and `Synchronized Force Platform Data`. `code/dataset1_references.py` builds the force-platform reference events (`derived/dataset1_reference_events.csv`, included here).

## Dataset 2 (Camargo et al.)

Download and unpack the three parts of the release. Each subject folder `ABxx/<session>/<activity>/` holds one MATLAB table per sensor and trial (`imu`, `fp`, `gcRight`, `conditions`). The compact files that the evaluation scripts read are built with one command.

```
python prep/build_compact_files.py /path/to/camargo_release /path/to/dataset2_compact
export DATASET2_DIR=/path/to/dataset2_compact
```

The release folder is searched recursively, so the three parts can stay in their own folders. Subjects can be listed after the two folders to convert only some of them. The script needs NumPy, SciPy and pandas, and it takes about 20 s per subject. It writes three files per subject.

* `ABxx_signals.npz` has one array per recording, named `<activity>__<trial>`. Column 0 is time, columns 1 to 3 the right-shank angular velocity (X, Y, Z) and columns 4 to 6 the right-shank acceleration. Column 2 is the sagittal angular velocity used by all detectors.
* `ABxx_events.csv` has one row per reference event, with columns `key, src, event, t, t_off, plate, peakF, rfoot_mid, rfoot_all_med`.
  * `src = gc` rows are the `gcRight` HS and TO, taken where the released HeelStrike or ToeOff phase restarts.
  * `src = fp` rows are force-platform contacts from the vertical force of every plate at 1000 Hz, with a 20 N threshold and a minimum duration of 0.2 s. `t` is the onset (HS), `t_off` the offset (TO) and `peakF` the peak force. `rfoot_mid` is the median right-foot angular velocity (norm of the 15 Hz low-pass filtered gyroscope) over the middle 40 % of the contact, and `rfoot_all_med` its median over the recording.
  * `src = cond` rows give the median belt speed of a treadmill trial.
* `treadmill_speed_ABxx.csv` holds the released belt speed of the treadmill trials every 0.1 s.

The evaluation scripts then keep the contacts with `rfoot_mid / rfoot_all_med < 0.5` (right foot in stance), a peak force of at least 300 N, a duration of 0.3 to 1.5 s and at least 1 s from both ends of the recording. The summed `Combined` channel is left out.

The script was checked against the compact files used for the paper. For all 22 subjects it reproduces every reference event and belt-speed record, and the signals to single-precision rounding.
