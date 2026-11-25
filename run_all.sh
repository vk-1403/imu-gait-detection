#!/usr/bin/env bash
# Reproduces every result of the paper. Set DATASET1_DIR and DATASET2_DIR first (see README.md).
set -e
cd "$(dirname "$0")/code"

python dataset1_references.py                         # Dataset 1 force-platform reference events

SUBJECTS="AB06 AB07 AB08 AB09 AB10 AB11 AB12 AB13 AB14 AB15 AB16 AB17 AB18 AB19 AB20 AB21 AB23 AB24 AB25 AB27 AB28 AB30"
for m in rf hgb cnn; do                               # trained models, leave-one-subject-out on Dataset 2
  for s in $SUBJECTS; do
    python train_loso.py $m $s
  done
  python train_loso.py $m final                       # trained on all of Dataset 2, applied to Dataset 1
  python train_loso.py $m merge
done

for d in peak_anchored salarian2004 catalfamo2010 greene2010 aminian2002 fixed_threshold rf hgb cnn; do
  python evaluate.py $d
done

python summarise.py
python statistics_tests.py
python speed_analysis.py
python phases_and_uncertainty.py
python polarity_check.py
python ablation.py
python heldout.py
python paper_numbers.py

for s in free train_hgb_HS train_hgb_TO train_rf_HS train_rf_TO time_hgb time_rf train_cnn time_cnn versions; do
  python run_time.py $s                               # run time on one CPU thread
done

python make_figures.py
