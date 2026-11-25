"""Folder locations. Set the environment variables or edit the defaults below.

DATASET1_DIR  Pacini Panebianco release (folder with 'IMU data' and 'Synchronized Force Platform Data')
DATASET2_DIR  compact Camargo files (ABxx_signals.npz, ABxx_events.csv, treadmill_speed_*.csv), see data/README.md
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, 'code')
PAC = os.environ.get('DATASET1_DIR', os.path.join(ROOT, 'data', 'dataset1_raw'))
CAM = os.environ.get('DATASET2_DIR', os.path.join(ROOT, 'data', 'dataset2_compact'))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results'))
CACHE = os.environ.get('CACHE_DIR', os.path.join(ROOT, 'cache'))
FIG = os.path.join(RESULTS, 'figures')
GT = os.path.join(ROOT, 'data', 'derived', 'dataset1_reference_events.csv')
for _d in (RESULTS, CACHE, FIG): os.makedirs(_d, exist_ok=True)
