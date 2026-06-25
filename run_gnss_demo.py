"""
Standalone GNSS emergence demo — one chart, one story.
Loads Layer 1 parquet and produces gnss_emergence.png.
Run: uv run python run_gnss_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import matplotlib
matplotlib.use('Agg')

from anomaly import plot_gnss_emergence

asrs = pd.read_parquet("outputs/data/asrs_layer1.parquet")
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')
print(f"Loaded {len(asrs):,} records")

plot_gnss_emergence(asrs)
