"""
Layer 1 runner: load data, explode anomaly field, SPC on top 5 categories,
Isolation Forest, 2x2 quadrant.
Run from project root: uv run python run_layer1.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # project root — needed for src.X imports inside modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode

from data_loader import load_and_merge_asrs
from spc import explode_anomaly_field, get_top_anomaly_categories, run_spc_pipeline, plot_spc_results
from anomaly import build_isolation_forest, build_2x2_quadrant


# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading ASRS data")
print("=" * 60)
asrs = load_and_merge_asrs("data/raw")

print(f"\nDataFrame shape: {asrs.shape}")
print(f"\nColumn sample (first 30):")
for c in asrs.columns[:30]:
    print(f"  {c}")

# Check key columns
key_cols = ['ACN', 'Time | Date', 'Events | Anomaly', 'Report 1 | Narrative',
            'Report 2 | Narrative', 'Aircraft 1 | Flight Phase',
            'Assessments | Primary Problem']
print("\nKey column availability:")
for c in key_cols:
    present = c in asrs.columns
    if present:
        null_pct = asrs[c].isna().mean() * 100
        print(f"  {'OK' if present else 'MISSING'} {c}  - null: {null_pct:.1f}%")
    else:
        print(f"  MISSING {c}  - NOT FOUND")

print(f"\nNarrative word count stats:")
print(asrs['narrative_word_count'].describe().to_string())

# ── 2. EXPLODE ANOMALY FIELD ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Exploding anomaly field")
print("=" * 60)

if 'Events | Anomaly' not in asrs.columns:
    print("ERROR: 'Events | Anomaly' column not found!")
    print("Available columns matching 'anomaly':")
    for c in asrs.columns:
        if 'anomaly' in c.lower() or 'event' in c.lower():
            print(f"  {c}")
    sys.exit(1)

asrs = explode_anomaly_field(asrs)

# Top categories
top_cats = get_top_anomaly_categories(asrs, top_n=15)
print(f"\nTop 15 anomaly categories by incident count:")
for cat, cnt in top_cats.items():
    pct = cnt / len(asrs) * 100
    print(f"  {cnt:5,} ({pct:5.1f}%)  {cat}")

TOP_5 = top_cats.head(5).index.tolist()
print(f"\nTop 5 selected for SPC: {TOP_5}")

# ── 3. SPC — CUSUM on top 5 categories ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Running SPC (CUSUM) pipeline")
print("=" * 60)

spc_results = {}
for cat in TOP_5:
    print(f"\nProcessing: {cat}")
    result = run_spc_pipeline(asrs, category_value=cat)
    spc_results[cat] = result

valid_results = {k: v for k, v in spc_results.items() if v is not None}
print(f"\n{len(valid_results)}/{len(TOP_5)} categories passed SPC minimum requirements")
for cat, res in valid_results.items():
    print(f"  {cat}: {len(res['alarms'])} CUSUM alarms")
    if res['alarms']:
        alarm_strs = [str(a)[:7] for a in res['alarms'][:5]]
        print(f"    First alarms: {', '.join(alarm_strs)}")

# Plot SPC
if valid_results:
    plot_spc_results(valid_results)

# ── 4. ISOLATION FOREST ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Isolation Forest")
print("=" * 60)

asrs, iso_model = build_isolation_forest(asrs)
print(f"\nIF score distribution:")
print(asrs['if_score'].describe().to_string())

# ── 5. 2×2 QUADRANT ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Building 2×2 risk quadrant")
print("=" * 60)

asrs = build_2x2_quadrant(asrs, spc_results)

# ── 6. SAVE ENRICHED DATASET ─────────────────────────────────────────────────
out = "outputs/data/asrs_layer1.parquet"
save_df = asrs.copy()
for col in save_df.select_dtypes(include=['object']).columns:
    save_df[col] = save_df[col].astype(str)
save_df.to_parquet(out, index=False)
print(f"\nLayer 1 enriched dataset saved to: {out}")
print(f"Final shape: {asrs.shape}")

# Print GNSS signal preview
print("\n" + "=" * 60)
print("GNSS / Navigation anomaly signal (monthly mentions in narratives):")
print("=" * 60)
# Narrow to actual spoofing/jamming/GPS-denial events, not generic nav errors
gnss_mask = asrs['full_narrative'].str.lower().str.contains(
    r'spoof|jamm|gps.{0,25}interfer|gnss.{0,25}interfer|gps.{0,25}denial|'
    r'gps.{0,25}unreliable|gps.{0,25}degrad|position.{0,25}spoof|'
    r'gps.{0,25}lost|navigation.{0,25}warn|gps.{0,25}alert',
    regex=True, na=False
)
gnss_monthly = (asrs[gnss_mask]
                .groupby(asrs[gnss_mask]['date'].dt.to_period('M'))['ACN'].count())
print(f"GNSS-related incidents: {gnss_mask.sum():,} ({gnss_mask.mean()*100:.1f}% of corpus)")
if not gnss_monthly.empty:
    print(gnss_monthly.to_string())

print("\nLayer 1 complete.")
