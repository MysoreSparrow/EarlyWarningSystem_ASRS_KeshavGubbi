"""
Layer 3 runner: rule-based precursor risk scorer.

Applies transparent, auditable risk scoring to all incidents.
No ML — every score component maps to a known human factors category.

Outputs:
  outputs/figures/precursor_risk_distribution.png
  outputs/data/layer3_high_risk_incidents.csv
  outputs/data/asrs_layer3.parquet

Run: uv run python run_layer3.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # project root — needed for src.X imports inside modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from risk_scorer import apply_risk_scorer, plot_risk_distribution, export_high_risk_incidents

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading enriched dataset")
print("=" * 60)
src = "outputs/data/asrs_layer2.parquet"
if not os.path.exists(src):
    src = "outputs/data/asrs_layer1.parquet"
    print(f"Layer 2 not found, falling back to {src}")
asrs = pd.read_parquet(src)
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')
print(f"Loaded {len(asrs):,} records  |  columns: {asrs.shape[1]}")
print(f"Quadrant breakdown:")
print(asrs['quadrant'].value_counts().to_string())

# ── 2. SCORE ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Applying rule-based risk scorer")
print("=" * 60)
asrs = apply_risk_scorer(asrs)

# Summary stats
print(f"\nPrecursor score distribution:")
print(asrs['precursor_score'].describe().to_string())

# Cross-tab: high risk by quadrant
print(f"\nHigh-risk incidents by quadrant:")
print(
    asrs.groupby('quadrant')['high_precursor_risk']
    .agg(['sum', 'count', 'mean'])
    .rename(columns={'sum': 'high_risk_count', 'count': 'total', 'mean': 'rate'})
    .assign(rate=lambda df: df['rate'].map('{:.1%}'.format))
    .to_string()
)

# Top 5 incidents overall
print(f"\nTop 5 highest-risk incidents (all quadrants):")
component_cols = [c for c in asrs.columns if c.startswith('component_')]
show_cols = ['ACN', 'date', 'quadrant', 'precursor_score'] + component_cols + ['Events | Anomaly']
show_cols = [c for c in show_cols if c in asrs.columns]
print(asrs.nlargest(5, 'precursor_score')[show_cols].to_string())

# ── 3. CHART ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Generating risk score distribution chart")
print("=" * 60)
plot_risk_distribution(asrs)

# ── 4. EXPORT HIGH-RISK CSV ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Exporting top 100 high-risk incidents")
print("=" * 60)
top100 = export_high_risk_incidents(asrs)
print(f"\nTop 10 from export:")
preview_cols = ['ACN', 'date', 'quadrant', 'precursor_score'] + component_cols[:3]
preview_cols = [c for c in preview_cols if c in top100.columns]
print(top100.head(10)[preview_cols].to_string())

# ── 5. SAVE PARQUET ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Saving Layer 3 parquet")
print("=" * 60)
out = "outputs/data/asrs_layer3.parquet"
save = asrs.copy()
for col in save.select_dtypes(include=['object']).columns:
    save[col] = save[col].astype(str)
save.to_parquet(out, index=False)
print(f"Saved: {out}  |  shape: {asrs.shape}")

print(f"\n{'='*60}")
print(f"LAYER 3 COMPLETE")
print(f"  Incidents scored : {len(asrs):,}")
print(f"  High-risk (top 10%): {asrs['high_precursor_risk'].sum():,}")
print(f"  Chart : outputs/figures/precursor_risk_distribution.png")
print(f"  CSV   : outputs/data/layer3_high_risk_incidents.csv")
print(f"  Data  : {out}")
print(f"{'='*60}")
