"""
Layer 2 runner: BERTopic on full ASRS corpus.
Loads Layer 1 output, fits BERTopic, tracks topics over time,
identifies GNSS spoofing topic cluster, saves figures + enriched parquet.

Run from project root: uv run python run_layer2.py
Expected runtime: 10-20 min on CPU (embedding 43k docs + UMAP + HDBSCAN).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # project root — needed for src.X imports inside modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

from topics import (
    run_bertopic,
    find_gnss_topic_ids,
    get_topic_summary,
    plot_topic_landscape,
    plot_gnss_timeline,
    plot_red_quadrant_topics,
    plot_topic_heatmap,
)

LAYER1_PATH = "outputs/data/asrs_layer1.parquet"
OUT_FIGURES = "outputs/figures"
OUT_DATA = "outputs/data"
os.makedirs(OUT_FIGURES, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)


# ── 1. LOAD LAYER 1 OUTPUT ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Layer 1 enriched dataset")
print("=" * 60)
asrs = pd.read_parquet(LAYER1_PATH)
# Restore object dtypes that were stringified for parquet
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')
print(f"Loaded {len(asrs):,} records  |  {asrs['date'].min().date()} to "
      f"{asrs['date'].max().date()}")
print(f"Quadrant counts:\n{asrs['quadrant'].value_counts().to_string()}")


# ── 2. PREPARE DOCUMENTS ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preparing documents")
print("=" * 60)
MIN_WORDS = 30
mask = (
    asrs['full_narrative'].notna() &
    (asrs['narrative_word_count'] >= MIN_WORDS) &
    asrs['date'].notna()
)
corpus = asrs[mask].copy()
print(f"Documents with >={MIN_WORDS} words and valid date: {len(corpus):,}")
print(f"Dropped {len(asrs) - len(corpus):,} short/undated records")

docs = corpus['full_narrative'].tolist()
timestamps = corpus['date'].tolist()
print(f"\nDate range in corpus: {corpus['date'].min().date()} to "
      f"{corpus['date'].max().date()}")


# ── 3. FIT BERTOPIC ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Fitting BERTopic (this takes ~10-20 min on CPU)")
print("=" * 60)
topic_model, topics, topics_over_time_df = run_bertopic(
    docs=docs,
    timestamps=timestamps,
    nr_topics=40,
    min_cluster_size=30,
)

# Attach topic IDs back to the corpus rows
corpus = corpus.copy()
corpus['topic_id'] = topics
topic_info = topic_model.get_topic_info()
topic_labels = topic_info.set_index('Topic')['Name'].to_dict()
corpus['topic_label'] = corpus['topic_id'].map(topic_labels)

# Merge topic columns back to full asrs
asrs = asrs.merge(
    corpus[['topic_id', 'topic_label']],
    left_index=True, right_index=True, how='left',
)
asrs['topic_id'] = asrs['topic_id'].fillna(-2).astype(int)

topic_summary = get_topic_summary(topic_model, top_n=20)
print("\nTop 20 topics:")
print(topic_summary.to_string(index=False))
topic_summary.to_csv(os.path.join(OUT_DATA, "layer2_topic_summary.csv"), index=False)


# ── 4. IDENTIFY GNSS TOPICS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Identifying GNSS / spoofing topic clusters")
print("=" * 60)
gnss_topic_ids = find_gnss_topic_ids(topic_model)
print(f"GNSS-related topics: {gnss_topic_ids}")
for tid in gnss_topic_ids:
    words = topic_model.get_topic(tid)
    kw = ', '.join(f"{w}({s:.3f})" for w, s in words[:10]) if words else "n/a"
    count = (corpus['topic_id'] == tid).sum()
    print(f"  Topic {tid}  |  n={count:,}  |  {kw}")


# ── 5. BUILD GNSS MONTHLY SERIES (Layer 1 regex signal) ──────────────────────
print("\n" + "=" * 60)
print("STEP 5: Rebuilding GNSS regex signal for comparison")
print("=" * 60)
gnss_regex = (
    r'spoof|jamm|gps.{0,25}interfer|gnss.{0,25}interfer|gps.{0,25}denial|'
    r'gps.{0,25}unreliable|gps.{0,25}degrad|position.{0,25}spoof|'
    r'gps.{0,25}lost|navigation.{0,25}warn|gps.{0,25}alert'
)
gnss_mask = asrs['full_narrative'].astype(str).str.lower().str.contains(
    gnss_regex, regex=True, na=False
)
gnss_monthly = (
    asrs[gnss_mask]
    .groupby(asrs[gnss_mask]['date'].dt.to_period('M'))['ACN']
    .count()
)
print(f"GNSS incidents: {gnss_mask.sum():,}")
print(f"Monthly range: {gnss_monthly.min()} to {gnss_monthly.max()}")
avg_pre2023 = gnss_monthly[gnss_monthly.index < '2023'].mean()
avg_post2023 = gnss_monthly[gnss_monthly.index >= '2023'].mean()
print(f"Average pre-2023: {avg_pre2023:.1f}/month")
print(f"Average 2023+:    {avg_post2023:.1f}/month")
print(f"Uplift factor:    {avg_post2023/avg_pre2023:.1f}x")


# ── 6. PLOTS ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Generating figures")
print("=" * 60)

plot_topic_landscape(topic_model, save_dir=OUT_FIGURES)
plot_gnss_timeline(
    topics_over_time_df=topics_over_time_df,
    gnss_topic_ids=gnss_topic_ids,
    all_gnss_monthly=gnss_monthly,
    save_dir=OUT_FIGURES,
)
plot_red_quadrant_topics(asrs, topic_model, save_dir=OUT_FIGURES)
plot_topic_heatmap(topics_over_time_df, topic_model,
                   top_n=15, save_dir=OUT_FIGURES)


# ── 7. TOPICS OVER TIME: GNSS STATS ──────────────────────────────────────────
if topics_over_time_df is not None and gnss_topic_ids:
    print("\n" + "=" * 60)
    print("GNSS topic frequency by year (BERTopic topics_over_time):")
    print("=" * 60)
    gnss_tot = topics_over_time_df[
        topics_over_time_df['Topic'].isin(gnss_topic_ids)
    ].copy()
    gnss_tot['Year'] = pd.to_datetime(gnss_tot['Timestamp']).dt.year
    yearly = gnss_tot.groupby('Year')['Frequency'].sum()
    print(yearly.to_string())


# ── 8. SAVE ENRICHED DATASET ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Saving Layer 2 enriched dataset")
print("=" * 60)
out = os.path.join(OUT_DATA, "asrs_layer2.parquet")
save_df = asrs.copy()
for col in save_df.select_dtypes(include=['object', 'str']).columns:
    save_df[col] = save_df[col].astype(str)
save_df.to_parquet(out, index=False)
print(f"Saved: {out}  |  shape: {asrs.shape}")

# Save topic model for Layer 4 RAG context
model_dir = os.path.join(OUT_DATA, "bertopic_model")
topic_model.save(model_dir, serialization="safetensors",
                 save_ctfidf=True, save_embedding_model=True)
print(f"BERTopic model saved to: {model_dir}")

print("\nLayer 2 complete.")
print(f"Figures: {OUT_FIGURES}")
print(f"Key outputs: layer2_topic_landscape.png, layer2_gnss_emergence.png, "
      f"layer2_red_topics.png, layer2_topic_heatmap.png")
