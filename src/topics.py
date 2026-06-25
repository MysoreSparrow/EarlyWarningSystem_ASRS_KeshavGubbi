import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from scipy.spatial.distance import cosine
from src.logger import get_logger

logger = get_logger(__name__)


# ── Core BERTopic runner ──────────────────────────────────────────────────────

def run_bertopic(
    docs: list,
    timestamps: list | None = None,
    nr_topics: int = 40,
    min_cluster_size: int = 30,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> tuple:
    """
    Fit BERTopic on `docs`.
    Returns (topic_model, topics, topics_over_time_df).
    topics_over_time_df is None if timestamps not supplied.
    """
    logger.info("Encoding %s documents with %s", f"{len(docs):,}", embedding_model_name)
    embedding_model = SentenceTransformer(embedding_model_name)

    umap_model = UMAP(
        n_components=5,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(docs)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    noise = topics.count(-1)
    logger.info("Discovered %d topics | %s noise incidents", n_topics, f"{noise:,}")

    topics_over_time_df = None
    if timestamps is not None:
        logger.info("Computing topics over time...")
        topics_over_time_df = topic_model.topics_over_time(
            docs, timestamps,
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=99,
        )

    return topic_model, topics, topics_over_time_df


# ── Topic identification helpers ──────────────────────────────────────────────

def find_gnss_topic_ids(topic_model) -> list:
    """Return topic IDs whose top keywords overlap with GNSS/spoofing terms."""
    gnss_terms = {
        'spoof', 'jamm', 'gps', 'gnss', 'navigation', 'position', 'satellite',
        'interference', 'denial', 'unreliable', 'degrad',
    }
    topic_info = topic_model.get_topic_info()
    gnss_ids = []
    for _, row in topic_info.iterrows():
        tid = row['Topic']
        if tid == -1:
            continue
        top_words = topic_model.get_topic(tid)
        if top_words:
            words = {w.lower() for w, _ in top_words}
            if words & gnss_terms:
                gnss_ids.append(tid)
    return gnss_ids


def get_topic_summary(topic_model, top_n: int = 20) -> pd.DataFrame:
    """Return top_n topics with their keywords as a clean DataFrame."""
    info = topic_model.get_topic_info()
    info = info[info['Topic'] != -1].head(top_n).copy()
    info['keywords'] = info['Topic'].apply(
        lambda tid: ', '.join(
            w for w, _ in (topic_model.get_topic(tid) or [])[:8]
        )
    )
    return info[['Topic', 'Count', 'Name', 'keywords']].reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_topic_landscape(topic_model, save_dir: str = "outputs/figures") -> None:
    """Horizontal bar chart of top 20 topics by document count."""
    os.makedirs(save_dir, exist_ok=True)
    info = topic_model.get_topic_info()
    info = info[info['Topic'] != -1].head(20).copy()
    info['short_name'] = info['Topic'].apply(
        lambda tid: ' | '.join(
            w for w, _ in (topic_model.get_topic(tid) or [])[:5]
        )
    )
    info = info.sort_values('Count')

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(info['short_name'], info['Count'],
                   color='steelblue', alpha=0.8)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=8)
    ax.set_xlabel("Incident count", fontsize=11)
    ax.set_title(
        "Layer 2: BERTopic — Top 20 Semantic Topics\nNASA ASRS 2018-2026",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "layer2_topic_landscape.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved topic landscape to %s", out)
    plt.close()


def plot_gnss_timeline(
    topics_over_time_df: pd.DataFrame,
    gnss_topic_ids: list,
    all_gnss_monthly: pd.Series,
    save_dir: str = "outputs/figures",
) -> None:
    """
    Two-panel figure:
      Top: GNSS topic frequency from BERTopic topics_over_time
      Bottom: Raw regex-based GNSS incident count (Layer 1 signal)
    Shows the emergence story from both angles.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    # ── Panel 1: BERTopic topics_over_time ──
    ax0 = axes[0]
    if topics_over_time_df is not None and gnss_topic_ids:
        gnss_tot = topics_over_time_df[
            topics_over_time_df['Topic'].isin(gnss_topic_ids)
        ].copy()
        gnss_tot['Timestamp'] = pd.to_datetime(gnss_tot['Timestamp'])
        grouped = gnss_tot.groupby('Timestamp')['Frequency'].sum().reset_index()
        grouped = grouped.sort_values('Timestamp')
        ax0.fill_between(grouped['Timestamp'], grouped['Frequency'],
                         alpha=0.3, color='royalblue')
        ax0.plot(grouped['Timestamp'], grouped['Frequency'],
                 color='royalblue', lw=2)
        ax0.axvline(pd.Timestamp('2023-10-01'), color='red', ls='--', lw=1.5,
                    label='Late-2023 inflection')
        ax0.legend(fontsize=9)
    else:
        ax0.text(0.5, 0.5, 'No GNSS topics identified in model',
                 ha='center', va='center', transform=ax0.transAxes)
    ax0.set_title("GNSS / Spoofing topic frequency — BERTopic semantic cluster",
                  fontsize=11, fontweight='bold')
    ax0.set_ylabel("Topic frequency (BERTopic)")

    # ── Panel 2: Raw regex count ──
    ax1 = axes[1]
    if all_gnss_monthly is not None and not all_gnss_monthly.empty:
        ts = all_gnss_monthly.index.to_timestamp() if hasattr(
            all_gnss_monthly.index, 'to_timestamp') else all_gnss_monthly.index
        ax1.fill_between(ts, all_gnss_monthly.values,
                         alpha=0.3, color='darkorange')
        ax1.plot(ts, all_gnss_monthly.values,
                 color='darkorange', lw=2, label='spoofing|jamming mentions')
        # Rolling 3-month mean
        roll = all_gnss_monthly.rolling(3, center=True).mean()
        ax1.plot(ts, roll.values, color='red', lw=2.5,
                 ls='-', label='3-month rolling mean')
        ax1.axvline(pd.Timestamp('2023-10-01'), color='red', ls='--', lw=1.5,
                    label='Late-2023 inflection')
        ax1.legend(fontsize=9)
    ax1.set_title("GNSS spoofing/jamming narrative mentions — regex signal (Layer 1)",
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel("Monthly incident count")
    ax1.set_xlabel("Date")

    fig.suptitle(
        "GNSS Spoofing Emergence: Near-Zero (2018-2022) to Sustained Signal (2023-2025)\n"
        "NASA ASRS | Layer 2 semantic validation of Layer 1 SPC signal",
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "layer2_gnss_emergence.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved GNSS timeline to %s", out)
    plt.close()


def plot_red_quadrant_topics(
    asrs: pd.DataFrame,
    topic_model,
    save_dir: str = "outputs/figures",
) -> None:
    """Top topics within the RED (novel + anomalous) quadrant incidents."""
    os.makedirs(save_dir, exist_ok=True)
    red = asrs[asrs.get('quadrant', pd.Series()) == 'RED'].copy()
    if 'topic_id' not in red.columns or red.empty:
        logger.warning("No RED quadrant topic data to plot")
        return

    counts = red['topic_id'].value_counts()
    counts = counts[counts.index != -1].head(15)

    topic_info = topic_model.get_topic_info().set_index('Topic')
    labels = []
    for tid in counts.index:
        words = topic_model.get_topic(tid)
        kw = ' | '.join(w for w, _ in words[:5]) if words else f"Topic {tid}"
        labels.append(kw)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels[::-1], counts.values[::-1],
                   color='#cc0000', alpha=0.75)
    ax.bar_label(bars, fmt='%d', padding=4, fontsize=8)
    ax.set_xlabel("Incident count", fontsize=11)
    ax.set_title(
        "Layer 2: Topic Breakdown — RED Quadrant Incidents\n"
        "(Novel pattern + SPC anomalous frequency)",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "layer2_red_topics.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved RED quadrant topics to %s", out)
    plt.close()


def plot_topic_heatmap(
    topics_over_time_df: pd.DataFrame,
    topic_model,
    top_n: int = 15,
    save_dir: str = "outputs/figures",
) -> None:
    """Heatmap of topic frequency over time for top_n topics."""
    os.makedirs(save_dir, exist_ok=True)
    if topics_over_time_df is None or topics_over_time_df.empty:
        return

    info = topic_model.get_topic_info()
    top_ids = info[info['Topic'] != -1].head(top_n)['Topic'].tolist()

    tot = topics_over_time_df[topics_over_time_df['Topic'].isin(top_ids)].copy()
    tot['Timestamp'] = pd.to_datetime(tot['Timestamp'])
    tot['YearQ'] = tot['Timestamp'].dt.to_period('Q').astype(str)

    pivot = tot.pivot_table(
        index='Topic', columns='YearQ', values='Frequency', aggfunc='sum'
    ).fillna(0)

    # Label rows with top keywords
    row_labels = []
    for tid in pivot.index:
        words = topic_model.get_topic(tid)
        kw = ' | '.join(w for w, _ in words[:4]) if words else f"T{tid}"
        row_labels.append(kw)
    pivot.index = row_labels

    # Normalise each row to 0-1 so rare topics are visible
    pivot_norm = pivot.div(pivot.max(axis=1) + 1e-9, axis=0)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(pivot_norm.values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label='Normalised frequency (per topic)')
    ax.set_title(
        "Layer 2: Topic Evolution Heatmap — Top 15 Topics by Quarter\nNASA ASRS 2018-2026",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = os.path.join(save_dir, "layer2_topic_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved topic heatmap to %s", out)
    plt.close()


# ── Semantic drift (kept from stub) ──────────────────────────────────────────

def compute_semantic_drift(
    asrs: pd.DataFrame,
    topic_model,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    For each topic, compute quarter-on-quarter cosine drift of the centroid.
    High drift = language describing that hazard is changing rapidly.
    """
    model = SentenceTransformer(embedding_model_name)
    asrs = asrs.copy()
    asrs['quarter'] = asrs['date'].dt.to_period('Q')

    drift_records = []
    for topic_id in asrs['topic_id'].dropna().unique():
        if topic_id < 0:
            continue
        topic_data = asrs[asrs['topic_id'] == topic_id]
        centroids = {}
        for q in sorted(topic_data['quarter'].dropna().unique()):
            q_texts = topic_data[topic_data['quarter'] == q][
                'full_narrative'].dropna().tolist()
            if len(q_texts) >= 3:
                emb = model.encode(q_texts, show_progress_bar=False)
                centroids[q] = emb.mean(axis=0)

        quarters = sorted(centroids.keys())
        for i in range(1, len(quarters)):
            qp, qc = quarters[i - 1], quarters[i]
            drift = cosine(centroids[qp], centroids[qc])
            drift_records.append({
                'topic_id': topic_id,
                'quarter': str(qc),
                'drift_velocity': drift,
                'n_incidents': (
                    (asrs['topic_id'] == topic_id) &
                    (asrs['quarter'] == qc)
                ).sum(),
            })

    drift_df = pd.DataFrame(drift_records)
    if not drift_df.empty:
        threshold = drift_df['drift_velocity'].quantile(0.80)
        drift_df['high_drift'] = drift_df['drift_velocity'] > threshold
    return drift_df
