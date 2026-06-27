"""
Layer 2 runner: BERTopic on full ASRS corpus.

Loads Layer 1 output, fits BERTopic, tracks topics over time, identifies GNSS
topic clusters, saves figures, topic model, and enriched parquet.

Run:
    uv run python run_layer2.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

from src.logger import get_logger
from src.plotter import (
    plot_gnss_timeline,
    plot_red_quadrant_topics,
    plot_topic_heatmap,
    plot_topic_landscape,
)
from src.topics import (
    find_gnss_topic_ids,
    get_topic_summary,
    run_bertopic,
)

matplotlib.use("Agg")

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

LAYER1_PATH = DATA_DIR / "asrs_layer1.parquet"
LAYER2_PATH = DATA_DIR / "asrs_layer2.parquet"
TOPIC_SUMMARY_PATH = DATA_DIR / "layer2_topic_summary.csv"
MODEL_DIR = DATA_DIR / "bertopic_model"

MIN_WORDS = 30

GNSS_REGEX = (
    r"spoof|jamm|gps.{0,25}interfer|gnss.{0,25}interfer|gps.{0,25}denial|"
    r"gps.{0,25}unreliable|gps.{0,25}degrad|position.{0,25}spoof|"
    r"gps.{0,25}lost|navigation.{0,25}warn|gps.{0,25}alert"
)


def _load_layer1_dataset() -> pd.DataFrame:
    asrs = pd.read_parquet(LAYER1_PATH)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")

    logger.info(
        "Loaded %s records from %s | %s to %s",
        f"{len(asrs):,}",
        LAYER1_PATH,
        asrs["date"].min().date(),
        asrs["date"].max().date(),
    )
    logger.info("Quadrant counts:\n%s", asrs["quadrant"].value_counts().to_string())
    return asrs


def _prepare_corpus(asrs: pd.DataFrame) -> pd.DataFrame:
    mask = (
        asrs["full_narrative"].notna()
        & (asrs["narrative_word_count"] >= MIN_WORDS)
        & asrs["date"].notna()
    )
    corpus = asrs.loc[mask].copy()

    logger.info(
        "Documents with >=%d words and valid date: %s",
        MIN_WORDS,
        f"{len(corpus):,}",
    )
    logger.info("Dropped short/undated records: %s", f"{len(asrs) - len(corpus):,}")
    logger.info(
        "Corpus date range: %s to %s",
        corpus["date"].min().date(),
        corpus["date"].max().date(),
    )
    return corpus


def _attach_topics(
    asrs: pd.DataFrame,
    corpus: pd.DataFrame,
    topics: list[int],
    topic_model,
) -> pd.DataFrame:
    corpus = corpus.copy()
    corpus["topic_id"] = topics

    topic_info = topic_model.get_topic_info()
    topic_labels = topic_info.set_index("Topic")["Name"].to_dict()
    corpus["topic_label"] = corpus["topic_id"].map(topic_labels)

    asrs = asrs.merge(
        corpus[["topic_id", "topic_label"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    asrs["topic_id"] = asrs["topic_id"].fillna(-2).astype(int)
    return asrs


def _log_gnss_regex_signal(asrs: pd.DataFrame) -> pd.Series:
    gnss_mask = asrs["full_narrative"].astype(str).str.lower().str.contains(
        GNSS_REGEX,
        regex=True,
        na=False,
    )
    gnss_monthly = (
        asrs.loc[gnss_mask]
        .groupby(asrs.loc[gnss_mask, "date"].dt.to_period("M"))["ACN"]
        .count()
    )

    logger.info("GNSS incidents: %s", f"{gnss_mask.sum():,}")
    logger.info("Monthly range: %s to %s", gnss_monthly.min(), gnss_monthly.max())

    pre2023 = gnss_monthly[gnss_monthly.index < "2023"].mean()
    post2023 = gnss_monthly[gnss_monthly.index >= "2023"].mean()
    logger.info("Average pre-2023: %.1f/month", pre2023)
    logger.info("Average 2023+: %.1f/month", post2023)
    logger.info("Uplift factor: %.1fx", post2023 / pre2023)

    return gnss_monthly


def _save_layer2_dataset(asrs: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_df = asrs.copy()
    for column in save_df.select_dtypes(include=["object"], exclude=["str"]).columns:
        save_df[column] = save_df[column].astype(str)

    save_df.to_parquet(LAYER2_PATH, index=False)
    logger.info("Saved: %s | shape: %s", LAYER2_PATH, asrs.shape)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("STEP 1: Loading Layer 1 enriched dataset")
    asrs = _load_layer1_dataset()

    logger.info("STEP 2: Preparing documents")
    corpus = _prepare_corpus(asrs)

    docs = corpus["full_narrative"].tolist()
    timestamps = corpus["date"].tolist()

    logger.info("STEP 3: Fitting BERTopic")
    topic_model, topics, topics_over_time_df = run_bertopic(
        docs=docs,
        timestamps=timestamps,
        nr_topics=40,
        min_cluster_size=30,
    )

    asrs = _attach_topics(asrs, corpus, topics, topic_model)

    topic_summary = get_topic_summary(topic_model, top_n=20)
    logger.info("Top 20 topics:\n%s", topic_summary.to_string(index=False))
    topic_summary.to_csv(TOPIC_SUMMARY_PATH, index=False)
    logger.info("Saved topic summary to %s", TOPIC_SUMMARY_PATH)

    logger.info("STEP 4: Identifying GNSS / spoofing topic clusters")
    gnss_topic_ids = find_gnss_topic_ids(topic_model)
    logger.info("GNSS-related topics: %s", gnss_topic_ids)

    for topic_id in gnss_topic_ids:
        words = topic_model.get_topic(topic_id)
        keywords = ", ".join(f"{word}({score:.3f})" for word, score in words[:10]) if words else "n/a"
        count = (asrs["topic_id"] == topic_id).sum()
        logger.info("Topic %s | n=%s | %s", topic_id, f"{count:,}", keywords)

    logger.info("STEP 5: Rebuilding GNSS regex signal for comparison")
    gnss_monthly = _log_gnss_regex_signal(asrs)

    logger.info("STEP 6: Generating figures")
    plot_topic_landscape(topic_model, save_dir=str(FIGURES_DIR))
    plot_gnss_timeline(
        topics_over_time_df=topics_over_time_df,
        gnss_topic_ids=gnss_topic_ids,
        all_gnss_monthly=gnss_monthly,
        save_dir=str(FIGURES_DIR),
    )
    plot_red_quadrant_topics(asrs, topic_model, save_dir=str(FIGURES_DIR))
    plot_topic_heatmap(
        topics_over_time_df,
        topic_model,
        top_n=15,
        save_dir=str(FIGURES_DIR),
    )

    if topics_over_time_df is not None and gnss_topic_ids:
        gnss_tot = topics_over_time_df[
            topics_over_time_df["Topic"].isin(gnss_topic_ids)
        ].copy()
        gnss_tot["Year"] = pd.to_datetime(gnss_tot["Timestamp"]).dt.year
        yearly = gnss_tot.groupby("Year")["Frequency"].sum()
        logger.info("GNSS topic frequency by year:\n%s", yearly.to_string())

    logger.info("STEP 7: Saving Layer 2 enriched dataset")
    _save_layer2_dataset(asrs)

    topic_model.save(
        MODEL_DIR,
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=True,
    )
    logger.info("BERTopic model saved to: %s", MODEL_DIR)

    logger.info("Layer 2 complete.")
    logger.info(
        "Key outputs: layer2_topic_landscape.png, layer2_gnss_emergence.png, "
        "layer2_red_topics.png, layer2_topic_heatmap.png"
    )


if __name__ == "__main__":
    main()