from scipy.spatial.distance import cosine
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from src.logger import get_logger

logger = get_logger(__name__)


def run_bertopic(
    docs: list[str],
    timestamps: list | None = None,
    nr_topics: int = 40,
    min_cluster_size: int = 30,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> tuple:
    """
    Fit BERTopic on narrative documents.

    Returns:
        (topic_model, topics, topics_over_time_df)

    `topics_over_time_df` is None if timestamps are not supplied.
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
            docs,
            timestamps,
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=99,
        )

    return topic_model, topics, topics_over_time_df


def find_gnss_topic_ids(topic_model) -> list[int]:
    """Return topic IDs whose top keywords overlap GNSS/spoofing terms."""
    gnss_terms = {
        "spoof",
        "jamm",
        "gps",
        "gnss",
        "navigation",
        "position",
        "satellite",
        "interference",
        "denial",
        "unreliable",
        "degrad",
    }

    topic_info = topic_model.get_topic_info()
    gnss_ids = []

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue

        top_words = topic_model.get_topic(topic_id)
        if not top_words:
            continue

        words = {word.lower() for word, _ in top_words}
        if words & gnss_terms:
            gnss_ids.append(topic_id)

    return gnss_ids


def get_topic_summary(topic_model, top_n: int = 20) -> pd.DataFrame:
    """Return top_n topics with keywords as a clean DataFrame."""
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].head(top_n).copy()
    info["keywords"] = info["Topic"].apply(
        lambda topic_id: ", ".join(
            word for word, _ in (topic_model.get_topic(topic_id) or [])[:8]
        )
    )
    return info[["Topic", "Count", "Name", "keywords"]].reset_index(drop=True)


def compute_semantic_drift(
    asrs: pd.DataFrame,
    topic_model,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Compute quarter-on-quarter centroid cosine drift for each topic.

    High drift means the language describing a topic is changing rapidly.
    This is exploratory and not currently part of the main presentation claims.
    """
    model = SentenceTransformer(embedding_model_name)
    asrs = asrs.copy()
    asrs["quarter"] = asrs["date"].dt.to_period("Q")

    drift_records = []
    for topic_id in asrs["topic_id"].dropna().unique():
        if topic_id < 0:
            continue

        topic_data = asrs[asrs["topic_id"] == topic_id]
        centroids = {}

        for quarter in sorted(topic_data["quarter"].dropna().unique()):
            quarter_texts = (
                topic_data.loc[topic_data["quarter"] == quarter, "full_narrative"]
                .dropna()
                .tolist()
            )
            if len(quarter_texts) >= 3:
                embeddings = model.encode(quarter_texts, show_progress_bar=False)
                centroids[quarter] = embeddings.mean(axis=0)

        quarters = sorted(centroids.keys())
        for index in range(1, len(quarters)):
            previous_quarter = quarters[index - 1]
            current_quarter = quarters[index]
            drift = cosine(centroids[previous_quarter], centroids[current_quarter])
            drift_records.append(
                {
                    "topic_id": topic_id,
                    "quarter": str(current_quarter),
                    "drift_velocity": drift,
                    "n_incidents": (
                        (asrs["topic_id"] == topic_id)
                        & (asrs["quarter"] == current_quarter)
                    ).sum(),
                }
            )

    drift_df = pd.DataFrame(drift_records)
    if not drift_df.empty:
        threshold = drift_df["drift_velocity"].quantile(0.80)
        drift_df["high_drift"] = drift_df["drift_velocity"] > threshold

    return drift_df