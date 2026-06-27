"""
Layer 4 runner: ChromaDB RAG demo.

Loads Layer 3 enriched dataset, indexes RED+ORANGE incidents, and runs all demo
queries when ANTHROPIC_API_KEY is available.

Run:
    uv run python run_layer4.py
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

from src.logger import get_logger
from src.rag import DEMO_QUERIES, build_rag_index, run_all_demos

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
CHROMA_DIR = DATA_DIR / "chromadb"

LAYER3_PATH = DATA_DIR / "asrs_layer3.parquet"
LAYER2_PATH = DATA_DIR / "asrs_layer2.parquet"


def _load_input_dataset() -> pd.DataFrame:
    if LAYER3_PATH.exists():
        source_path = LAYER3_PATH
    else:
        source_path = LAYER2_PATH
        logger.warning("Layer 3 not found, falling back to %s", source_path)

    asrs = pd.read_parquet(source_path)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")

    logger.info("Loaded %s records from %s", f"{len(asrs):,}", source_path)
    logger.info("Columns: %d", asrs.shape[1])
    logger.info("Quadrant breakdown:\n%s", asrs["quadrant"].value_counts().to_string())

    flagged_count = asrs["quadrant"].isin(["RED", "ORANGE"]).sum()
    logger.info("RED + ORANGE incidents: %s", f"{flagged_count:,}")

    if "precursor_score" in asrs.columns:
        top_pct = (
            asrs.loc[asrs["quadrant"].isin(["RED", "ORANGE"]), "precursor_score"]
            .quantile(0.9)
        )
        logger.info("90th-pct precursor score in flagged set: %.3f", top_pct)

    return asrs


def _spot_check_search(collection, embedding_model) -> None:
    query = "GPS spoofing radar ghost targets"
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    for index, (doc, meta, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        1,
    ):
        cosine_similarity = 1 - distance
        rows.append(
            {
                "rank": index,
                "acn": meta["acn"],
                "date": meta["date"],
                "cosine_similarity": round(cosine_similarity, 3),
                "risk": meta["precursor_score"],
                "preview": doc[:150].replace("\n", " "),
            }
        )

    logger.info("Spot-check query: %s", query)
    logger.info("Spot-check results:\n%s", pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    logger.info("STEP 1: Loading Layer 3 enriched dataset")
    asrs = _load_input_dataset()

    logger.info("STEP 2: Building/loading ChromaDB index")
    logger.info(
        "Indexing up to 3,000 RED+ORANGE incidents, prioritized by precursor_score"
    )

    start_time = time.time()
    collection, _client, embedding_model = build_rag_index(
        asrs,
        max_incidents=3000,
        persist_dir=str(CHROMA_DIR),
        force_rebuild=False,
    )
    elapsed = time.time() - start_time

    logger.info("Index stats:")
    logger.info("  Total indexed: %s", f"{collection.count():,}")
    logger.info("  Wall time: %.1fs", elapsed)

    logger.info("STEP 3: Spot-check semantic search")
    _spot_check_search(collection, embedding_model)

    logger.info("STEP 4: Running demo queries with Claude API if available")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.strip() == "your-api-key-here":
        logger.warning("ANTHROPIC_API_KEY not set in .env")
        logger.info("Index is persisted; no re-embedding needed on next run.")
    else:
        logger.info("API key: SET | Demo queries: %d", len(DEMO_QUERIES))
        run_all_demos(collection, embedding_model)

    logger.info("Layer 4 complete.")
    logger.info("Incidents indexed: %s", f"{collection.count():,}")
    logger.info("Embedding/index load time: %.1fs", elapsed)


if __name__ == "__main__":
    main()