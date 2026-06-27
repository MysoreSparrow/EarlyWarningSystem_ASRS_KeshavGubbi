"""
Layer 4: ChromaDB RAG demo.

Architecture decisions:
- ChromaDB PersistentClient: zero server setup, appropriate for demo.
  Production target: Qdrant with BGE-M3 for hybrid dense+sparse retrieval.
- all-MiniLM-L6-v2: fast CPU, 384-dim, good quality for macro retrieval.
  Production target: fine-tune on ASRS + SKYbrary for aviation acronyms.
- Rich metadata enables analyst-style filtered retrieval.
- System prompt forbids confabulation. Non-hallucination is mandatory for
  safety-critical workflows.

API key is loaded from .env. Never commit .env.
"""
import os
import time
from pathlib import Path
from typing import Any, Optional

import anthropic
import chromadb
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
COLLECTION_NAME = "asrs_flagged"
INDEX_VERSION = "2026-06-layer1-strict-spc-v1"

FLAGGED_QUADRANTS = ("RED", "ORANGE")
DEFAULT_PERSIST_DIR = Path("outputs/data/chromadb")


DEMO_QUERIES = [
    {
        "question": (
            "What patterns appear in incidents involving GPS, "
            "navigation errors, or unusual radar targets in 2023?"
        ),
        "filter_kwargs": {},
        "why": (
            "Opens the GNSS story and should surface ghost-target "
            "and spoofing-related narratives."
        ),
    },
    {
        "question": (
            "Which incidents show communication breakdown between "
            "ATC and pilots, and what were the outcomes?"
        ),
        "filter_kwargs": {},
        "why": "Tests the communication-breakdown precursor component.",
    },
    {
        "question": (
            "Show me the most serious incidents where pilots "
            "reported fatigue or inadequate rest."
        ),
        "filter_kwargs": {"min_precursor_score": 0.3},
        "why": "Tests metadata filtering by precursor score.",
    },
    {
        "question": (
            "What do the RED quadrant incidents, those that are "
            "both novel and anomalously frequent, have in common?"
        ),
        "filter_kwargs": {"filter_quadrant": "RED"},
        "why": "Tests quadrant filtering.",
    },
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Cast to float, replacing NaN/None with default."""
    try:
        parsed = float(value)
        return default if parsed != parsed else parsed
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Cast to int, replacing NaN/None with default."""
    try:
        parsed = float(value)
        return default if parsed != parsed else int(parsed)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    """Cast metadata values to clean strings."""
    if value is None:
        return default
    text = str(value)
    return default if text.lower() in {"nan", "none", "nat"} else text


def _stable_incident_id(row: pd.Series, fallback_index: int) -> str:
    """Build stable Chroma document ID from ACN when available."""
    acn = _safe_str(row.get("ACN", "")).strip()
    if acn:
        return f"acn_{acn}"
    return f"incident_{fallback_index}"


def _dedupe_ids(ids: list[str]) -> list[str]:
    """Ensure Chroma IDs are unique while preserving stable ACN-based IDs."""
    seen: dict[str, int] = {}
    deduped = []

    for doc_id in ids:
        seen[doc_id] = seen.get(doc_id, 0) + 1
        count = seen[doc_id]
        deduped.append(doc_id if count == 1 else f"{doc_id}_{count}")

    return deduped


def _index_metadata(max_incidents: int, embedding_model_name: str) -> dict[str, str]:
    """Metadata used to detect stale persisted Chroma collections."""
    return {
        "hnsw:space": "cosine",
        "index_version": INDEX_VERSION,
        "embedding_model": embedding_model_name,
        "max_incidents": str(max_incidents),
        "selection": "RED_ORANGE_sorted_by_precursor_score",
    }


def _collection_matches_expected(
    collection: chromadb.Collection,
    max_incidents: int,
    embedding_model_name: str,
) -> bool:
    """Return True when an existing collection matches expected index metadata."""
    metadata = collection.metadata or {}
    expected = _index_metadata(max_incidents, embedding_model_name)

    return all(
        str(metadata.get(key)) == str(value)
        for key, value in expected.items()
        if key != "hnsw:space"
    )


def _select_flagged_incidents(
    asrs: pd.DataFrame,
    max_incidents: int,
) -> pd.DataFrame:
    """Select RED/ORANGE incidents for indexing, prioritised by risk score."""
    required = {"quadrant", "full_narrative"}
    missing = sorted(required - set(asrs.columns))
    if missing:
        raise ValueError(f"Missing required columns for RAG indexing: {missing}")

    flagged = asrs[asrs["quadrant"].isin(FLAGGED_QUADRANTS)].copy()
    if flagged.empty:
        raise ValueError("No RED/ORANGE incidents available for RAG indexing.")

    if "precursor_score" not in flagged.columns:
        logger.warning(
            "Column 'precursor_score' not found. "
            "Index priority will fall back to if_score, then original order."
        )
        flagged["precursor_score"] = 0.0

    if "if_score" not in flagged.columns:
        flagged["if_score"] = 0.0

    flagged["precursor_score"] = flagged["precursor_score"].map(_safe_float)
    flagged["if_score"] = flagged["if_score"].map(_safe_float)

    flagged = flagged.sort_values(
        ["precursor_score", "if_score"],
        ascending=False,
    )

    if len(flagged) > max_incidents:
        flagged = flagged.head(max_incidents)

    return flagged


def _metadata_for_row(row: pd.Series) -> dict[str, str | int | float]:
    """Build Chroma-compatible primitive metadata for one incident."""
    date_str = _safe_str(row.get("date", ""))[:10]
    try:
        year = int(date_str[:4]) if len(date_str) >= 4 else 0
    except ValueError:
        year = 0

    return {
        "acn": _safe_str(row.get("ACN", "")),
        "date": date_str,
        "year": year,
        "anomaly": _safe_str(row.get("Events | Anomaly", ""))[:200],
        "flight_phase": _safe_str(
            row.get("Aircraft 1 | Flight Phase", "Unknown"),
            default="Unknown",
        ),
        "quadrant": _safe_str(row.get("quadrant", "UNKNOWN"), default="UNKNOWN"),
        "spc_flag": _safe_int(row.get("spc_flag", 0)),
        "if_score": round(_safe_float(row.get("if_score", 0.0)), 3),
        "precursor_score": round(_safe_float(row.get("precursor_score", 0.0)), 3),
        "topic_label": _safe_str(row.get("topic_label", "Unknown"))[:100],
        "component_fatigue": _safe_int(row.get("component_fatigue", 0)),
        "component_near_miss": _safe_int(row.get("component_near_miss", 0)),
        "component_comm_breakdown": _safe_int(row.get("component_comm_breakdown", 0)),
    }


def _build_where_filter(
    filter_quadrant: Optional[str],
    min_precursor_score: Optional[float],
) -> dict[str, Any] | None:
    """Build a ChromaDB metadata filter."""
    filters: list[dict[str, Any]] = []

    if filter_quadrant:
        filters.append({"quadrant": {"$eq": filter_quadrant}})

    if min_precursor_score is not None:
        filters.append({"precursor_score": {"$gte": min_precursor_score}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def build_rag_index(
    asrs: pd.DataFrame,
    max_incidents: int = 3000,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    force_rebuild: bool = False,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[chromadb.Collection, Any, SentenceTransformer]:
    """
    Index RED and ORANGE quadrant incidents in ChromaDB.

    Selection:
        RED/ORANGE only, sorted by precursor_score then if_score, capped at
        max_incidents.

    Persistence:
        Saved to persist_dir. Existing collections are reused only when metadata
        matches the current index version and embedding model.

    Returns:
        (collection, client, embedding_model)
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model: %s", embedding_model_name)
    embedding_model = SentenceTransformer(embedding_model_name)

    client = chromadb.PersistentClient(path=str(persist_path))

    if not force_rebuild:
        try:
            collection = client.get_collection(COLLECTION_NAME)
            existing = collection.count()

            if existing > 0 and _collection_matches_expected(
                collection,
                max_incidents=max_incidents,
                embedding_model_name=embedding_model_name,
            ):
                logger.info(
                    "Loaded existing index from %s: %s incidents",
                    persist_path,
                    f"{existing:,}",
                )
                return collection, client, embedding_model

            logger.warning("Existing Chroma index is stale; rebuilding.")
        except Exception:
            logger.info("No existing Chroma index found; building a fresh index.")

    flagged = _select_flagged_incidents(asrs, max_incidents=max_incidents)

    red_count = (flagged["quadrant"] == "RED").sum()
    orange_count = (flagged["quadrant"] == "ORANGE").sum()
    logger.info(
        "Incidents to index: %s (RED: %d, ORANGE: %d)",
        f"{len(flagged):,}",
        red_count,
        orange_count,
    )

    narratives = flagged["full_narrative"].fillna("").astype(str).tolist()
    if not narratives:
        raise ValueError("No narratives available for RAG indexing.")

    logger.info("Embedding %s narratives on CPU...", f"{len(narratives):,}")
    start_time = time.time()
    embeddings = embedding_model.encode(
        narratives,
        show_progress_bar=True,
        batch_size=64,
    )
    elapsed = time.time() - start_time
    rate = len(narratives) / max(elapsed, 1e-9)
    logger.info("Embedding complete in %.1fs (%.0f incidents/sec)", elapsed, rate)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata=_index_metadata(max_incidents, embedding_model_name),
    )

    metadatas = []
    ids = []
    for fallback_index, (_, row) in enumerate(flagged.iterrows()):
        metadatas.append(_metadata_for_row(row))
        ids.append(_stable_incident_id(row, fallback_index))

    ids = _dedupe_ids(ids)

    collection.add(
        documents=narratives,
        embeddings=[embedding.tolist() for embedding in embeddings],
        metadatas=metadatas,
        ids=ids,
    )

    logger.info("Index complete: %s incidents in ChromaDB", f"{collection.count():,}")
    return collection, client, embedding_model


def rag_query(
    question: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    n_results: int = 5,
    filter_quadrant: Optional[str] = None,
    min_precursor_score: Optional[float] = None,
) -> str:
    """
    Semantic search + Claude synthesis over indexed incidents.

    Returns a cited answer. Every claim should be traceable to a retrieved ACN.
    """
    if not question.strip():
        return "No question provided."

    collection_count = collection.count()
    if collection_count == 0:
        return "No incidents are available in the RAG index."

    where_filter = _build_where_filter(filter_quadrant, min_precursor_score)
    n = min(n_results, collection_count)

    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")
    distances = _first_result_list(results, "distances")
    
    if not documents:
        return (
            "No matching incidents were retrieved for this query/filter. "
            "Try relaxing the quadrant or precursor-score filter."
        )

    retrieved = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        1,
    ):
        retrieved.append(
            {
                "rank": rank,
                "document": str(document),
                "metadata": dict(metadata),
                "similarity": round(1 - float(distance), 3),
            }
        )

    context_parts = []
    for row in retrieved:
        index = row["rank"]
        document = row["document"]
        metadata = row["metadata"]
        similarity = row["similarity"]
        context_parts.append(
            f"[{index}] ACN:{metadata.get('acn', '')} | "
            f"Date:{metadata.get('date', '')} | "
            f"Phase:{metadata.get('flight_phase', '')}\n"
            f"Quadrant:{metadata.get('quadrant', '')} | "
            f"Risk:{metadata.get('precursor_score', 0):.2f} | "
            f"IF:{metadata.get('if_score', 0):.2f} | "
            f"Similarity:{similarity:.3f}\n"
            f"Anomaly: {str(metadata.get('anomaly', ''))[:120]}\n"
            f"Narrative: {str(document)[:500]}\n"
        )

    context = ("\n" + "-" * 60 + "\n").join(context_parts)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not found in .env"

    claude_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL)
    claude = anthropic.Anthropic(api_key=api_key)

    response = claude.messages.create(
        model=claude_model,
        max_tokens=700,
        system="""You are an aviation safety analyst assistant.
Your role is to identify patterns in incident reports that could indicate emerging safety risks.

Rules:
1. Cite specific incidents using [1], [2], [3] etc.
2. ONLY use information from the provided incident reports.
3. If evidence is insufficient, say so explicitly. Do not speculate.
4. Focus on patterns across multiple incidents, not individual cases.
5. Note risk scores and quadrant classifications where relevant.
6. Never hallucinate facts not present in the reports.""",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved incidents:\n{context}"
                ),
            }
        ],
    )

    answer = _text_from_anthropic_response(response)

    citations = "\n".join(
        [
            f"  [{row['rank']}] ACN:{row['metadata'].get('acn', '')} "
            f"({row['metadata'].get('date', '')}) - "
            f"Risk: {row['metadata'].get('precursor_score', 0):.2f} | "
            f"Quadrant: {row['metadata'].get('quadrant', '')}"
            for row in retrieved
        ]
    )

    return f"{answer}\n\n**Sources:**\n{citations}"


def retrieve_incidents(
    question: str,
    collection,
    embedding_model: SentenceTransformer,
    n_results: int = 5,
    filter_quadrant: Optional[str] = None,
    min_precursor_score: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Retrieve incidents from ChromaDB and return UI-friendly rows."""
    if not question.strip() or collection.count() == 0:
        return []

    where_filter = _build_where_filter(filter_quadrant, min_precursor_score)
    query_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")
    distances = _first_result_list(results, "distances")

    rows = []
    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        1,
    ):
        rows.append(
            {
                "rank": rank,
                "document": str(document),
                "metadata": dict(metadata),
                "similarity": round(1 - float(distance), 3),
            }
        )

    return rows


def _first_result_list(results: Any, key: str) -> list[Any]:
    """Safely return the first list from a Chroma query result key."""
    value = results.get(key)
    if not value:
        return []

    first = value[0]
    if not first:
        return []

    return list(first)


def _text_from_anthropic_response(response: Any) -> str:
    """Extract text blocks from an Anthropic message response."""
    text_parts = []

    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                text_parts.append(str(text))

    answer = "\n".join(text_parts).strip()
    return answer or "No text response was returned by the model."


def run_all_demos(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
) -> None:
    """Run all demo queries and log results."""
    for index, demo in enumerate(DEMO_QUERIES, 1):
        logger.info("=" * 65)
        logger.info("DEMO QUERY %d: %s", index, demo["question"][:70])
        logger.info("Why: %s", demo["why"])

        filter_kwargs = demo.get("filter_kwargs", {})
        if filter_kwargs:
            logger.info("Filter: %s", filter_kwargs)

        answer = rag_query(
            demo["question"],
            collection,
            embedding_model,
            **filter_kwargs,
        )
        logger.info("Answer:\n%s", answer)