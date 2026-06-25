"""
Layer 4: ChromaDB RAG demo.

Architecture decisions (know these for cross-examination):
- ChromaDB in-memory: zero server setup, appropriate for demo.
  Production: Qdrant with BGE-M3 for hybrid dense+sparse retrieval.
- all-MiniLM-L6-v2: fast CPU, 384-dim, good quality for macro retrieval.
  Production: fine-tune on ASRS + SKYbrary for aviation acronyms.
- Rich metadata: enables analyst-style filtered queries beyond semantic search.
- System prompt forbids confabulation: non-hallucination is a hard requirement
  in safety-critical applications, not a nice-to-have.

API key loaded from .env only — NEVER commit .env.
"""
import os
import math
import time
from typing import Optional
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from src.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


# ── DEMO QUERIES — run all four before Monday, know what each returns ──────
DEMO_QUERIES = [
    {
        "question": (
            "What patterns appear in incidents involving GPS, "
            "navigation errors, or unusual radar targets in 2023?"
        ),
        "filter_kwargs": {},
        "why": (
            "Opens the GNSS story — should surface ghost target "
            "incident and spoofing narratives"
        ),
    },
    {
        "question": (
            "Which incidents show communication breakdown between "
            "ATC and pilots, and what were the outcomes?"
        ),
        "filter_kwargs": {},
        "why": "Tests comm_breakdown precursor component",
    },
    {
        "question": (
            "Show me the most serious incidents where pilots "
            "reported fatigue or inadequate rest."
        ),
        "filter_kwargs": {"min_precursor_score": 0.3},
        "why": "Tests metadata filtering by risk score",
    },
    {
        "question": (
            "What do the RED quadrant incidents — those that are "
            "both novel and anomalously frequent — have in common?"
        ),
        "filter_kwargs": {"filter_quadrant": "RED"},
        "why": "Tests quadrant filtering — should return IF+SPC flagged",
    },
]


def _safe_float(val: object, default: float = 0.0) -> float:
    """Cast to float, replacing NaN/None with default."""
    try:
        v = float(val)
        return default if (v != v) else v  # v != v is True for NaN
    except (TypeError, ValueError):
        return default


def _safe_int(val: object, default: int = 0) -> int:
    try:
        v = float(val)
        return default if (v != v) else int(v)
    except (TypeError, ValueError):
        return default


def build_rag_index(
    asrs: pd.DataFrame,
    max_incidents: int = 3000,
    persist_dir: str = "outputs/data/chromadb",
    force_rebuild: bool = False,
) -> tuple:
    """
    Index RED and ORANGE quadrant incidents in ChromaDB.

    Selection: RED+ORANGE only, sorted by precursor_score descending,
    capped at max_incidents. High-risk incidents get priority slots.

    Persistence: saved to persist_dir so the 3-min embed only runs once.
    Set force_rebuild=True to re-embed from scratch.

    Returns: (collection, client, embedding_model)
    Keep embedding_model — reuse it in rag_query to avoid reloading.
    """
    logger.info("Loading embedding model: all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Try loading persisted index first
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    if not force_rebuild:
        try:
            collection = client.get_collection("asrs_flagged")
            existing = collection.count()
            if existing > 0:
                logger.info("Loaded existing index from %s: %s incidents (skip embed)", persist_dir, f"{existing:,}")
                return collection, client, embedding_model
        except Exception:
            pass

    # Build fresh
    flagged = (
        asrs[asrs['quadrant'].isin(['RED', 'ORANGE'])]
        .copy()
    )

    if len(flagged) > max_incidents:
        flagged = (flagged
                   .sort_values('precursor_score', ascending=False)
                   .head(max_incidents))

    logger.info("Incidents to index: %s (RED: %d, ORANGE: %d)",
                f"{len(flagged):,}",
                (flagged['quadrant'] == 'RED').sum(),
                (flagged['quadrant'] == 'ORANGE').sum())

    narratives = flagged['full_narrative'].fillna('').tolist()

    logger.info("Embedding %s narratives (CPU) ...", f"{len(narratives):,}")
    t0 = time.time()
    embeddings = embedding_model.encode(
        narratives,
        show_progress_bar=True,
        batch_size=64,
    )
    elapsed = time.time() - t0
    logger.info("Embedding complete in %.1fs (%.0f incidents/sec)", elapsed, len(narratives) / elapsed)

    # Build ChromaDB collection with cosine distance
    try:
        client.delete_collection("asrs_flagged")
    except Exception:
        pass
    collection = client.create_collection(
        "asrs_flagged",
        metadata={"hnsw:space": "cosine"},  # cosine similarity, not L2
    )

    # Build metadata — all values must be primitive (str/int/float/bool), no NaN
    metadatas = []
    for _, row in flagged.iterrows():
        date_str = str(row.get('date', ''))[:10]
        try:
            year = int(date_str[:4]) if len(date_str) >= 4 else 2020
        except ValueError:
            year = 2020

        meta = {
            'acn':               str(row.get('ACN', '')),
            'date':              date_str,
            'year':              year,
            'anomaly':           str(row.get('Events | Anomaly', ''))[:200],
            'flight_phase':      str(row.get('Aircraft 1 | Flight Phase', 'Unknown')),
            'quadrant':          str(row.get('quadrant', 'UNKNOWN')),
            'spc_flag':          _safe_int(row.get('spc_flag', 0)),
            'if_score':          round(_safe_float(row.get('if_score', 0.0)), 3),
            'precursor_score':   round(_safe_float(row.get('precursor_score', 0.0)), 3),
            'topic_label':       str(row.get('topic_label', 'Unknown'))[:100],
            'component_fatigue': _safe_int(row.get('component_fatigue', 0)),
            'component_near_miss': _safe_int(row.get('component_near_miss', 0)),
            'component_comm_breakdown': _safe_int(row.get('component_comm_breakdown', 0)),
        }
        metadatas.append(meta)

    ids = [f"incident_{i}" for i in range(len(narratives))]

    collection.add(
        documents=narratives,
        embeddings=[e.tolist() for e in embeddings],
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

    Returns cited answer. Every claim traceable to an ACN number.

    System prompt: explicitly forbids confabulation.
    Metadata filters: enable analyst-style targeted queries.
    ChromaDB where: single filter or $and for multiple conditions.
    """
    # Build where filter — ChromaDB requires $and for multiple conditions
    filters = []
    if filter_quadrant:
        filters.append({'quadrant': {'$eq': filter_quadrant}})
    if min_precursor_score is not None:
        filters.append({'precursor_score': {'$gte': min_precursor_score}})

    if len(filters) == 0:
        where = None
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = {'$and': filters}

    # Clamp n_results to collection size to avoid ChromaDB error
    n = min(n_results, collection.count())

    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        where=where,
        include=['documents', 'metadatas', 'distances'],
    )

    # Format context — rich metadata for each retrieved incident
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0],
    )):
        similarity = round(1 - dist, 3)
        context_parts.append(
            f"[{i+1}] ACN:{meta['acn']} | Date:{meta['date']} | "
            f"Phase:{meta['flight_phase']}\n"
            f"Quadrant:{meta['quadrant']} | "
            f"Risk:{meta['precursor_score']:.2f} | "
            f"IF:{meta['if_score']:.2f} | Similarity:{similarity:.3f}\n"
            f"Anomaly: {meta['anomaly'][:120]}\n"
            f"Narrative: {doc[:500]}\n"
        )

    context = ("\n" + "─" * 60 + "\n").join(context_parts)

    # Generate cited answer — safety-focused system prompt
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not found in .env"

    claude = anthropic.Anthropic(api_key=api_key)

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system="""You are an aviation safety analyst assistant.
Your role is to identify patterns in incident reports that could indicate emerging safety risks.

Rules:
1. Cite specific incidents using [1], [2], [3] etc.
2. ONLY use information from the provided incident reports.
3. If evidence is insufficient, say so explicitly — do not speculate.
4. Focus on patterns across multiple incidents, not individual cases.
5. Note risk scores and quadrant classifications where relevant.
6. Never hallucinate facts not present in the reports.""",
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved incidents:\n{context}"
            ),
        }],
    )

    answer = response.content[0].text

    citations = "\n".join([
        f"  [{i+1}] ACN:{meta['acn']} ({meta['date']}) — "
        f"Risk: {meta['precursor_score']:.2f} | "
        f"Quadrant: {meta['quadrant']}"
        for i, meta in enumerate(results['metadatas'][0])
    ])

    return f"{answer}\n\n**Sources:**\n{citations}"


def run_all_demos(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
) -> None:
    """Run all four DEMO_QUERIES and print results. Test all before Monday."""
    for i, demo in enumerate(DEMO_QUERIES, 1):
        logger.info("=" * 65)
        logger.info("DEMO QUERY %d: %s", i, demo['question'][:70])
        logger.info("Why: %s", demo['why'])
        if demo['filter_kwargs']:
            logger.info("Filter: %s", demo['filter_kwargs'])

        answer = rag_query(
            demo['question'],
            collection,
            embedding_model,
            **demo['filter_kwargs'],
        )
        logger.info("Answer:\n%s", answer)
