"""
ASRS Aviation Safety Early Warning - Streamlit Demo App
Layer 4: RAG Query Interface

Run:
    uv run streamlit run app.py
"""
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.rag import DEFAULT_CLAUDE_MODEL, build_rag_index, rag_query, retrieve_incidents

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
LAYER3_PATH = DATA_DIR / "asrs_layer3.parquet"
CHROMA_DIR = DATA_DIR / "chromadb"

DEFAULT_MAX_INCIDENTS = 3000
DEFAULT_EMBEDDINGS_LABEL = "all-MiniLM-L6-v2"

SUGGESTED_QUERIES = [
    "GPS spoofing and navigation errors in 2023-2024",
    "ATC and pilot communication breakdown patterns",
    "Fatigue-related incidents near airports",
    "What are the common factors in RED quadrant incidents?",
    "Runway incursion near misses at night",
]


st.set_page_config(
    page_title="ASRS Safety Early Warning",
    page_icon="ASRS",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _quadrant_filter_from_option(option: str) -> str | None:
    """Map UI quadrant option to metadata filter value."""
    if option == "RED only":
        return "RED"
    if option == "ORANGE only":
        return "ORANGE"
    return None


def _build_where_filter(
    quadrant_filter: str | None,
    min_score_filter: float | None,
) -> dict[str, Any] | None:
    """Build ChromaDB where filter for raw retrieval display."""
    filters: list[dict[str, Any]] = []

    if quadrant_filter:
        filters.append({"quadrant": {"$eq": quadrant_filter}})
    if min_score_filter is not None:
        filters.append({"precursor_score": {"$gte": min_score_filter}})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}

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

@st.cache_resource(show_spinner="Building index... first run can take a few minutes")
def load_index():
    """Load or build the ChromaDB index and embedding model."""
    asrs = pd.read_parquet(LAYER3_PATH)
    collection, _client, embedding_model = build_rag_index(
        asrs,
        max_incidents=DEFAULT_MAX_INCIDENTS,
        persist_dir=CHROMA_DIR,
        force_rebuild=False,
    )
    return collection, embedding_model


def render_sidebar() -> tuple[str | None, float | None, int, str]:
    """Render sidebar controls and return selected filter values."""
    with st.sidebar:
        st.title("ASRS Early Warning")
        st.caption("Layer 4 - RAG Query Interface")
        st.divider()

        st.subheader("Query Filters")
        quadrant_option = st.selectbox(
            "Quadrant",
            options=["All (RED + ORANGE)", "RED only", "ORANGE only"],
            help="RED = novel + anomalous frequency.",
        )
        quadrant_filter = _quadrant_filter_from_option(quadrant_option)

        min_score = st.slider(
            "Minimum precursor risk score",
            min_value=0.0,
            max_value=0.8,
            value=0.0,
            step=0.05,
            help="0 = no filter; 0.25 = high-risk queue; 0.5 = very high risk.",
        )
        min_score_filter = min_score if min_score > 0.0 else None

        n_results = st.slider(
            "Incidents to retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of semantically similar incidents sent to Claude.",
        )

        st.divider()
        st.subheader("Suggested queries")
        for suggested_query in SUGGESTED_QUERIES:
            if st.button(suggested_query, use_container_width=True):
                st.session_state["question_input"] = suggested_query

        st.divider()
        st.caption(
            f"**Data:** NASA ASRS 2018-2026, 43,829 incidents  \n"
            f"**Index:** {DEFAULT_MAX_INCIDENTS:,} RED+ORANGE incidents  \n"
            f"**Model:** {os.getenv('ANTHROPIC_MODEL', DEFAULT_CLAUDE_MODEL)}  \n"
            f"**Embeddings:** {DEFAULT_EMBEDDINGS_LABEL}"
        )

    return quadrant_filter, min_score_filter, n_results, quadrant_option


def render_raw_retrieval(
    question: str,
    collection,
    embedding_model,
    n_results: int,
    quadrant_filter: str | None,
    min_score_filter: float | None,
) -> None:
    """Render raw retrieved incidents for debugging/auditability."""
    rows = retrieve_incidents(
        question=question,
        collection=collection,
        embedding_model=embedding_model,
        n_results=n_results,
        filter_quadrant=quadrant_filter,
        min_precursor_score=min_score_filter,
    )

    if not rows:
        st.info("No raw incidents matched the current filters.")
        return

    for row in rows:
        metadata = row["metadata"]
        document = row["document"]
        quadrant = metadata.get("quadrant", "")
        quadrant_marker = "[RED]" if quadrant == "RED" else "[ORANGE]"

        with st.container():
            st.markdown(
                f"**[{row['rank']}]** {quadrant_marker} "
                f"ACN {metadata.get('acn', '')} - {str(metadata.get('date', ''))[:7]} - "
                f"Phase: {metadata.get('flight_phase', '')} - "
                f"Risk: `{metadata.get('precursor_score', 0):.2f}` - "
                f"Similarity: `{row['similarity']:.3f}`"
            )
            preview = str(document)[:400]
            st.caption(preview + ("..." if len(str(document)) > 400 else ""))
            st.divider()


def main() -> None:
    """Render Streamlit app."""
    load_dotenv()

    quadrant_filter, min_score_filter, n_results, quadrant_option = render_sidebar()

    st.title("ASRS Aviation Safety Early Warning")
    st.markdown(
        "Ask any question about the flagged incident corpus. "
        "Every claim in the answer is cited to a specific ACN (ASRS report number)."
    )

    col_question, col_info = st.columns([3, 1])

    with col_info:
        with st.expander("Active filters"):
            st.write(f"**Quadrant:** {quadrant_option}")
            st.write(
                f"**Min risk score:** {min_score_filter:.2f}"
                if min_score_filter is not None
                else "**Min risk score:** none"
            )
            st.write(f"**Results retrieved:** {n_results}")

    with col_question:
        question = st.text_area(
            "Your question:",
            height=100,
            placeholder="e.g. What patterns appear in GPS spoofing incidents from 2024?",
            key="question_input",
        )

    submit = st.button("Analyse", type="primary")

    if submit and not question.strip():
        st.warning("Please enter a question.")
        return

    if not submit:
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in .env. Cannot run Claude synthesis.")
        st.stop()

    try:
        collection, embedding_model = load_index()
    except Exception as exc:
        st.error(f"Failed to load index: {exc}")
        st.stop()

    with st.spinner("Retrieving relevant incidents and synthesising answer..."):
        answer = rag_query(
            question=question.strip(),
            collection=collection,
            embedding_model=embedding_model,
            n_results=n_results,
            filter_quadrant=quadrant_filter,
            min_precursor_score=min_score_filter,
        )

    if answer.startswith("[ERROR]"):
        st.error(answer)
        return

    st.divider()
    st.subheader("Analysis")

    if "**Sources:**" in answer:
        body, sources = answer.split("**Sources:**", 1)
        st.markdown(body.strip())
        with st.expander("Sources (ACN citations)", expanded=True):
            st.markdown("**Sources:**" + sources)
    else:
        st.markdown(answer)

    with st.expander("Retrieved incidents (raw)", expanded=False):
        render_raw_retrieval(
            question=question.strip(),
            collection=collection,
            embedding_model=embedding_model,
            n_results=n_results,
            quadrant_filter=quadrant_filter,
            min_score_filter=min_score_filter,
        )

    st.divider()
    st.caption(
        "5-layer Aviation Safety Early Warning System | "
        "Layers 1-3: SPC + IF + Rule-based scorer | "
        f"Layer 4: ChromaDB RAG + Claude {os.getenv('ANTHROPIC_MODEL', DEFAULT_CLAUDE_MODEL)} | "
        "Layer 5: LangGraph production agent architecture"
    )


if __name__ == "__main__":
    main()