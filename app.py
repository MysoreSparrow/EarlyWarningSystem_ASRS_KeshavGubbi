"""
ASRS Aviation Safety Early Warning — Streamlit Demo App
Layer 4: RAG Query Interface

Run: uv run streamlit run app.py
"""
import sys
import os

import streamlit as st

st.set_page_config(
    page_title="ASRS Safety Early Warning",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load index once and cache ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building index… (first run ~3 min, subsequent runs ~8s)")
def load_index():
    import pandas as pd
    from src.rag import build_rag_index
    asrs = pd.read_parquet("outputs/data/asrs_layer3.parquet")
    collection, client, embedding_model = build_rag_index(
        asrs,
        max_incidents=3000,
        persist_dir="outputs/data/chromadb",
        force_rebuild=False,
    )
    return collection, embedding_model


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ASRS Early Warning")
    st.caption("Layer 4 — RAG Query Interface")
    st.divider()

    st.subheader("Query Filters")
    quadrant_opt = st.selectbox(
        "Quadrant",
        options=["All (RED + ORANGE)", "RED only", "ORANGE only"],
        help="RED = novel + anomalous frequency (highest priority)",
    )
    quadrant_filter = None
    if quadrant_opt == "RED only":
        quadrant_filter = "RED"
    elif quadrant_opt == "ORANGE only":
        quadrant_filter = "ORANGE"

    min_score = st.slider(
        "Minimum precursor risk score",
        min_value=0.0, max_value=0.8, value=0.0, step=0.05,
        help="0 = no filter · 0.25 = top 10% risk · 0.5 = very high risk",
    )
    min_score_filter = min_score if min_score > 0.0 else None

    n_results = st.slider(
        "Incidents to retrieve",
        min_value=3, max_value=10, value=5,
        help="Number of semantically similar incidents sent to Claude",
    )

    st.divider()
    st.subheader("Suggested queries")
    suggested = [
        "GPS spoofing and navigation errors in 2023-2024",
        "ATC and pilot communication breakdown patterns",
        "Fatigue-related incidents near airports",
        "What are the common factors in RED quadrant incidents?",
        "Runway incursion near misses at night",
    ]
    for s in suggested:
        if st.button(s, use_container_width=True):
            st.session_state["question_input"] = s

    st.divider()
    st.caption(
        "**Data:** NASA ASRS 2018–2026 · 43,829 incidents  \n"
        "**Index:** 3,000 RED+ORANGE incidents  \n"
        "**Model:** claude-sonnet-4-6  \n"
        "**Embeddings:** all-MiniLM-L6-v2"
    )


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("✈ ASRS Aviation Safety Early Warning")
st.markdown(
    "Ask any question about the flagged incident corpus. "
    "Every claim in the answer is cited to a specific ACN (ASRS report number)."
)

col_q, col_info = st.columns([3, 1])
with col_info:
    with st.expander("Active filters"):
        st.write(f"**Quadrant:** {quadrant_opt}")
        st.write(f"**Min risk score:** {min_score:.2f}" if min_score > 0 else "**Min risk score:** none")
        st.write(f"**Results retrieved:** {n_results}")

with col_q:
    question = st.text_area(
        "Your question:",
        value=st.session_state.get("question_input", ""),
        height=100,
        placeholder="e.g. What patterns appear in GPS spoofing incidents from 2024?",
        key="question_input",
    )

submit = st.button("Analyse", type="primary", use_container_width=False)

# ── Run query ─────────────────────────────────────────────────────────────────
if submit and question.strip():
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in .env — cannot run Claude synthesis.")
        st.stop()

    try:
        collection, embedding_model = load_index()
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        st.stop()

    from src.rag import rag_query

    with st.spinner("Retrieving relevant incidents and synthesising answer…"):
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
    else:
        st.divider()
        st.subheader("Analysis")

        # Split answer from sources
        if "**Sources:**" in answer:
            body, sources = answer.split("**Sources:**", 1)
            st.markdown(body.strip())
            with st.expander("Sources (ACN citations)", expanded=True):
                st.markdown("**Sources:**" + sources)
        else:
            st.markdown(answer)

        # Show raw retrieved incidents
        with st.expander("Retrieved incidents (raw)", expanded=False):
            q_emb = embedding_model.encode([question.strip()]).tolist()
            filters = []
            if quadrant_filter:
                filters.append({"quadrant": {"$eq": quadrant_filter}})
            if min_score_filter is not None:
                filters.append({"precursor_score": {"$gte": min_score_filter}})
            where = None if not filters else (filters[0] if len(filters) == 1 else {"$and": filters})

            results = collection.query(
                query_embeddings=q_emb,
                n_results=min(n_results, collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                similarity = round(1 - dist, 3)
                quad_color = "🔴" if meta["quadrant"] == "RED" else "🟠"
                with st.container():
                    st.markdown(
                        f"**[{i+1}]** {quad_color} ACN {meta['acn']} · {meta['date'][:7]} · "
                        f"Phase: {meta['flight_phase']} · "
                        f"Risk: `{meta['precursor_score']:.2f}` · "
                        f"Similarity: `{similarity:.3f}`"
                    )
                    st.caption(doc[:400] + ("…" if len(doc) > 400 else ""))
                    st.divider()

elif submit and not question.strip():
    st.warning("Please enter a question.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "5-layer Aviation Safety Early Warning System · "
    "Layers 1-3: SPC + IF + Rule-based scorer · "
    "Layer 4: ChromaDB RAG + Claude claude-sonnet-4-6 · "
    "Layer 5: LangGraph production agent (architecture)"
)
