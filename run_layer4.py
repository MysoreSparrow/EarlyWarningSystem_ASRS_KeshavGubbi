"""
Layer 4 runner: ChromaDB RAG demo.

Loads Layer 3 enriched dataset, indexes RED+ORANGE incidents,
runs all four DEMO_QUERIES with live Claude API.

Run: uv run python run_layer4.py
Requires: ANTHROPIC_API_KEY in .env
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # project root — needed for src.X imports inside modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Claude API responses contain Unicode (em dashes, smart quotes).
# Windows cp1252 console can't encode them — reconfigure stdout to UTF-8.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import pandas as pd
from rag import build_rag_index, rag_query, run_all_demos, DEMO_QUERIES

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading Layer 3 enriched dataset")
print("=" * 65)
src = "outputs/data/asrs_layer3.parquet"
if not os.path.exists(src):
    src = "outputs/data/asrs_layer2.parquet"
    print(f"Layer 3 not found, falling back to {src}")

asrs = pd.read_parquet(src)
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')
print(f"Loaded {len(asrs):,} records  |  columns: {asrs.shape[1]}")
print(f"Quadrant breakdown:")
print(asrs['quadrant'].value_counts().to_string())

flagged_count = (asrs['quadrant'].isin(['RED', 'ORANGE'])).sum()
print(f"\nRED + ORANGE incidents: {flagged_count:,}")
if 'precursor_score' in asrs.columns:
    top_pct = asrs[asrs['quadrant'].isin(['RED','ORANGE'])]['precursor_score'].quantile(0.9)
    print(f"90th-pct precursor score in flagged set: {top_pct:.3f}")

# ── 2. BUILD INDEX ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Building ChromaDB index")
print("=" * 65)
print("Indexing up to 3,000 RED+ORANGE incidents, "
      "prioritised by precursor_score descending")

t_start = time.time()
collection, client, embedding_model = build_rag_index(
    asrs, max_incidents=3000,
    persist_dir="outputs/data/chromadb",
    force_rebuild=False,  # set True to re-embed from scratch
)
t_total = time.time() - t_start

print(f"\nIndex stats:")
print(f"  Total indexed : {collection.count():,}")
print(f"  Wall time     : {t_total:.1f}s")

# ── 3. SPOT-CHECK QUERY (verify index works before API calls) ─────────────────
print("\n" + "=" * 65)
print("STEP 3: Spot-check — semantic search only (no Claude API)")
print("=" * 65)
test_q = "GPS spoofing radar ghost targets"
test_emb = embedding_model.encode([test_q]).tolist()
test_res = collection.query(
    query_embeddings=test_emb, n_results=3,
    include=['documents', 'metadatas', 'distances'],
)
print(f"Query: '{test_q}'")
for i, (doc, meta, dist) in enumerate(zip(
    test_res['documents'][0],
    test_res['metadatas'][0],
    test_res['distances'][0],
), 1):
    # With cosine distance space: dist=0 is identical, dist=1 is orthogonal
    cosine_sim = 1 - dist
    print(f"  [{i}] ACN:{meta['acn']} | {meta['date']} | "
          f"Cosine sim:{cosine_sim:.3f} | Risk:{meta['precursor_score']:.2f}")
    print(f"       {doc[:150].replace(chr(10), ' ')}")

# ── 4. ALL FOUR DEMO QUERIES ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Running all four DEMO_QUERIES with Claude API")
print("=" * 65)
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key or api_key.strip() == "your-api-key-here":
    print("ANTHROPIC_API_KEY not set in .env")
    print("Add your key to .env and re-run step 4 only:")
    print("  from src.rag import rag_query, run_all_demos")
    print("  run_all_demos(collection, embedding_model)")
    print("\nIndex is persisted — no re-embedding needed on next run.")
else:
    print(f"API key: SET  |  Model: claude-sonnet-4-6")
    run_all_demos(collection, embedding_model)

# ── 5. SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"LAYER 4 COMPLETE")
print(f"  Incidents indexed : {collection.count():,}")
print(f"  Embedding time    : {t_total:.1f}s")
print(f"  Demo queries run  : {len(DEMO_QUERIES)}")
print(f"  collection and embedding_model objects are live.")
print(f"  To run additional queries interactively:")
print(f"    from src.rag import rag_query")
print(f"    rag_query('your question', collection, embedding_model)")
print(f"{'='*65}")
