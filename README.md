# ASRS Aviation Safety Early Warning System

<!-- markdownlint-disable MD024 -->

A five-layer early warning system for emerging aviation safety risks,
built on NASA ASRS incident report data (2018–2026, 43,829 incidents).
Built as a case study for the IATA Data Science interview (June 2026).

---

## Key findings

| Finding | Detail |
| --- | --- |
| **GNSS spoofing emergence** | CUSUM fired first alarm **April 2024** on spoofing/jamming narratives. Pre-2023 mean: 4.2/month → 2024+ mean: 10.5/month (**2.5× baseline**). Consistent with IATA's published +193% spoofing increase in the 2025 Safety Report. |
| **Post-COVID deferred maintenance** | Equipment Critical incidents spiked **May 2022** — flagged by SPC within the first month of the return-to-service surge. 13 consecutive alarm months followed. |
| **Ghost target incident** | ACN from January 2023: ATC controller reports cloned transponder codes across a 20nm radius. Isolation Forest flagged it from **structured features alone** — no text, no keywords — two years before IATA published spoofing statistics. |
| **BERTopic independent validation** | BERTopic discovered GPS spoofing (Topic 12, 499 docs) and 5G/altimeter interference (Topic 27, 106 docs) as **separate clusters without being told they exist**. |
| **Risk quadrant** | 818 RED incidents (novel + anomalous), 14,770 ORANGE, 1,374 YELLOW, 26,867 GREEN across the full 2018–2026 corpus. |

---

## The five-layer architecture

| Layer | What it does | Approach | Status |
| --- | --- | --- | --- |
| **1 — Dual anomaly detection** | Flags statistically unusual and structurally novel incidents | SPC (STL + CUSUM) + Isolation Forest → 2×2 risk quadrant | Done |
| **2 — Semantic pattern discovery** | Discovers and tracks emerging topic clusters in narrative text | BERTopic (UMAP + HDBSCAN + c-TF-IDF) + topics over time | Done |
| **3 — Rule-based risk scoring** | Scores every incident on five human-factors precursor categories | Transparent term-matching scorer, fully auditable | Done |
| **4 — RAG analyst assistant** | Answers natural-language queries over flagged incidents with citations | ChromaDB + sentence-transformers + Claude Sonnet 4.6 | Done |
| **5 — Production agent design** | LangGraph multi-agent orchestration blueprint | Mermaid diagram — Prognosis pattern applied to aviation safety | Done |

---

## Data

- **Source:** [NASA ASRS Database](https://asrs.arc.nasa.gov/search/database.html)
- **Coverage:** January 2018 – March 2026
- **Format:** `.xls` files (TSV with two header rows despite the extension)
- **Records:** 43,829 incidents after merging 16 batch files
- **IF Baseline:** 2018–2019 only (12,058 records) — 2020 excluded because COVID disrupted flight operations from March 2020 onward

---

## Layer 1 — Dual Anomaly Detection

### What it does

Two independent anomaly signals are computed per incident and combined into a 2×2 risk quadrant:

- **SPC (Statistical Process Control):** STL seasonal decomposition on monthly incident counts per anomaly category. Two-sided CUSUM (k=0.5, h=5.0) on standardised residuals. Detects when a category shifts to a new frequency regime.
- **Isolation Forest:** Trained on 2018–2019 structured features (operator type, flight phase, anomaly category, who detected it). Scores every subsequent incident for novelty — how unlike the pre-COVID baseline it is. High score = the model has never seen this combination of features before.

### Modules

| Module | Role |
| --- | --- |
| [`src/data_loader.py`](src/data_loader.py) | Loads and merges ASRS TSV batch files; parses two-row header; exports `asrs_merged.parquet` |
| [`src/spc.py`](src/spc.py) | STL decomposition + two-sided CUSUM per anomaly category; `run_spc_pipeline()` returns monthly series + alarm dates |
| [`src/anomaly.py`](src/anomaly.py) | Isolation Forest training on 2018–2019 baseline; 2×2 quadrant assignment; `plot_gnss_emergence()` for the spoofing CUSUM chart |

### Runner scripts

| Script | Purpose |
| --- | --- |
| [`run_layer1.py`](run_layer1.py) | Full Layer 1 pipeline: load → SPC → IF → quadrant → save `asrs_layer1.parquet` |
| [`run_gnss_demo.py`](run_gnss_demo.py) | Standalone GNSS spoofing emergence chart (tight spoofing/jamming regex + SPC) |
| [`run_equipment_spc.py`](run_equipment_spc.py) | Standalone Equipment Critical SPC chart (post-COVID maintenance spike) |
| [`run_red_incidents.py`](run_red_incidents.py) | Prints top 5 RED quadrant narratives; saves `red_top20_incidents.csv` |

### Data assets

| Asset | Direction | Description |
| --- | --- | --- |
| `data/raw/*.xls` | Input | 16 ASRS batch files, 6-month intervals (not committed) |
| `outputs/data/asrs_merged.parquet` | Intermediate | Raw merged corpus, 43,829 × ~190 cols, 84.2 MB |
| `outputs/data/asrs_layer1.parquet` | Output | + `if_score`, `if_flag`, `spc_flag`, `quadrant` columns, 84.7 MB |
| `outputs/data/red_top20_incidents.csv` | Output | Top 20 RED quadrant incidents by IF score, 21 rows |

### Output figures

| Figure | Description |
| --- | --- |
| [`outputs/figures/layer1_spc_cusum.png`](outputs/figures/layer1_spc_cusum.png) | Two-panel CUSUM chart for top 5 anomaly categories — monthly counts + cumulative sum with alarm shading |
| [`outputs/figures/2x2_quadrant.png`](outputs/figures/2x2_quadrant.png) | 2×2 risk quadrant scatter: IF novelty score (x) vs SPC alarm flag (y), coloured by quadrant |
| [`outputs/figures/gnss_emergence.png`](outputs/figures/gnss_emergence.png) | GNSS spoofing monthly counts + STL + CUSUM — first alarm April 2024 marked |
| [`outputs/figures/equipment_critical_spc.png`](outputs/figures/equipment_critical_spc.png) | Equipment Critical SPC chart — May 2022 first alarm, 13 alarm months |

### Key results

```text
Quadrant breakdown (43,829 incidents):
  GREEN  (known, normal frequency)      26,867   61.3%
  ORANGE (known, anomalous frequency)   14,770   33.7%
  YELLOW (novel, normal frequency)       1,374    3.1%
  RED    (novel, anomalous frequency)      818    1.9%

SPC first alarms:
  Equipment Critical    May 2022   post-COVID deferred maintenance
  ATC Issues            Feb 2020   COVID groundings -> ATC breakdown
  Procedural Policy     Oct 2019   pre-COVID deviation surge

GNSS spoofing:
  Pre-2023 mean: 4.2/month
  2024+ mean:   10.5/month  (2.5x baseline)
  First CUSUM alarm: April 2024
```

---

## Layer 2 — Semantic Pattern Discovery

### What it does

BERTopic runs on the full narrative corpus (43,829 documents) to discover and track topic clusters independently of the SPC/IF anomaly signals. Key design decisions:

- **Embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) — fast CPU inference, 384 dimensions
- **Dimensionality reduction:** UMAP (`n_components=5`, `low_memory=True`)
- **Clustering:** HDBSCAN (`min_cluster_size=30`)
- **Topic reduction:** `nr_topics=40` (from 94 natural clusters)
- **Temporal tracking:** `topics_over_time` with `nr_bins=99` to track topic volume by year

The GNSS spoofing signal is validated independently: BERTopic found two distinct RF interference clusters (GPS jamming and 5G/altimeter) without any guidance.

### Modules

| Module | Role |
| --- | --- |
| [`src/topics.py`](src/topics.py) | `run_bertopic()`, `find_gnss_topic_ids()`, `get_topic_summary()`, `plot_topic_landscape()`, `plot_gnss_timeline()`, `plot_red_quadrant_topics()`, `plot_topic_heatmap()`, `compute_semantic_drift()` |

### Runner scripts

| Script | Purpose |
| --- | --- |
| [`run_layer2.py`](run_layer2.py) | Full Layer 2: loads `asrs_layer1.parquet`, runs BERTopic, saves model + topic assignments back to parquet |

### Data assets

| Asset | Direction | Description |
| --- | --- | --- |
| `outputs/data/asrs_layer1.parquet` | Input | Layer 1 enriched corpus with quadrant assignments |
| `outputs/data/asrs_layer2.parquet` | Output | + `topic_id`, `topic_label` columns, 43,829 × 206 cols, 84.7 MB |
| `outputs/data/bertopic_model/` | Output | Saved BERTopic model (safetensors serialisation) |
| `outputs/data/layer2_topic_summary.csv` | Output | 39 topics × keywords + document counts, 21 rows |

### Output figures

| Figure | Description |
| --- | --- |
| [`outputs/figures/layer2_topic_landscape.png`](outputs/figures/layer2_topic_landscape.png) | Top 20 BERTopic clusters by document count |
| [`outputs/figures/layer2_gnss_emergence.png`](outputs/figures/layer2_gnss_emergence.png) | GNSS topic (Topic 12) document count by year — BERTopic semantic validation |
| [`outputs/figures/layer2_red_topics.png`](outputs/figures/layer2_red_topics.png) | Topic distribution within RED quadrant incidents |
| [`outputs/figures/layer2_topic_heatmap.png`](outputs/figures/layer2_topic_heatmap.png) | Topic × year heatmap showing growth trajectories |

### Key results

```text
Topics discovered: 39 (reduced from 94 natural HDBSCAN clusters)
Noise documents:   16,376 (38% -- normal for heterogeneous safety text)

Top topics by volume:
  Topic 0:  3,429 docs  Engine incidents
  Topic 1:  3,397 docs  Approach / traffic
  Topic 2:  2,980 docs  Gear / landing / runway
  Topic 3:  2,760 docs  Ground ops / runway incursion
  Topic 9:    807 docs  Drone / UAS conflicts
  Topic 12:   499 docs  GPS / jamming / navigation  <- GNSS cluster
  Topic 13:   424 docs  COVID mask incidents
  Topic 27:   106 docs  5G / altimeter interference <- separate RF cluster

GNSS topic (Topic 12) by year:
  2018: 36  2019: 62  2020: 18  2021: 95  2022: 50
  2023: 80  2024: 151 (peak)    2025: 112 (sustained)
```

---

## Layer 3 — Rule-Based Risk Scorer

### What it does

Assigns a transparent precursor risk score (0–1) to every incident based on five human-factors categories. Deliberately rule-based, not ML.

**Architecture decision:** Every score component maps directly to a known human factors category. A safety analyst can point to exactly which terms drove the score without needing to understand gradient boosting or SHAP values. Appropriate for a proof-of-concept in a regulated safety environment. In production this would be replaced by a LightGBM classifier trained on ASRS/NTSB accident-linkage data.

Five weighted components (each capped at 2 term hits):

| Component | Weight | Example terms |
| --- | --- | --- |
| `fatigue` | 2.5 | fatigue, tired, exhausted, duty time, not rested |
| `near_miss` | 2.5 | nearly, almost, nmac, close call, feet away |
| `comm_breakdown` | 2.0 | miscommunication, wrong frequency, readback, misheard |
| `procedure_deviation` | 1.5 | skipped, omitted, failed to, non-standard, violation |
| `urgency` | 1.5 | emergency, mayday, pan pan, dangerous, critical |

Final score: `min(weighted_sum / max_possible, 1.0)`

A separate GNSS forecast uses LightGBM with lag features (lags 1/2/3/6/12 months + rolling statistics) on the monthly spoofing/jamming count series to project 6 months beyond the data end.

### Modules

| Module | Role |
| --- | --- |
| [`src/risk_scorer.py`](src/risk_scorer.py) | `score_incident()`, `apply_risk_scorer()`, `plot_risk_distribution()`, `export_high_risk_incidents()` |
| [`src/forecasting.py`](src/forecasting.py) | Time-series feature engineering and LightGBM GNSS forecast |

### Runner scripts

| Script | Purpose |
| --- | --- |
| [`run_layer3.py`](run_layer3.py) | Scores all 43,829 incidents; exports risk distribution chart and high-risk CSV |
| [`run_gnss_forecast.py`](run_gnss_forecast.py) | LightGBM 6-month forecast on GNSS monthly count series |

### Data assets

| Asset | Direction | Description |
| --- | --- | --- |
| `outputs/data/asrs_layer2.parquet` | Input | Layer 2 corpus with topic assignments |
| `outputs/data/asrs_layer3.parquet` | Output | + `precursor_score`, `component_fatigue`, `component_near_miss`, `component_comm_breakdown`, `component_procedure_deviation`, `component_urgency`, `high_precursor_risk`; 43,829 × 213 cols, 84.8 MB |
| `outputs/data/layer3_high_risk_incidents.csv` | Output | Top 100 RED/ORANGE incidents by precursor score with narrative preview, 101 rows, 54 KB |

### Output figures

| Figure | Description |
| --- | --- |
| [`outputs/figures/precursor_risk_distribution.png`](outputs/figures/precursor_risk_distribution.png) | Two-panel: risk score histogram (RED+ORANGE) with 90th-pct threshold + mean component scores for top 50 incidents |
| [`outputs/figures/gnss_forecast.png`](outputs/figures/gnss_forecast.png) | GNSS monthly count actuals + LightGBM 6-month forecast (Mar–Aug 2026) showing elevated rate continuing above baseline |

### Key results

```text
Incidents scored: 43,829
Score range:      0.000 - 0.900  (mean: 0.094)
90th-pct threshold: 0.250
High-risk incidents (>= 90th pct): 4,889

Top scoring incident: ACN 2317180 (Dec 2025)
  Score: 0.900  |  Fatigue: 2  |  Near-miss: 2  |  Urgency: 3
  Anomaly: Aircraft Equipment Problem Critical; Inflight Fuel Issue

GNSS forecast (LightGBM, lags 1/2/3/6/12):
  Mar 2026: 10.7  Apr 2026: 12.1  May 2026: 8.1
  Jun 2026: 7.1   Jul 2026: 5.4   Aug 2026: 7.0
  All months above pre-2023 baseline (4.2/month)
```

---

## Layer 4 — RAG Analyst Assistant

### What it does

Indexes all flagged (RED + ORANGE) incidents in ChromaDB. Accepts natural-language analyst queries, retrieves semantically relevant incidents with metadata filtering, and generates cited answers via Claude Sonnet 4.6. Every claim is linked to a specific ACN number.

**Architecture decisions:**

- **ChromaDB (persistent):** Zero server setup; appropriate for a proof of concept. Production target: Qdrant with BGE-M3 for hybrid dense + sparse retrieval (aviation acronyms like TCAS, NMAC need exact term matching alongside semantic search).
- **`all-MiniLM-L6-v2`:** Fast CPU inference, 384 dimensions, cosine similarity. Known limitation: aviation acronyms may not be optimally embedded. Production fix: fine-tune on ASRS narratives + SKYbrary articles.
- **Rich metadata per document:** `quadrant`, `precursor_score`, `if_score`, `spc_flag`, `topic_label`, `component_fatigue/near_miss/comm_breakdown` — enables filtered queries beyond pure semantic search.
- **System prompt forbids confabulation:** "If evidence is insufficient, say so explicitly." Non-hallucination is a hard requirement in safety-critical applications.
- **Persistence:** Index saved to `outputs/data/chromadb/`; subsequent runs load in ~8s, skipping the ~3-minute embed.

### Modules

| Module | Role |
| --- | --- |
| [`src/rag.py`](src/rag.py) | `build_rag_index()`, `rag_query()`, `run_all_demos()`, `DEMO_QUERIES` list |

### Runner scripts

| Script | Purpose |
| --- | --- |
| [`run_layer4.py`](run_layer4.py) | Builds/loads ChromaDB index; runs all 4 DEMO_QUERIES with live Claude API |

### Data assets

| Asset | Direction | Description |
| --- | --- | --- |
| `outputs/data/asrs_layer3.parquet` | Input | Full enriched corpus with all Layer 1–3 signals |
| `outputs/data/chromadb/` | Output | Persistent ChromaDB HNSW index — 3,000 incidents × 384-dim cosine embeddings + SQLite metadata (~82 MB) |

### Index composition

```text
Total indexed:    3,000 incidents (top by precursor_score from RED+ORANGE)
  RED quadrant:     131 incidents
  ORANGE quadrant: 2,869 incidents

Embedding time:   ~173s on CPU (17 incidents/sec) -- first run only
Load time:          ~8s on subsequent runs (from persisted index)
```

### Demo queries

| # | Query | Filter | Tests |
| --- | --- | --- | --- |
| 1 | GPS/navigation errors and unusual radar targets in 2023 | None | GNSS signal retrieval |
| 2 | Communication breakdown patterns between ATC and pilots | None | `comm_breakdown` component |
| 3 | Serious fatigue or inadequate rest incidents | `precursor_score >= 0.3` | Metadata score filter |
| 4 | What do RED quadrant incidents have in common? | `quadrant = RED` | Quadrant metadata filter |

**Live demo query for the presentation:** Query 4 with `filter_quadrant="RED"`. Surfaces three converging patterns: ATC inter-sector handoff failures, weather-induced airspace compression, and compounding simultaneous conflicts.

### Streamlit Interface

A web interface for the RAG query system, built with Streamlit. Run with `uv run streamlit run app.py` — no Jupyter required.

![ASRS RAG Query Interface](outputs/figures/streamlit_demo.png)

### Running additional queries

```python
from src.rag import build_rag_index, rag_query
import pandas as pd

asrs = pd.read_parquet("outputs/data/asrs_layer3.parquet")
collection, client, embedding_model = build_rag_index(asrs)  # loads from disk in ~8s

answer = rag_query(
    "What emerging risks appear in 2024 that were not present in 2019?",
    collection, embedding_model
)
print(answer)
```

---

## Layer 5 — Production Agent Design

LangGraph multi-agent orchestration — the Prognosis pattern applied to aviation safety.
Source: [`architecture/production_agent.mmd`](architecture/production_agent.mmd)

![LangGraph Production Agent Architecture](architecture/production_agent.png)

**Nodes:** Supervisor → Planner → Fan-out workers (ASRS retriever, NTSB linker, Weather enricher, Trend analyser) → Relevancy assessment → Token trimming → Sufficiency loop → Generate brief → Self-critique → HITL interrupt → Publish.

**State:** `EarlyWarningState` TypedDict with `signal_flags`, `incident_ids`, `themes`, `risk_scores`, `enabled_tools`, `gap_reasoning`, `iteration`, `brief_draft`, `analyst_override`. Chunks list uses `operator.add` reducer for parallel fan-in.

---

## Running the full pipeline

```bash
# Install dependencies
uv sync

# Set API key for Layer 4
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# Layer 1: SPC + Isolation Forest + 2x2 quadrant (~2 min)
uv run python run_layer1.py

# Layer 1 standalone charts
uv run python run_gnss_demo.py
uv run python run_equipment_spc.py
uv run python run_red_incidents.py

# Layer 2: BERTopic topic modelling (~15 min CPU)
uv run python run_layer2.py

# Layer 3: Rule-based risk scorer + GNSS forecast (~30 sec)
uv run python run_layer3.py
uv run python run_gnss_forecast.py

# Layer 4: Build ChromaDB index + run demo queries (~3 min first run, ~8s after)
uv run python run_layer4.py

# Layer 4: Streamlit analyst UI (interactive queries)
uv run streamlit run app.py
```

---

## All output files

### Figures

| File | Layer | Description |
| --- | --- | --- |
| [`outputs/figures/layer1_spc_cusum.png`](outputs/figures/layer1_spc_cusum.png) | 1 | CUSUM control charts for top 5 anomaly categories |
| [`outputs/figures/2x2_quadrant.png`](outputs/figures/2x2_quadrant.png) | 1 | 2×2 risk quadrant — 43,829 incidents plotted |
| [`outputs/figures/gnss_emergence.png`](outputs/figures/gnss_emergence.png) | 1 | GNSS spoofing monthly series + CUSUM, first alarm Apr 2024 |
| [`outputs/figures/equipment_critical_spc.png`](outputs/figures/equipment_critical_spc.png) | 1 | Equipment Critical SPC — post-COVID maintenance spike |
| [`outputs/figures/layer2_topic_landscape.png`](outputs/figures/layer2_topic_landscape.png) | 2 | Top 20 BERTopic clusters by document count |
| [`outputs/figures/layer2_gnss_emergence.png`](outputs/figures/layer2_gnss_emergence.png) | 2 | GNSS topic (Topic 12) document count by year |
| [`outputs/figures/layer2_red_topics.png`](outputs/figures/layer2_red_topics.png) | 2 | Topic distribution within RED quadrant incidents |
| [`outputs/figures/layer2_topic_heatmap.png`](outputs/figures/layer2_topic_heatmap.png) | 2 | Topic × year heatmap showing growth trajectories |
| [`outputs/figures/precursor_risk_distribution.png`](outputs/figures/precursor_risk_distribution.png) | 3 | Risk score histogram + component breakdown for top 50 |
| [`outputs/figures/gnss_forecast.png`](outputs/figures/gnss_forecast.png) | 3 | LightGBM 6-month GNSS forecast (Mar–Aug 2026) |

### Data

| File | Layer | Rows x Cols | Size | Description |
| --- | --- | --- | --- | --- |
| `outputs/data/asrs_merged.parquet` | 0 | 43,829 x ~190 | 84.2 MB | Raw merged corpus |
| `outputs/data/asrs_layer1.parquet` | 1 | 43,829 x ~194 | 84.7 MB | + IF score, SPC flag, quadrant |
| `outputs/data/asrs_layer2.parquet` | 2 | 43,829 x 206 | 84.7 MB | + topic_id, topic_label |
| `outputs/data/asrs_layer3.parquet` | 3 | 43,829 x 213 | 84.8 MB | + precursor_score, 5 components, high_risk flag |
| `outputs/data/red_top20_incidents.csv` | 1 | 20 rows | 5 KB | Top RED incidents by IF score |
| `outputs/data/layer2_topic_summary.csv` | 2 | 21 rows | 2 KB | 39 topics x keywords + counts |
| `outputs/data/layer3_high_risk_incidents.csv` | 3 | 100 rows | 54 KB | Top 100 RED/ORANGE by precursor score + narrative preview |
| `outputs/data/bertopic_model/` | 2 | — | — | Saved BERTopic model (safetensors) |
| `outputs/data/chromadb/` | 4 | 3,000 vectors | ~82 MB | Persistent ChromaDB HNSW index + SQLite metadata |

---

## Project structure

```text
asrs-early-warning/
├── CLAUDE.md                        # Build spec v2 (authoritative)
├── pyproject.toml                   # uv dependencies
├── .env                             # ANTHROPIC_API_KEY (never committed)
├── .gitignore
│
├── src/
│   ├── logger.py                    # Shared: console + file logger (all modules use this)
│   ├── data_loader.py               # ASRS TSV parsing, merge, parquet export
│   ├── spc.py                       # STL + two-sided CUSUM pipeline
│   ├── anomaly.py                   # Isolation Forest, 2x2 quadrant, GNSS chart
│   ├── topics.py                    # BERTopic + topics_over_time + semantic drift
│   ├── risk_scorer.py               # Layer 3: rule-based precursor risk scorer
│   ├── forecasting.py               # Layer 3: LightGBM GNSS time-series forecast
│   └── rag.py                       # Layer 4: ChromaDB + Claude RAG
│
├── run_layer1.py                    # Layer 1 full pipeline runner
├── run_layer2.py                    # Layer 2 BERTopic runner
├── run_layer3.py                    # Layer 3 risk scorer runner
├── run_layer4.py                    # Layer 4 RAG demo runner
├── run_gnss_demo.py                 # Standalone: GNSS emergence chart
├── run_equipment_spc.py             # Standalone: Equipment Critical SPC
├── run_red_incidents.py             # Standalone: top RED narratives
├── run_gnss_forecast.py             # Standalone: GNSS 6-month forecast
│
├── app.py                           # Streamlit analyst UI for Layer 4 RAG queries
│
├── architecture/
│   ├── production_agent.mmd         # Layer 5: LangGraph agent Mermaid diagram
│   └── production_agent.png         # Rendered diagram
│
├── notebooks/
│   ├── 01_layer0_data_and_eda.ipynb
│   ├── 02_layer1_anomaly_detection.ipynb
│   ├── 03_layer2_bertopic.ipynb
│   ├── 04_layer3_risk_scorer.ipynb
│   └── 05_layer4_rag.ipynb
│
├── data/
│   └── raw/                         # ASRS .xls batch files (not committed)
│
└── outputs/
    ├── figures/                     # All generated charts (10 files)
    └── data/                        # Parquet datasets, CSVs, model files
```

---

## Reproducibility note

All layers except Layer 4 reproduce fully without an API key.
Layer 4 requires an Anthropic API key in `.env` for answer generation.
The ChromaDB index persists to `outputs/data/chromadb/` after first build
and reloads in ~8s on subsequent runs.

A Streamlit analyst UI is included (`app.py`). Run with:

```bash
uv run streamlit run app.py
```

## Environment

```bash
uv sync   # installs all dependencies from pyproject.toml
```

Requires Python >= 3.11. All dependencies managed via `uv`.
Requires `ANTHROPIC_API_KEY` in `.env` for Layer 4 RAG demo only.

---

## Three numbers to cite in the presentation

- **818** RED quadrant incidents — novel AND anomalous
- **April 2024** — first CUSUM alarm on GNSS spoofing/jamming
- **2.5x** — GNSS narrative rate, pre-2023 baseline vs 2024
