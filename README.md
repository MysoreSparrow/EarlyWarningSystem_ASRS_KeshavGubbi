# ASRS Aviation Safety Early Warning System

A five-layer early warning system for emerging aviation safety risks, built on NASA Aviation Safety Reporting System (ASRS) incident reports.

Built as an IATA Data Science interview case study, June 2026.

---

## What This Repository Does

This project turns ASRS narratives and structured event fields into:

1. anomaly and novelty signals (Layer 1)
2. semantic topic discovery (Layer 2)
3. transparent precursor risk scores (Layer 3)
4. a cited analyst assistant with retrieval (Layer 4)
5. a production-ready agent architecture blueprint (Layer 5)

---

## At a Glance

| Layer | Purpose | Main Output |
| --- | --- | --- |
| Layer 0 | Data ingestion and preparation | asrs_merged.parquet |
| Layer 1 | Frequency anomaly + novelty detection | asrs_layer1.parquet, quadrant labels |
| Layer 2 | BERTopic semantic themes | asrs_layer2.parquet, topic summary |
| Layer 3 | Rule-based precursor scoring | asrs_layer3.parquet, high-risk export |
| Layer 4 | Cited RAG assistant | persisted ChromaDB index, Streamlit app |
| Layer 5 | Production operating design | architecture diagram and flow |

---

## Key Visuals

### Layer 4 Streamlit Prototype

![Streamlit RAG analyst prototype](outputs/figures/streamlit_demo.png)

### Layer 5 Production Architecture

![Layer 5 production agent architecture](architecture/production_agent.png)

---

## Key Results

- 43,829 ASRS incidents processed (2018 to 2026 coverage window)
- Layer 1 quadrant split includes 272 RED incidents
- Layer 2 identifies aviation narrative themes including GNSS-related clusters
- Layer 3 scores all incidents and exports top high-risk RED/ORANGE records
- Layer 4 indexes 3,000 prioritized incidents with ACN-cited responses

---

## Repository Layout

- Core modules: src
- Layer runners: run_layer1.py, run_layer2.py, run_layer3.py, run_layer4.py
- Analyst app: app.py
- Notebooks: notebooks
- Outputs: outputs
- Architecture assets: architecture

---

## Layer Artifacts

### Layer 0

- Data: asrs_merged.parquet
- Notebook: 01_layer0_data_and_eda.ipynb

### Layer 1

- Code: spc.py, anomaly.py, helper.py
- Runner: run_layer1.py
- Data: asrs_layer1.parquet
- Figures:
  - layer1_spc_cusum.png
  - 2x2_quadrant.png
  - gnss_emergence.png
  - equipment_critical_spc.png

### Layer 2

- Code: topics.py
- Runner: run_layer2.py
- Data: asrs_layer2.parquet
- Summary: layer2_topic_summary.csv
- Model: bertopic_model
- Figures:
  - layer2_topic_landscape.png
  - layer2_gnss_emergence.png
  - layer2_red_topics.png
  - layer2_topic_heatmap.png

### Layer 3

- Code: risk_scorer.py
- Runner: run_layer3.py
- Data: asrs_layer3.parquet
- Export: layer3_high_risk_incidents.csv
- Figure: precursor_risk_distribution.png

### Layer 4

- Code: rag.py
- Runner: run_layer4.py
- App: app.py
- Index: chromadb
- Notebook: 05_layer4_rag.ipynb

### Layer 5

- Diagram source: production_agent.mmd
- Rendered diagram: production_agent.png

---

## Quick Start

### Prerequisites

- Python 3.11+
- uv

### Install

    uv sync

### Environment

Create a .env file for Layer 4 only:

    ANTHROPIC_API_KEY=your_key_here

---

## Run the Pipeline

    uv run python run_layer1.py
    uv run python run_layer2.py
    uv run python run_layer3.py
    uv run python run_layer4.py

Run the analyst UI:

    uv run streamlit run app.py

Optional Layer 1 focused runs:

    uv run python run_gnss_demo.py
    uv run python run_equipment_spc.py
    uv run python run_red_incidents.py

---

## Notebook Execution

    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_layer0_data_and_eda.ipynb --ExecutePreprocessor.timeout=1800
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_layer1_anomaly_detection.ipynb --ExecutePreprocessor.timeout=1800
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_layer2_bertopic.ipynb --ExecutePreprocessor.timeout=3600
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_layer3_risk_scorer.ipynb --ExecutePreprocessor.timeout=1800
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_layer4_rag.ipynb --ExecutePreprocessor.timeout=1800

---

## Limitations

- ASRS is voluntary reporting, not a full census of incidents
- Frequency anomalies do not imply causality
- Novelty scores are not severity labels
- Rule-based scoring is transparent but not fully context-aware
- RAG outputs should remain analyst-reviewed

---

