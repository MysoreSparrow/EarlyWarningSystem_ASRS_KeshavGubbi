"""
Layer 3 runner: rule-based precursor risk scorer.

Applies transparent, auditable risk scoring to all incidents.

Run:
    uv run python run_layer3.py
"""
from pathlib import Path

import pandas as pd

from src.logger import get_logger
from src.plotter import plot_risk_distribution
from src.risk_scorer import (
    apply_risk_scorer,
    export_high_risk_incidents,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"

LAYER2_PATH = DATA_DIR / "asrs_layer2.parquet"
LAYER1_PATH = DATA_DIR / "asrs_layer1.parquet"
LAYER3_PATH = DATA_DIR / "asrs_layer3.parquet"


def _load_input_dataset() -> pd.DataFrame:
    """Load Layer 2 data if present; otherwise fall back to Layer 1."""
    if LAYER2_PATH.exists():
        source_path = LAYER2_PATH
    else:
        source_path = LAYER1_PATH
        logger.warning("Layer 2 not found, falling back to %s", source_path)

    asrs = pd.read_parquet(source_path)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")

    logger.info("Loaded %s records from %s", f"{len(asrs):,}", source_path)
    logger.info("Columns: %d", asrs.shape[1])
    logger.info("Quadrant breakdown:\n%s", asrs["quadrant"].value_counts().to_string())
    return asrs


def _save_layer3_dataset(asrs: pd.DataFrame) -> None:
    """Persist Layer 3 enriched dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_df = asrs.copy()
    for column in save_df.select_dtypes(include=["object"], exclude=["str"]).columns:
        save_df[column] = save_df[column].astype(str)

    save_df.to_parquet(LAYER3_PATH, index=False)
    logger.info("Saved: %s | shape: %s", LAYER3_PATH, asrs.shape)


def main() -> None:
    logger.info("STEP 1: Loading enriched dataset")
    asrs = _load_input_dataset()

    logger.info("STEP 2: Applying rule-based risk scorer")
    asrs = apply_risk_scorer(asrs)

    logger.info(
        "Precursor score distribution:\n%s",
        asrs["precursor_score"].describe().to_string(),
    )

    high_risk_by_quadrant = (
        asrs.groupby("quadrant")["high_precursor_risk"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "high_risk_count", "count": "total", "mean": "rate"})
        .assign(rate=lambda frame: frame["rate"].map("{:.1%}".format))
    )
    logger.info("High-risk incidents by quadrant:\n%s", high_risk_by_quadrant.to_string())

    component_cols = [column for column in asrs.columns if column.startswith("component_")]
    show_cols = [
        "ACN",
        "date",
        "quadrant",
        "precursor_score",
        *component_cols,
        "Events | Anomaly",
    ]
    show_cols = [column for column in show_cols if column in asrs.columns]
    logger.info(
        "Top 5 highest-risk incidents:\n%s",
        asrs.nlargest(5, "precursor_score")[show_cols].to_string(index=False),
    )

    logger.info("STEP 3: Generating risk score distribution chart")
    plot_risk_distribution(asrs)

    logger.info("STEP 4: Exporting top 100 high-risk incidents")
    top100 = export_high_risk_incidents(asrs)

    preview_cols = [
        "ACN",
        "date",
        "quadrant",
        "precursor_score",
        *component_cols[:3],
    ]
    preview_cols = [column for column in preview_cols if column in top100.columns]
    logger.info("Top 10 from export:\n%s", top100.head(10)[preview_cols].to_string(index=False))

    logger.info("STEP 5: Saving Layer 3 parquet")
    _save_layer3_dataset(asrs)

    logger.info("Layer 3 complete.")
    logger.info("Incidents scored: %s", f"{len(asrs):,}")
    logger.info("High-risk incidents: %s", f"{asrs['high_precursor_risk'].sum():,}")


if __name__ == "__main__":
    main()