"""
Layer 1 runner: load data, explode anomaly field, SPC on top 5 categories,
Isolation Forest, and build the 2x2 risk quadrant.

Run from project root:
    uv run python run_layer1.py
"""
from pathlib import Path

import matplotlib

from src.anomaly import build_2x2_quadrant, build_isolation_forest
from src.data_loader import load_and_merge_asrs
from src.logger import get_logger
from src.plotter import plot_spc_results
from src.spc import (
    explode_anomaly_field,
    get_top_anomaly_categories,
    run_spc_pipeline,
)

matplotlib.use("Agg")

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LAYER1_PATH = DATA_DIR / "asrs_layer1.parquet"

KEY_COLUMNS = [
    "ACN",
    "Time | Date",
    "Events | Anomaly",
    "Report 1 | Narrative",
    "Report 2 | Narrative",
    "Aircraft 1 | Flight Phase",
    "Assessments | Primary Problem",
]

GNSS_SIGNAL_REGEX = (
    r"spoof|jamm|gps.{0,25}interfer|gnss.{0,25}interfer|gps.{0,25}denial|"
    r"gps.{0,25}unreliable|gps.{0,25}degrad|position.{0,25}spoof|"
    r"gps.{0,25}lost|navigation.{0,25}warn|gps.{0,25}alert"
)


def validate_required_columns(asrs) -> None:
    """Validate columns needed for Layer 1 and log key-column completeness."""
    logger.info("Key column availability:")
    for column in KEY_COLUMNS:
        if column in asrs.columns:
            null_pct = asrs[column].isna().mean() * 100
            logger.info("  OK %s - null: %.1f%%", column, null_pct)
        else:
            logger.warning("  MISSING %s - NOT FOUND", column)

    if "Events | Anomaly" not in asrs.columns:
        matching_cols = [
            column
            for column in asrs.columns
            if "anomaly" in column.lower() or "event" in column.lower()
        ]
        raise ValueError(
            "'Events | Anomaly' column not found. "
            f"Available anomaly/event-like columns: {matching_cols}"
        )


def run_spc_for_top_categories(asrs, top_n: int = 5) -> dict:
    """Run SPC over the top anomaly categories and return category results."""
    top_cats = get_top_anomaly_categories(asrs, top_n=15)

    logger.info("Top 15 anomaly categories by incident count:")
    for category, count in top_cats.items():
        pct = count / len(asrs) * 100
        logger.info("  %5s (%.1f%%)  %s", f"{count:,}", pct, category)

    selected_categories = top_cats.head(top_n).index.tolist()
    logger.info("Top %d selected for SPC: %s", top_n, selected_categories)

    spc_results = {}
    for category in selected_categories:
        logger.info("Processing SPC category: %s", category)
        spc_results[category] = run_spc_pipeline(asrs, category_value=category)

    valid_results = {
        category: result
        for category, result in spc_results.items()
        if result is not None
    }

    logger.info(
        "%d/%d categories passed SPC minimum requirements",
        len(valid_results),
        len(selected_categories),
    )
    for category, result in valid_results.items():
        logger.info("  %s: %d CUSUM alarms", category, len(result["alarms"]))
        if result["alarms"]:
            alarm_strs = [str(alarm)[:7] for alarm in result["alarms"][:5]]
            logger.info("    First alarms: %s", ", ".join(alarm_strs))

    if valid_results:
        plot_spc_results(valid_results)

    return spc_results


def save_layer1_dataset(asrs) -> None:
    """Persist Layer 1 enriched dataset to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_df = asrs.copy()
    for column in save_df.select_dtypes(include=["object"], exclude=["str"]).columns:
        save_df[column] = save_df[column].astype(str)

    save_df.to_parquet(LAYER1_PATH, index=False)
    logger.info("Layer 1 enriched dataset saved to: %s", LAYER1_PATH)
    logger.info("Final shape: %s", asrs.shape)


def log_gnss_signal_preview(asrs) -> None:
    """Log the GNSS narrative signal used as a Layer 1 storyline preview."""
    gnss_mask = asrs["full_narrative"].str.lower().str.contains(
        GNSS_SIGNAL_REGEX,
        regex=True,
        na=False,
    )
    gnss_monthly = (
        asrs[gnss_mask]
        .groupby(asrs[gnss_mask]["date"].dt.to_period("M"))["ACN"]
        .count()
    )

    logger.info(
        "GNSS-related incidents: %s (%.1f%% of corpus)",
        f"{gnss_mask.sum():,}",
        gnss_mask.mean() * 100,
    )
    if not gnss_monthly.empty:
        logger.info("GNSS monthly counts:\n%s", gnss_monthly.to_string())


def main() -> None:
    logger.info("STEP 1: Loading ASRS data")
    asrs = load_and_merge_asrs(RAW_DATA_DIR)

    logger.info("DataFrame shape: %s", asrs.shape)
    logger.info("Column sample (first 30): %s", list(asrs.columns[:30]))
    logger.info(
        "Narrative word count stats:\n%s",
        asrs["narrative_word_count"].describe().to_string(),
    )

    validate_required_columns(asrs)

    logger.info("STEP 2: Exploding anomaly field")
    asrs = explode_anomaly_field(asrs)

    logger.info("STEP 3: Running SPC pipeline")
    spc_results = run_spc_for_top_categories(asrs)

    logger.info("STEP 4: Training Isolation Forest")
    asrs, _iso_model = build_isolation_forest(asrs)
    logger.info("IF score distribution:\n%s", asrs["if_score"].describe().to_string())

    logger.info("STEP 5: Building 2x2 risk quadrant")
    asrs = build_2x2_quadrant(asrs, spc_results)

    logger.info("STEP 6: Saving Layer 1 dataset")
    save_layer1_dataset(asrs)

    logger.info("GNSS / navigation narrative signal preview")
    log_gnss_signal_preview(asrs)

    logger.info("Layer 1 complete.")


if __name__ == "__main__":
    main()