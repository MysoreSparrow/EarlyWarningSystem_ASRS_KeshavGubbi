"""Export top RED quadrant incidents and log the top narratives."""
from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
LAYER1_PATH = PROJECT_ROOT / "outputs" / "data" / "asrs_layer1.parquet"
OUT_PATH = PROJECT_ROOT / "outputs" / "data" / "red_top20_incidents.csv"

RED_EXPORT_COLUMNS = [
    "ACN",
    "date",
    "Events | Anomaly",
    "Aircraft 1 | Make Model Name",
    "Aircraft 1 | Flight Phase",
    "Assessments | Primary Problem",
    "if_score",
    "spc_flag",
    "full_narrative",
]


def _format_narrative_preview(narrative: str, line_width: int = 100, max_chars: int = 600) -> str:
    """Format a narrative into fixed-width preview lines."""
    narrative = narrative[:max_chars]
    return "\n".join(
        f"      {narrative[index:index + line_width]}"
        for index in range(0, len(narrative), line_width)
    )


def main() -> None:
    asrs = pd.read_parquet(LAYER1_PATH)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")

    missing = [column for column in RED_EXPORT_COLUMNS if column not in asrs.columns]
    if missing:
        raise ValueError(f"Missing required columns for RED export: {missing}")

    red = asrs[asrs["quadrant"] == "RED"].copy()
    red_top = (
        red.nlargest(20, "if_score")[RED_EXPORT_COLUMNS]
        .copy()
    )

    logger.info("Total RED incidents: %s", f"{len(red):,}")
    logger.info(
        "Top 20 RED incidents by IF score:\n%s",
        red_top[
            [
                "ACN",
                "date",
                "if_score",
                "Events | Anomaly",
                "Aircraft 1 | Make Model Name",
                "Aircraft 1 | Flight Phase",
            ]
        ].to_string(index=False),
    )

    logger.info("Top 5 RED narratives:")
    for index, (_, row) in enumerate(red_top.head(5).iterrows(), 1):
        date_str = row["date"].date() if pd.notna(row["date"]) else "unknown"
        logger.info(
            "\n[%d] ACN: %s | %s | IF score: %.4f\n"
            "    Aircraft: %s\n"
            "    Phase:    %s\n"
            "    Anomaly:  %s\n"
            "    Problem:  %s\n"
            "    Narrative:\n%s",
            index,
            row["ACN"],
            date_str,
            row["if_score"],
            row["Aircraft 1 | Make Model Name"],
            row["Aircraft 1 | Flight Phase"],
            str(row["Events | Anomaly"])[:120],
            row["Assessments | Primary Problem"],
            _format_narrative_preview(str(row["full_narrative"])),
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    red_top.drop(columns=["full_narrative"]).to_csv(OUT_PATH, index=False)
    logger.info("Saved: %s", OUT_PATH)


if __name__ == "__main__":
    main()