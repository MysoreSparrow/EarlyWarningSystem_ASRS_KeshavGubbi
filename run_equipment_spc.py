"""Generate the Aircraft Equipment Problem Critical SPC chart."""
from pathlib import Path

import matplotlib
import pandas as pd

from src.logger import get_logger
from src.plotter import plot_equipment_spc
from src.spc import run_spc_pipeline

matplotlib.use("Agg")

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
LAYER1_PATH = PROJECT_ROOT / "outputs" / "data" / "asrs_layer1.parquet"

EQUIPMENT_CRITICAL_CATEGORY = "Aircraft Equipment Problem Critical"


def main() -> None:
    asrs = pd.read_parquet(LAYER1_PATH)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")
    logger.info("Loaded %s records from %s", f"{len(asrs):,}", LAYER1_PATH)

    result = run_spc_pipeline(
        asrs,
        category_value=EQUIPMENT_CRITICAL_CATEGORY,
        start_date="2018-01-01",
    )

    if result is None:
        raise ValueError(
            "SPC returned no result for "
            f"'{EQUIPMENT_CRITICAL_CATEGORY}'. Check category name or data."
        )

    logger.info("Alarms: %d", len(result["alarms"]))
    logger.info(
        "First alarm: %s",
        result["alarms"][0] if result["alarms"] else "none",
    )

    plot_equipment_spc(result)


if __name__ == "__main__":
    main()