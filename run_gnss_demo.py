"""Generate the GNSS emergence presentation chart."""
from pathlib import Path

import matplotlib
import pandas as pd

from src.logger import get_logger
from src.plotter import plot_gnss_emergence

matplotlib.use("Agg")

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
LAYER1_PATH = PROJECT_ROOT / "outputs" / "data" / "asrs_layer1.parquet"


def main() -> None:
    asrs = pd.read_parquet(LAYER1_PATH)
    asrs["date"] = pd.to_datetime(asrs["date"], errors="coerce")

    logger.info("Loaded %s records from %s", f"{len(asrs):,}", LAYER1_PATH)
    plot_gnss_emergence(asrs)


if __name__ == "__main__":
    main()