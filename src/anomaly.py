import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

from src.helper import (
    build_spc_flag_for_incidents,
    get_if_feature_columns,
)

from src.logger import get_logger
from src.plotter import plot_2x2_quadrant

logger = get_logger(__name__)


def build_isolation_forest(
    asrs: pd.DataFrame,
    contamination: float = 0.05,
) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Train Isolation Forest on a 2018-2019 pre-COVID baseline and score all years.

    Categorical ASRS fields are one-hot encoded, not ordinally encoded, so the
    model does not infer false numeric ordering between categories. Calendar year
    is deliberately excluded from features; otherwise later years become novel by
    construction rather than because their incident patterns changed.
    """
    feature_cols = get_if_feature_columns(asrs)

    if not feature_cols:
        raise ValueError(
            "No expected feature columns found in dataframe. "
            f"Available columns: {list(asrs.columns[:20])}"
        )

    baseline_mask = asrs["date"].dt.year.isin([2018, 2019])
    if baseline_mask.sum() == 0:
        years = sorted(asrs["date"].dt.year.dropna().unique())[:2]
        baseline_mask = asrs["date"].dt.year.isin(years)
        logger.warning("No 2018-2019 data found, using %s as IF baseline", years)

    if baseline_mask.sum() == 0:
        raise ValueError("No dated records available for Isolation Forest baseline")

    logger.info("Using %d feature columns: %s", len(feature_cols), feature_cols)

    cat_features = asrs[feature_cols].fillna("UNKNOWN").astype(str)
    train_cat = cat_features.loc[baseline_mask].to_numpy()
    all_cat = cat_features.to_numpy()

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    x_cat_train = np.asarray(encoder.fit_transform(train_cat), dtype=np.float32)
    x_cat_all = np.asarray(encoder.transform(all_cat), dtype=np.float32)

    month = asrs["date"].dt.month.fillna(0).astype(float)
    month_features = pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=asrs.index,
    ).to_numpy(dtype=np.float32)

    x_train = np.hstack([
        x_cat_train,
        month_features[baseline_mask.to_numpy()],
    ])
    x_all = np.hstack([
        x_cat_all,
        month_features,
    ])

    logger.info(
        "Training Isolation Forest on %s baseline records",
        f"{x_train.shape[0]:,}",
    )

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(x_train)

    raw_scores = iso.decision_function(x_all)

    asrs = asrs.copy()
    asrs["if_score"] = 1 - (
        (raw_scores - raw_scores.min()) /
        (raw_scores.max() - raw_scores.min() + 1e-9)
    )
    asrs["if_flag"] = (
        asrs["if_score"] > asrs["if_score"].quantile(0.95)
    ).astype(int)

    logger.info(
        "Flagged %s anomalous incidents (%.1f%% of corpus)",
        f"{asrs['if_flag'].sum():,}",
        asrs["if_flag"].mean() * 100,
    )

    return asrs, iso


def build_2x2_quadrant(
    asrs: pd.DataFrame,
    spc_results: dict,
    save_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    Create the 2x2 risk quadrant.

    X-axis: novel (IF flagged) vs known.
    Y-axis: normal frequency vs anomalous frequency (SPC).

    An incident is SPC-flagged only when one of its own anomaly categories
    breached CUSUM in that same month.
    """
    asrs = asrs.copy()
    asrs["spc_flag"] = build_spc_flag_for_incidents(asrs, spc_results).astype(int)

    novel = asrs["if_flag"].eq(1)
    anomalous = asrs["spc_flag"].eq(1)

    asrs["quadrant"] = np.select(
        [
            novel & anomalous,
            ~novel & anomalous,
            novel & ~anomalous,
        ],
        ["RED", "ORANGE", "YELLOW"],
        default="GREEN",
    )

    counts = asrs["quadrant"].value_counts()
    logger.info("2x2 Quadrant Summary:")
    for quadrant in ["RED", "ORANGE", "YELLOW", "GREEN"]:
        count = counts.get(quadrant, 0)
        pct = count / len(asrs) * 100
        logger.info("  %s: %s incidents (%.1f%%)", quadrant, f"{count:,}", pct)

    plot_2x2_quadrant(counts, save_dir=save_dir)

    return asrs
