import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.helper import has_anomaly_category, split_anomaly_categories
from src.logger import get_logger

logger = get_logger(__name__)

ANOMALY_COL = "Events | Anomaly"
DATE_COL = "date"
ACN_COL = "ACN"


def _safe_anomaly_column_name(category: str, max_len: int = 80) -> str:
    """
    Convert an ASRS anomaly category into a stable binary column name.

    Keeps column names readable while removing punctuation and repeated
    separators. Prefix is added by the caller.
    """
    name = re.sub(r"[^0-9A-Za-z]+", "_", category.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len] or "unknown"


def explode_anomaly_field(
    asrs: pd.DataFrame,
    anomaly_col: str = ANOMALY_COL,
) -> pd.DataFrame:
    """
    Create one binary column per unique ASRS anomaly category.

    The source field is semicolon-separated. Matching is exact by category after
    splitting, not regex substring matching, so categories containing regex
    metacharacters cannot produce false matches.
    """
    if anomaly_col not in asrs.columns:
        raise ValueError(f"Missing required anomaly column: {anomaly_col}")

    category_sets = asrs[anomaly_col].map(split_anomaly_categories)
    all_categories = sorted({
        category
        for categories in category_sets
        for category in categories
        if len(category) > 2
    })

    logger.info("Found %d unique anomaly types", len(all_categories))

    used_names: Counter[str] = Counter()
    binary_cols = {}

    for category in all_categories:
        base_name = _safe_anomaly_column_name(category)
        used_names[base_name] += 1
        suffix = "" if used_names[base_name] == 1 else f"_{used_names[base_name]}"
        column_name = f"anomaly_{base_name}{suffix}"

        binary_cols[column_name] = category_sets.map(
            lambda categories, target=category: target in categories
        ).astype(int)

    if not binary_cols:
        return asrs.copy()

    return pd.concat(
        [asrs, pd.DataFrame(binary_cols, index=asrs.index)],
        axis=1,
    )


def get_top_anomaly_categories(
    asrs: pd.DataFrame,
    top_n: int = 10,
    anomaly_col: str = ANOMALY_COL,
) -> pd.Series:
    """Return the top_n most frequent anomaly categories by incident count."""
    if anomaly_col not in asrs.columns:
        raise ValueError(f"Missing required anomaly column: {anomaly_col}")

    counts: Counter[str] = Counter()
    for value in asrs[anomaly_col]:
        counts.update(
            category
            for category in split_anomaly_categories(value)
            if len(category) > 2
        )

    if not counts:
        return pd.Series(dtype=int)

    return pd.Series(counts).sort_values(ascending=False).head(top_n)


def _build_category_mask(
    asrs: pd.DataFrame,
    category_value: str,
    anomaly_col: str,
) -> pd.Series:
    """Return exact incident mask for one ASRS anomaly category."""
    return asrs[anomaly_col].map(
        lambda value: has_anomaly_category(value, category_value)
    )


def _build_monthly_category_counts(
    asrs: pd.DataFrame,
    category_value: str,
    start_date: str | pd.Timestamp | None,
    anomaly_col: str,
    date_col: str,
    acn_col: str,
) -> pd.Series:
    """Build monthly incident counts for one exact anomaly category."""
    category_mask = _build_category_mask(asrs, category_value, anomaly_col)

    if start_date is not None:
        category_mask = category_mask & (asrs[date_col] >= pd.Timestamp(start_date))

    subset = asrs.loc[category_mask].copy()

    monthly = (
        subset.groupby(subset[date_col].dt.to_period("M"))[acn_col]
        .count()
        .to_timestamp()
        .resample("MS")
        .sum()
        .fillna(0)
    )

    return monthly


def _run_two_sided_cusum(
    z_scores: pd.Series,
    k: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Run two-sided CUSUM over standardized residuals."""
    s_pos = [0.0]
    s_neg = [0.0]
    alarms: list[pd.Timestamp] = []

    for index, value in enumerate(z_scores.iloc[1:], 1):
        s_pos.append(max(0.0, s_pos[-1] + value - k))
        s_neg.append(max(0.0, s_neg[-1] - value - k))

        if s_pos[-1] > h or s_neg[-1] > h:
            alarms.append(z_scores.index[index])

    return np.array(s_pos), np.array(s_neg), alarms


def run_spc_pipeline(
    asrs: pd.DataFrame,
    category_value: str,
    k: float = 0.5,
    h: float = 5.0,
    min_monthly: int = 5,
    start_date: str | pd.Timestamp | None = "2018-01-01",
    anomaly_col: str = ANOMALY_COL,
    date_col: str = DATE_COL,
    acn_col: str = ACN_COL,
) -> dict[str, Any] | None:
    """
    Run STL + two-sided CUSUM for one exact ASRS anomaly category.

    k:
        CUSUM allowance. 0.5 detects roughly one-sigma shifts.
    h:
        Control limit. h=5 is a conventional moderate false-alarm setting.
    min_monthly:
        Skip sparse categories with fewer average monthly incidents.
    start_date:
        Clip series to avoid artefacts from sparse pre-history records.
    """
    required_cols = [anomaly_col, date_col, acn_col]
    missing = [column for column in required_cols if column not in asrs.columns]
    if missing:
        raise ValueError(f"Missing required columns for SPC: {missing}")

    monthly = _build_monthly_category_counts(
        asrs=asrs,
        category_value=category_value,
        start_date=start_date,
        anomaly_col=anomaly_col,
        date_col=date_col,
        acn_col=acn_col,
    )

    if len(monthly) < 24 or monthly.mean() < min_monthly:
        logger.debug(
            "Skipping '%s': %d months, mean %.1f/month",
            category_value,
            len(monthly),
            monthly.mean() if len(monthly) else 0.0,
        )
        return None

    stl_result = STL(monthly, period=12, robust=True).fit()
    residual = stl_result.resid

    mu = residual.mean()
    sigma = residual.std(ddof=1)
    if sigma < 1e-6:
        logger.debug("Skipping '%s': residual sigma too small", category_value)
        return None

    z_scores = (residual - mu) / sigma
    s_pos, s_neg, alarms = _run_two_sided_cusum(z_scores, k=k, h=h)

    logger.info(
        "'%s': %d months, mean %.1f/month, %d CUSUM alarm months",
        category_value,
        len(monthly),
        monthly.mean(),
        len(alarms),
    )

    return {
        "category": category_value,
        "monthly_counts": monthly,
        "trend": stl_result.trend,
        "seasonal": stl_result.seasonal,
        "residual": residual,
        "s_pos": s_pos,
        "s_neg": s_neg,
        "alarms": alarms,
        "control_limit": h,
    }