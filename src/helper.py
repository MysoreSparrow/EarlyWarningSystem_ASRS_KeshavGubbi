from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


IF_BASE_FEATURE_COLUMNS = [
    "Aircraft 1 | Flight Phase",
    "Aircraft 1 | Aircraft Operator",
    "Events | Detector",
    "Assessments | Primary Problem",
]

RESULT_COLUMN_CANDIDATES = (
    "Events | Result",
    "Events | Result 1",
)


def select_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """Return the first candidate column present in the DataFrame."""
    return next((column for column in candidates if column in df.columns), None)


def get_result_column(asrs: pd.DataFrame) -> str | None:
    """Return the ASRS result column, handling known export-name variants."""
    return select_first_existing_column(asrs, RESULT_COLUMN_CANDIDATES)


def get_if_feature_columns(asrs: pd.DataFrame) -> list[str]:
    """
    Return categorical feature columns used by Isolation Forest.

    The ASRS export has appeared with both `Events | Result` and
    `Events | Result 1`; include whichever exists.
    """
    feature_cols = [
        column for column in IF_BASE_FEATURE_COLUMNS
        if column in asrs.columns
    ]

    result_col = get_result_column(asrs)
    if result_col is not None:
        feature_cols.append(result_col)

    return feature_cols


def split_anomaly_categories(value: Any) -> set[str]:
    """Split ASRS semicolon-separated anomaly categories into normalized labels."""
    if value is None or pd.isna(value):
        return set()

    return {
        category.strip()
        for category in str(value).split(";")
        if category.strip()
    }


def has_anomaly_category(value: Any, category: str) -> bool:
    """Return True when a semicolon-separated anomaly field contains category."""
    return category in split_anomaly_categories(value)


def normalize_to_month_period(value: Any) -> pd.Period:
    """Normalize date-like values to a monthly pandas Period."""
    return pd.Timestamp(value).to_period("M")


def build_spc_alarm_months_by_category(
    spc_results: dict,
) -> dict[str, set[pd.Period]]:
    """
    Convert SPC result objects into a category -> alarm-month set.

    Expected SPC result shape:
        {
            "Category name": {
                "alarms": [Timestamp(...), ...],
                ...
            }
        }
    """
    alarm_months_by_category: dict[str, set[pd.Period]] = {}

    for category, result in spc_results.items():
        if result is None:
            continue

        alarm_months: set[pd.Period] = set()
        for alarm in result.get("alarms", []):
            alarm_months.add(normalize_to_month_period(alarm))

        alarm_months_by_category[str(category)] = alarm_months

    return alarm_months_by_category


def build_spc_flag_for_incidents(
    asrs: pd.DataFrame,
    spc_results: dict,
    anomaly_col: str = "Events | Anomaly",
    date_col: str = "date",
) -> np.ndarray:
    """
    Build incident-level SPC flags.

    An incident is flagged only when:
    1. one of its own anomaly categories has an SPC alarm, and
    2. the incident occurred in an alarm month for that category.

    This avoids marking all incidents in an alarm month as anomalous just because
    an unrelated category breached CUSUM.
    """
    if anomaly_col not in asrs.columns:
        raise ValueError(f"Missing required anomaly column: {anomaly_col}")
    if date_col not in asrs.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    alarm_months_by_category = build_spc_alarm_months_by_category(spc_results)

    incident_months = asrs[date_col].dt.to_period("M")
    anomaly_values = asrs[anomaly_col].fillna("").astype(str)

    spc_flag = np.zeros(len(asrs), dtype=bool)

    for category, alarm_months in alarm_months_by_category.items():
        if not alarm_months:
            continue

        category_mask = anomaly_values.map(
            lambda value: has_anomaly_category(value, category)
        )
        month_mask = incident_months.isin(alarm_months)

        spc_flag |= (category_mask & month_mask).to_numpy()

    return spc_flag