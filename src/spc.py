import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import STL
from src.logger import get_logger

logger = get_logger(__name__)


def explode_anomaly_field(asrs: pd.DataFrame) -> pd.DataFrame:
    """
    The Events | Anomaly field contains semicolon-separated values.
    Create one binary column per unique anomaly type.
    Returns asrs with new anomaly_* columns added.
    """
    anomaly_col = asrs['Events | Anomaly'].fillna('')

    all_types = set()
    for val in anomaly_col:
        for t in val.split(';'):
            t = t.strip()
            if t and len(t) > 2:
                all_types.add(t)

    logger.info("Found %d unique anomaly types", len(all_types))

    # Build all binary columns at once, then concat to avoid fragmentation
    binary_cols = {}
    for atype in all_types:
        safe_name = (atype[:50].strip()
                     .replace(' ', '_')
                     .replace('/', '_')
                     .replace('(', '')
                     .replace(')', ''))
        binary_cols[f"anomaly_{safe_name}"] = anomaly_col.str.contains(
            atype.replace('(', r'\(').replace(')', r'\)'),
            regex=True, na=False
        ).astype(int)

    return pd.concat([asrs, pd.DataFrame(binary_cols, index=asrs.index)], axis=1)


def get_top_anomaly_categories(asrs: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """Return the top_n most frequent anomaly categories by incident count."""
    anomaly_col = asrs['Events | Anomaly'].fillna('')
    counts = {}
    for val in anomaly_col:
        for t in val.split(';'):
            t = t.strip()
            if t and len(t) > 2:
                counts[t] = counts.get(t, 0) + 1
    return pd.Series(counts).sort_values(ascending=False).head(top_n)


def run_spc_pipeline(asrs: pd.DataFrame,
                     category_value: str,
                     k: float = 0.5,
                     h: float = 5.0,
                     min_monthly: int = 5,
                     start_date: str = '2018-01-01') -> dict | None:
    """
    Full SPC pipeline (CUSUM) for one anomaly category.

    k: CUSUM allowance — 0.5 detects 1-sigma shifts
    h: Control limit — 5 → ARL ≈ 500 in-control observations
    min_monthly: skip categories with fewer average monthly incidents
    start_date: clip series to avoid artefacts from sparse pre-history records
    """
    mask = asrs['Events | Anomaly'].fillna('').str.contains(
        category_value.replace('(', r'\(').replace(')', r'\)'),
        case=False, na=False
    )
    # Clip to start_date to avoid step-change artefacts from stray outlier records
    if start_date:
        mask = mask & (asrs['date'] >= pd.Timestamp(start_date))
    subset = asrs[mask].copy()

    monthly = (subset.groupby(subset['date'].dt.to_period('M'))
               ['ACN'].count()
               .to_timestamp()
               .resample('MS').sum()
               .fillna(0))

    if len(monthly) < 24 or monthly.mean() < min_monthly:
        logger.debug("Skipping '%s': %d months, mean %.1f/month", category_value, len(monthly), monthly.mean())
        return None

    # STL decomposition — remove seasonality before CUSUM
    stl = STL(monthly, period=12, robust=True)
    stl_result = stl.fit()
    residual = stl_result.resid

    mu = np.mean(residual)
    sigma = np.std(residual, ddof=1)
    if sigma < 1e-6:
        return None
    z = (residual - mu) / sigma

    # Two-sided CUSUM
    s_pos, s_neg = [0.0], [0.0]
    alarms = []
    for i, x in enumerate(z[1:], 1):
        s_pos.append(max(0.0, s_pos[-1] + x - k))
        s_neg.append(max(0.0, s_neg[-1] - x - k))
        if s_pos[-1] > h or s_neg[-1] > h:
            alarms.append(monthly.index[i])

    logger.info("'%s': %d months, mean %.1f/month, %d CUSUM alarms",
                category_value, len(monthly), monthly.mean(), len(alarms))

    return {
        'category': category_value,
        'monthly_counts': monthly,
        'trend': stl_result.trend,
        'seasonal': stl_result.seasonal,
        'residual': residual,
        's_pos': np.array(s_pos),
        's_neg': np.array(s_neg),
        'alarms': alarms,
        'control_limit': h,
    }


def plot_spc_results(spc_results: dict, save_dir: str = "outputs/figures") -> None:
    """Plot CUSUM charts for all SPC results side by side."""
    os.makedirs(save_dir, exist_ok=True)
    valid = {k: v for k, v in spc_results.items() if v is not None}
    if not valid:
        logger.warning("No valid SPC results to plot")
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for ax_row, (cat, res) in zip(axes, valid.items()):
        # Left: monthly counts + STL trend
        ax_l = ax_row[0]
        ax_l.plot(res['monthly_counts'].index, res['monthly_counts'].values,
                  color='steelblue', alpha=0.5, label='Monthly count')
        ax_l.plot(res['trend'].index, res['trend'].values,
                  color='navy', lw=2, label='STL trend')
        for alarm in res['alarms']:
            ax_l.axvline(alarm, color='red', alpha=0.4, lw=1.5)
        ax_l.set_title(f"{cat[:60]}\nMonthly counts + STL trend", fontsize=9)
        ax_l.legend(fontsize=8)
        ax_l.set_ylabel("Incidents")

        # Right: CUSUM statistic
        ax_r = ax_row[1]
        idx = res['monthly_counts'].index
        ax_r.plot(idx, res['s_pos'], color='red', label='S+ (upward shift)')
        ax_r.plot(idx, res['s_neg'], color='blue', label='S− (downward shift)')
        ax_r.axhline(res['control_limit'], color='black', ls='--',
                     lw=1.5, label=f'h={res["control_limit"]}')
        for alarm in res['alarms']:
            ax_r.axvline(alarm, color='orange', alpha=0.5, lw=1.5)
        ax_r.set_title(f"CUSUM statistic ({len(res['alarms'])} alarms)", fontsize=9)
        ax_r.legend(fontsize=8)
        ax_r.set_ylabel("CUSUM value")

    plt.suptitle("Layer 1: SPC — CUSUM on Top Anomaly Categories\nNASA ASRS 2018–2026",
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, "layer1_spc_cusum.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved SPC chart to %s", out)
    plt.show()
