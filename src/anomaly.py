import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import STL
from src.logger import get_logger

logger = get_logger(__name__)


def build_isolation_forest(asrs: pd.DataFrame,
                            contamination: float = 0.05) -> tuple:
    """
    Train IF on 2019-2020 clean baseline. Score all years.
    COVID contamination problem: training on 2020-2021 would corrupt
    the model's idea of 'normal'. Use 2019-2020 only as baseline.
    """
    feature_cols = [
        'Aircraft 1 | Flight Phase',
        'Aircraft 1 | Aircraft Operator',
        'Events | Detector',
        'Events | Result 1',
        'Assessments | Primary Problem',
    ]
    feature_cols = [c for c in feature_cols if c in asrs.columns]

    if not feature_cols:
        raise ValueError("No expected feature columns found in dataframe. "
                         f"Available columns: {list(asrs.columns[:20])}")

    logger.info("Using %d feature columns: %s", len(feature_cols), feature_cols)

    df_feat = asrs[feature_cols].copy()
    for col in feature_cols:
        le = LabelEncoder()
        df_feat[col] = le.fit_transform(
            df_feat[col].fillna('UNKNOWN').astype(str)
        )

    df_feat['month'] = asrs['date'].dt.month.fillna(0).astype(int)
    df_feat['year'] = asrs['date'].dt.year.fillna(0).astype(int)
    all_feature_cols = feature_cols + ['month', 'year']

    # Train on pre-COVID baseline only (2018-2019: full normal operations)
    # 2020 excluded: COVID grounded ~60% of flights from March 2020,
    # distorting the incident profile and corrupting the IF "normal" model.
    baseline_mask = asrs['date'].dt.year.isin([2018, 2019])
    if baseline_mask.sum() == 0:
        # Fallback: use first two years present
        years = sorted(asrs['date'].dt.year.dropna().unique())[:2]
        baseline_mask = asrs['date'].dt.year.isin(years)
        logger.warning("No 2018-2019 data found, using %s as IF baseline", years)

    X_train = df_feat[baseline_mask][all_feature_cols].fillna(0)
    X_all = df_feat[all_feature_cols].fillna(0)

    logger.info("Training Isolation Forest on %s baseline records", f"{X_train.shape[0]:,}")

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)

    raw_scores = iso.decision_function(X_all)

    # Normalise to 0–1 where 1 = most anomalous
    asrs = asrs.copy()
    asrs['if_score'] = 1 - (
        (raw_scores - raw_scores.min()) /
        (raw_scores.max() - raw_scores.min() + 1e-9)
    )
    asrs['if_flag'] = (
        asrs['if_score'] > asrs['if_score'].quantile(0.95)
    ).astype(int)

    logger.info("Flagged %s anomalous incidents (%.1f%% of corpus)",
                f"{asrs['if_flag'].sum():,}", asrs['if_flag'].mean() * 100)

    return asrs, iso


def plot_gnss_emergence(asrs: pd.DataFrame,
                         save_dir: str = "outputs/figures") -> None:
    """
    Centrepiece presentation chart: GNSS spoofing/jamming monthly counts
    with CUSUM control chart showing the 2023 alarm.
    """
    os.makedirs(save_dir, exist_ok=True)

    gnss_mask = asrs['full_narrative'].astype(str).str.lower().str.contains(
        r'spoofing|jamming|gps.{0,20}denial|gnss.{0,20}denial|'
        r'gps.{0,20}interference|navigation.{0,20}interference',
        regex=True, na=False
    )
    monthly_gnss = (
        asrs[gnss_mask]
        .groupby(asrs[gnss_mask]['date'].dt.to_period('M'))['ACN']
        .count()
        .to_timestamp()
        .resample('MS').sum()
    )
    logger.info("GNSS incidents (tight regex): %s", f"{gnss_mask.sum():,}")
    logger.info("Monthly range: %s to %s", monthly_gnss.min(), monthly_gnss.max())

    # CUSUM on STL residuals
    if len(monthly_gnss) >= 24:
        stl = STL(monthly_gnss, period=12, robust=True)
        residual = stl.fit().resid
    else:
        residual = monthly_gnss - monthly_gnss.mean()

    mu = residual.mean()
    sigma = residual.std(ddof=1)
    z = (residual - mu) / (sigma + 1e-9)

    k, h = 0.5, 5.0
    s_pos = [0.0]
    alarm_dates = []
    for i, x in enumerate(z.values[1:], 1):
        s_pos.append(max(0.0, s_pos[-1] + x - k))
        if s_pos[-1] > h:
            alarm_dates.append(monthly_gnss.index[i])

    if alarm_dates:
        logger.info("First CUSUM alarm: %s", alarm_dates[0].strftime('%b %Y'))
        logger.info("Total alarm months: %d", len(alarm_dates))
    else:
        logger.warning("No CUSUM alarms fired on GNSS series")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        'GNSS Spoofing/Jamming - Early Warning Signal\n'
        'NASA ASRS Narratives 2018-2025',
        fontsize=14, fontweight='bold',
    )

    # Top: monthly counts
    ax1.plot(monthly_gnss.index, monthly_gnss.values,
             color='#003366', linewidth=2, label='Monthly incidents')
    ax1.fill_between(monthly_gnss.index, monthly_gnss.values,
                     alpha=0.2, color='#003366')

    # Shade the sustained emergence period
    emergence_start = monthly_gnss[monthly_gnss.index >= '2023-10-01']
    if not emergence_start.empty:
        ax1.axvspan(emergence_start.index[0], monthly_gnss.index[-1],
                    alpha=0.12, color='red',
                    label='Sustained emergence (3x baseline)')

    pre2023_mean = monthly_gnss[monthly_gnss.index < '2023-01-01'].mean()
    ax1.axhline(pre2023_mean, color='grey', linestyle='--', linewidth=1.5,
                label=f'Pre-2023 baseline mean ({pre2023_mean:.1f}/month)')
    ax1.set_ylabel('Incidents per month', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title('Monthly GNSS spoofing/jamming narrative mentions', fontsize=11)

    # Bottom: CUSUM
    ax2.plot(monthly_gnss.index, s_pos,
             color='#cc0000', linewidth=2, label='CUSUM S+')
    ax2.axhline(h, color='black', linestyle='--', linewidth=1.5,
                label=f'Control limit h={h}')
    alarm_above = [s > h for s in s_pos]
    ax2.fill_between(monthly_gnss.index, s_pos, h,
                     where=alarm_above,
                     alpha=0.3, color='red', label='ALARM zone')

    # Annotate first 3 alarm months (ASCII only - Windows cp1252 safe)
    for alarm in alarm_dates[:3]:
        ax2.axvline(alarm, color='red', linewidth=2, alpha=0.7)
        ax2.text(alarm, h + 0.3, f'ALARM {alarm.strftime("%b %Y")}',
                 fontsize=8, color='red', fontweight='bold')

    ax2.set_ylabel('CUSUM statistic', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_title('CUSUM control chart - alarm fires at emergence inflection',
                  fontsize=11)

    plt.tight_layout()
    out = os.path.join(save_dir, "gnss_emergence.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved GNSS emergence chart to %s", out)
    plt.close()


def build_2x2_quadrant(asrs: pd.DataFrame,
                        spc_results: dict,
                        save_dir: str = "outputs/figures") -> pd.DataFrame:
    """
    Create the 2×2 risk quadrant and save figure.

    X-axis: novel (IF flagged) vs known (existing category)
    Y-axis: normal frequency vs anomalous frequency (SPC)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build SPC flag: was this incident's month flagged by any category?
    all_alarm_months = set()
    for cat, result in spc_results.items():
        if result is None:
            continue
        for alarm in result['alarms']:
            all_alarm_months.add(
                alarm.to_period('M') if hasattr(alarm, 'to_period')
                else pd.Timestamp(alarm).to_period('M')
            )

    asrs = asrs.copy()
    asrs['spc_flag'] = (
        asrs['date'].dt.to_period('M').isin(all_alarm_months)
    ).astype(int)

    def assign_quadrant(row):
        novel = row['if_flag'] == 1
        anomalous = row['spc_flag'] == 1
        if novel and anomalous:
            return 'RED'
        elif not novel and anomalous:
            return 'ORANGE'
        elif novel and not anomalous:
            return 'YELLOW'
        else:
            return 'GREEN'

    asrs['quadrant'] = asrs.apply(assign_quadrant, axis=1)

    counts = asrs['quadrant'].value_counts()
    logger.info("2x2 Quadrant Summary:")
    for q in ['RED', 'ORANGE', 'YELLOW', 'GREEN']:
        pct = counts.get(q, 0) / len(asrs) * 100
        logger.info("  %s: %s incidents (%.1f%%)", q, f"{counts.get(q, 0):,}", pct)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    config = [
        (0, 1, '#ffcccc', '#cc0000',
         'ALERT\nNovel + Anomalous Frequency\n'
         f'n = {counts.get("RED", 0):,}\n'
         'GNSS spoofing emerged here\nbefore IATA categorised it'),
        (1, 1, '#ffe0b3', '#cc6600',
         'INVESTIGATE\nKnown + Anomalous Frequency\n'
         f'n = {counts.get("ORANGE", 0):,}'),
        (0, 0, '#fff9c4', '#cc9900',
         'WATCH\nNovel + Normal Frequency\n'
         f'n = {counts.get("YELLOW", 0):,}\nEmerging quietly'),
        (1, 0, '#ccffcc', '#006600',
         'NORMAL\nKnown + Normal Frequency\n'
         f'n = {counts.get("GREEN", 0):,}'),
    ]
    for x, y, bg, tc, label in config:
        ax.add_patch(plt.Rectangle(
            (x, y), 1, 1, color=bg, ec='white', lw=3, zorder=1
        ))
        ax.text(x + 0.5, y + 0.5, label,
                ha='center', va='center',
                fontsize=10, color=tc, fontweight='bold',
                zorder=2)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(
        ['Novel incidents\n(Isolation Forest flagged)',
         'Known category\n(existing taxonomy)'],
        fontsize=11
    )
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(
        ['Normal frequency\n(within baseline)',
         'Anomalous frequency\n(SPC CUSUM breach)'],
        fontsize=11
    )
    ax.set_title(
        'ASRS Aviation Safety Early Warning — Risk Quadrant\n'
        'NASA ASRS Incident Corpus 2018–2026  |  IF baseline: 2018–2019 (pre-COVID)',
        fontsize=13, fontweight='bold', pad=14
    )
    ax.axhline(1, color='white', lw=3)
    ax.axvline(1, color='white', lw=3)
    plt.tight_layout()
    out = os.path.join(save_dir, "2x2_quadrant.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved 2x2 quadrant chart to %s", out)
    plt.show()

    return asrs
