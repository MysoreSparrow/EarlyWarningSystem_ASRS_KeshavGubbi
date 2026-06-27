from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from statsmodels.tsa.seasonal import STL

from src.logger import get_logger

logger = get_logger(__name__)


GNSS_REGEX = (
    r"spoofing|jamming|gps.{0,20}denial|gnss.{0,20}denial|"
    r"gps.{0,20}interference|navigation.{0,20}interference"
)


def _ensure_save_dir(save_dir: str | Path) -> Path:
    """Create and return a plotting output directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _save_figure(fig, save_dir: str | Path, filename: str) -> Path:
    """Save and close a matplotlib figure."""
    save_dir = _ensure_save_dir(save_dir)
    out = save_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", out)
    plt.close(fig)
    return out


def _plot_series_with_fill(
    ax,
    x,
    y,
    *,
    color: str,
    label: str,
    linewidth: float = 2.0,
    alpha_fill: float = 0.2,
) -> None:
    """Plot a line series with translucent area fill."""
    ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    ax.fill_between(x, y, alpha=alpha_fill, color=color)


def _plot_vertical_markers(
    ax,
    markers: Sequence,
    *,
    color: str = "red",
    alpha: float = 0.5,
    linewidth: float = 1.5,
) -> None:
    """Draw vertical marker lines."""
    for marker in markers:
        ax.axvline(marker, color=color, alpha=alpha, lw=linewidth)


def _plot_cusum_panel(
    ax,
    index,
    s_pos,
    control_limit: float,
    *,
    s_neg=None,
    title: str = "CUSUM control chart",
    mark_first_alarm: bool = False,
    alarms: Sequence | None = None,
) -> None:
    """Plot a CUSUM panel with optional S- and alarm shading."""
    ax.plot(index, s_pos, color="#cc0000", lw=2, label="CUSUM S+")

    if s_neg is not None:
        ax.plot(index, s_neg, color="blue", lw=1.5, label="CUSUM S-")

    ax.axhline(
        control_limit,
        color="black",
        ls="--",
        lw=1.5,
        label=f"Control limit h={control_limit}",
    )

    alarm_above = [value > control_limit for value in s_pos]
    ax.fill_between(
        index,
        s_pos,
        control_limit,
        where=alarm_above,
        alpha=0.3,
        color="red",
        label="ALARM zone",
    )

    if alarms:
        _plot_vertical_markers(ax, alarms, color="red", alpha=0.5)

        if mark_first_alarm:
            first_alarm = alarms[0]
            ax.axvline(first_alarm, color="red", lw=2.5, alpha=0.8)
            ax.text(
                first_alarm,
                control_limit + 0.5,
                f"First alarm\n{first_alarm.strftime('%b %Y')}",
                fontsize=9,
                color="red",
                fontweight="bold",
            )

    ax.set_ylabel("CUSUM statistic", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)


def _barh_with_labels(
    ax,
    labels: Sequence[str],
    values,
    *,
    color: str | Sequence[str],
    alpha: float = 0.8,
    xlabel: str,
    title: str,
    title_size: int = 11,
) -> None:
    """Plot horizontal bars with value labels."""
    values_array = np.asarray(values)
    bars = ax.barh(labels, values_array, color=color, alpha=alpha)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=title_size, fontweight="bold")


def _topic_keywords(topic_model, topic_id: int, n_words: int = 5) -> str:
    """Return compact topic keyword label."""
    words = topic_model.get_topic(topic_id)
    return " | ".join(word for word, _ in words[:n_words]) if words else f"Topic {topic_id}"


def _as_timeline_index(series: pd.Series):
    """Convert PeriodIndex to Timestamp index for plotting."""
    if isinstance(series.index, pd.PeriodIndex):
        return series.index.to_timestamp()
    return series.index


def plot_gnss_emergence(
    asrs: pd.DataFrame,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """
    Plot GNSS spoofing/jamming monthly counts with a CUSUM control chart.

    This is presentation/visualization code, separated from anomaly detection
    model logic.
    """
    save_dir = _ensure_save_dir(save_dir)

    gnss_mask = asrs["full_narrative"].astype(str).str.lower().str.contains(
        GNSS_REGEX,
        regex=True,
        na=False,
    )
    monthly_gnss = (
        asrs[gnss_mask]
        .groupby(asrs[gnss_mask]["date"].dt.to_period("M"))["ACN"]
        .count()
        .to_timestamp()
        .resample("MS")
        .sum()
    )

    logger.info("GNSS incidents (tight regex): %s", f"{gnss_mask.sum():,}")
    logger.info("Monthly range: %s to %s", monthly_gnss.min(), monthly_gnss.max())

    if len(monthly_gnss) >= 24:
        residual = STL(monthly_gnss, period=12, robust=True).fit().resid
    else:
        residual = monthly_gnss - monthly_gnss.mean()

    mu = residual.mean()
    sigma = residual.std(ddof=1)
    z_scores = (residual - mu) / (sigma + 1e-9)

    k = 0.5
    h = 5.0
    s_pos = [0.0]
    alarm_dates = []

    for index, value in enumerate(z_scores.values[1:], 1):
        s_pos.append(max(0.0, s_pos[-1] + value - k))
        if s_pos[-1] > h:
            alarm_dates.append(monthly_gnss.index[index])

    if alarm_dates:
        logger.info("First CUSUM alarm: %s", alarm_dates[0].strftime("%b %Y"))
        logger.info("Total alarm months: %d", len(alarm_dates))
    else:
        logger.warning("No CUSUM alarms fired on GNSS series")

    fig, (ax_counts, ax_cusum) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "GNSS Spoofing/Jamming - Early Warning Signal\n"
        "NASA ASRS Narratives 2018-2025",
        fontsize=14,
        fontweight="bold",
    )

    _plot_series_with_fill(
        ax_counts,
        monthly_gnss.index,
        monthly_gnss.values,
        color="#003366",
        label="Monthly incidents",
        alpha_fill=0.2,
    )

    emergence_start = monthly_gnss[monthly_gnss.index >= "2023-10-01"]
    if not emergence_start.empty:
        ax_counts.axvspan(
            emergence_start.index[0],
            monthly_gnss.index[-1],
            alpha=0.12,
            color="red",
            label="Sustained emergence (3x baseline)",
        )

    pre2023_mean = monthly_gnss[monthly_gnss.index < "2023-01-01"].mean()
    ax_counts.axhline(
        pre2023_mean,
        color="grey",
        linestyle="--",
        linewidth=1.5,
        label=f"Pre-2023 baseline mean ({pre2023_mean:.1f}/month)",
    )
    ax_counts.set_ylabel("Incidents per month", fontsize=11)
    ax_counts.legend(fontsize=9)
    ax_counts.set_title(
        "Monthly GNSS spoofing/jamming narrative mentions",
        fontsize=11,
    )

    ax_cusum.plot(
        monthly_gnss.index,
        s_pos,
        color="#cc0000",
        linewidth=2,
        label="CUSUM S+",
    )
    ax_cusum.axhline(
        h,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Control limit h={h}",
    )

    alarm_above = [value > h for value in s_pos]
    ax_cusum.fill_between(
        monthly_gnss.index,
        s_pos,
        h,
        where=alarm_above,
        alpha=0.3,
        color="red",
        label="ALARM zone",
    )

    for alarm in alarm_dates[:3]:
        ax_cusum.axvline(alarm, color="red", linewidth=2, alpha=0.7)
        ax_cusum.text(
            alarm,
            h + 0.3,
            f"ALARM {alarm.strftime('%b %Y')}",
            fontsize=8,
            color="red",
            fontweight="bold",
        )

    ax_cusum.set_ylabel("CUSUM statistic", fontsize=11)
    ax_cusum.set_xlabel("Date", fontsize=11)
    ax_cusum.legend(fontsize=9)
    ax_cusum.set_title(
        "CUSUM control chart - alarm fires at emergence inflection",
        fontsize=11,
    )

    plt.tight_layout()
    _save_figure(fig, save_dir, "gnss_emergence.png")


def plot_2x2_quadrant(
    counts: pd.Series,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot the 2x2 early-warning risk quadrant from quadrant counts."""
    fig, ax = plt.subplots(figsize=(9, 7))

    config = [
        (
            0,
            1,
            "#ffcccc",
            "#cc0000",
            "ALERT\nNovel + Anomalous Frequency\n"
            f'n = {counts.get("RED", 0):,}\n'
            "GNSS spoofing emerged here\nbefore IATA categorised it",
        ),
        (
            1,
            1,
            "#ffe0b3",
            "#cc6600",
            "INVESTIGATE\nKnown + Anomalous Frequency\n"
            f'n = {counts.get("ORANGE", 0):,}',
        ),
        (
            0,
            0,
            "#fff9c4",
            "#cc9900",
            "WATCH\nNovel + Normal Frequency\n"
            f'n = {counts.get("YELLOW", 0):,}\nEmerging quietly',
        ),
        (
            1,
            0,
            "#ccffcc",
            "#006600",
            "NORMAL\nKnown + Normal Frequency\n"
            f'n = {counts.get("GREEN", 0):,}',
        ),
    ]

    for x, y, background_color, text_color, label in config:
        ax.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                color=background_color,
                ec="white",
                lw=3,
                zorder=1,
            )
        )
        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=text_color,
            fontweight="bold",
            zorder=2,
        )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(
        [
            "Novel incidents\n(Isolation Forest flagged)",
            "Known category\n(existing taxonomy)",
        ],
        fontsize=11,
    )
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(
        [
            "Normal frequency\n(within baseline)",
            "Anomalous frequency\n(SPC CUSUM breach)",
        ],
        fontsize=11,
    )
    ax.set_title(
        "ASRS Aviation Safety Early Warning - Risk Quadrant\n"
        "NASA ASRS Incident Corpus 2018-2026  |  "
        "IF baseline: 2018-2019 (pre-COVID)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.axhline(1, color="white", lw=3)
    ax.axvline(1, color="white", lw=3)

    plt.tight_layout()
    _save_figure(fig, save_dir, "2x2_quadrant.png")


def plot_spc_results(
    spc_results: dict,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot CUSUM charts for all valid SPC results side by side."""
    valid = {
        category: result
        for category, result in spc_results.items()
        if result is not None
    }
    if not valid:
        logger.warning("No valid SPC results to plot")
        return

    n_results = len(valid)
    fig, axes = plt.subplots(n_results, 2, figsize=(16, 4 * n_results))
    if n_results == 1:
        axes = [axes]

    for ax_row, (category, result) in zip(axes, valid.items()):
        ax_left = ax_row[0]
        _plot_series_with_fill(
            ax_left,
            result["monthly_counts"].index,
            result["monthly_counts"].values,
            color="steelblue",
            label="Monthly count",
            alpha_fill=0.12,
        )
        ax_left.plot(
            result["trend"].index,
            result["trend"].values,
            color="navy",
            lw=2,
            label="STL trend",
        )
        _plot_vertical_markers(ax_left, result["alarms"], color="red", alpha=0.4)
        ax_left.set_title(
            f"{category[:60]}\nMonthly counts + STL trend",
            fontsize=9,
        )
        ax_left.legend(fontsize=8)
        ax_left.set_ylabel("Incidents")

        _plot_cusum_panel(
            ax_row[1],
            result["monthly_counts"].index,
            result["s_pos"],
            result["control_limit"],
            s_neg=result["s_neg"],
            title=f"CUSUM statistic ({len(result['alarms'])} alarm months)",
            alarms=result["alarms"],
        )

    plt.suptitle(
        "Layer 1: SPC - CUSUM on Top Anomaly Categories\n"
        "NASA ASRS 2018-2026",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_figure(fig, save_dir, "layer1_spc_cusum.png")


def plot_equipment_spc(
    result: dict,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot the Equipment Critical SPC chart used in the Layer 1 story."""
    monthly_counts = result["monthly_counts"]
    control_limit = result["control_limit"]

    fig, (ax_counts, ax_cusum) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        "Aircraft Equipment Problem Critical - Monthly Rate + CUSUM\n"
        "Post-COVID return-to-service deferred maintenance spike (May 2022)",
        fontsize=13,
        fontweight="bold",
    )

    _plot_series_with_fill(
        ax_counts,
        monthly_counts.index,
        monthly_counts.values,
        color="#003366",
        label="Monthly incidents",
        alpha_fill=0.12,
    )
    ax_counts.plot(
        result["trend"].index,
        result["trend"].values,
        color="navy",
        lw=2.5,
        label="STL trend",
    )
    ax_counts.axvspan(
        pd.Timestamp("2020-03-01"),
        pd.Timestamp("2021-12-01"),
        alpha=0.08,
        color="grey",
        label="COVID operations (2020-2021)",
    )
    ax_counts.axvspan(
        pd.Timestamp("2022-05-01"),
        pd.Timestamp("2022-12-01"),
        alpha=0.15,
        color="red",
        label="SPC alarm - maintenance spike",
    )
    _plot_vertical_markers(
        ax_counts,
        result["alarms"],
        color="red",
        alpha=0.4,
        linewidth=1,
    )
    ax_counts.set_ylabel("Incidents per month", fontsize=11)
    ax_counts.set_title("Monthly counts and STL trend", fontsize=10)
    ax_counts.legend(fontsize=9)

    _plot_cusum_panel(
        ax_cusum,
        monthly_counts.index,
        result["s_pos"],
        control_limit,
        alarms=result["alarms"],
        mark_first_alarm=True,
    )
    ax_cusum.set_xlabel("Date", fontsize=11)

    plt.tight_layout()
    _save_figure(fig, save_dir, "equipment_critical_spc.png")


def plot_topic_landscape(
    topic_model,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot top BERTopic clusters by document count."""
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].head(20).copy()
    info["short_name"] = info["Topic"].apply(
        lambda topic_id: _topic_keywords(topic_model, topic_id, n_words=5)
    )
    info = info.sort_values("Count")

    fig, ax = plt.subplots(figsize=(12, 8))
    _barh_with_labels(
        ax,
        labels=info["short_name"].tolist(),
        values=info["Count"].to_numpy(),
        color="steelblue",
        xlabel="Incident count",
        title="Layer 2: BERTopic - Top 20 Semantic Topics\nNASA ASRS 2018-2026",
        title_size=13,
    )

    plt.tight_layout()
    _save_figure(fig, save_dir, "layer2_topic_landscape.png")


def plot_gnss_timeline(
    topics_over_time_df: pd.DataFrame,
    gnss_topic_ids: list[int],
    all_gnss_monthly: pd.Series,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot GNSS emergence from BERTopic topic frequency and regex monthly counts."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    ax_top = axes[0]
    if topics_over_time_df is not None and gnss_topic_ids:
        gnss_tot = topics_over_time_df[
            topics_over_time_df["Topic"].isin(gnss_topic_ids)
        ].copy()
        gnss_tot["Timestamp"] = pd.to_datetime(gnss_tot["Timestamp"])
        grouped = (
            gnss_tot.groupby("Timestamp")["Frequency"]
            .sum()
            .reset_index()
            .sort_values("Timestamp")
        )
        _plot_series_with_fill(
            ax_top,
            grouped["Timestamp"],
            grouped["Frequency"],
            color="royalblue",
            label="GNSS topic frequency",
            alpha_fill=0.3,
        )
        ax_top.axvline(
            pd.Timestamp("2023-10-01"),
            color="red",
            ls="--",
            lw=1.5,
            label="Late-2023 inflection",
        )
        ax_top.legend(fontsize=9)
    else:
        ax_top.text(
            0.5,
            0.5,
            "No GNSS topics identified in model",
            ha="center",
            va="center",
            transform=ax_top.transAxes,
        )

    ax_top.set_title(
        "GNSS / Spoofing topic frequency - BERTopic semantic cluster",
        fontsize=11,
        fontweight="bold",
    )
    ax_top.set_ylabel("Topic frequency (BERTopic)")

    ax_bottom = axes[1]
    if all_gnss_monthly is not None and not all_gnss_monthly.empty:
        timeline_index = _as_timeline_index(all_gnss_monthly)
        _plot_series_with_fill(
            ax_bottom,
            timeline_index,
            all_gnss_monthly.values,
            color="darkorange",
            label="spoofing|jamming mentions",
            alpha_fill=0.3,
        )
        rolling_mean = all_gnss_monthly.rolling(3, center=True).mean()
        ax_bottom.plot(
            timeline_index,
            rolling_mean.values,
            color="red",
            lw=2.5,
            label="3-month rolling mean",
        )
        ax_bottom.axvline(
            pd.Timestamp("2023-10-01"),
            color="red",
            ls="--",
            lw=1.5,
            label="Late-2023 inflection",
        )
        ax_bottom.legend(fontsize=9)

    ax_bottom.set_title(
        "GNSS spoofing/jamming narrative mentions - regex signal (Layer 1)",
        fontsize=11,
        fontweight="bold",
    )
    ax_bottom.set_ylabel("Monthly incident count")
    ax_bottom.set_xlabel("Date")

    fig.suptitle(
        "GNSS Spoofing Emergence: Near-Zero (2018-2022) "
        "to Sustained Signal (2023-2025)\n"
        "NASA ASRS | Layer 2 semantic validation of Layer 1 SPC signal",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    _save_figure(fig, save_dir, "layer2_gnss_emergence.png")


def plot_red_quadrant_topics(
    asrs: pd.DataFrame,
    topic_model,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot top BERTopic topics within RED quadrant incidents."""
    if "quadrant" not in asrs.columns or "topic_id" not in asrs.columns:
        logger.warning("No RED quadrant topic data to plot")
        return

    red = asrs.loc[asrs["quadrant"] == "RED"].copy()
    if red.empty:
        logger.warning("No RED quadrant incidents to plot")
        return

    counts = red["topic_id"].value_counts()
    counts = counts[counts.index != -1].head(15)

    labels = [_topic_keywords(topic_model, topic_id) for topic_id in counts.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    _barh_with_labels(
        ax,
        labels=labels[::-1],
        values=counts.astype(int).to_numpy()[::-1],
        color="#cc0000",
        alpha=0.75,
        xlabel="Incident count",
        title=(
            "Layer 2: Topic Breakdown - RED Quadrant Incidents\n"
            "(Novel pattern + SPC anomalous frequency)"
        ),
        title_size=12,
    )

    plt.tight_layout()
    _save_figure(fig, save_dir, "layer2_red_topics.png")


def plot_topic_heatmap(
    topics_over_time_df: pd.DataFrame,
    topic_model,
    top_n: int = 15,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot topic frequency evolution heatmap for top_n topics."""
    if topics_over_time_df is None or topics_over_time_df.empty:
        logger.warning("No topics_over_time data to plot")
        return

    info = topic_model.get_topic_info()
    top_ids = info[info["Topic"] != -1].head(top_n)["Topic"].tolist()

    topics_over_time = topics_over_time_df[
        topics_over_time_df["Topic"].isin(top_ids)
    ].copy()
    topics_over_time["Timestamp"] = pd.to_datetime(topics_over_time["Timestamp"])
    topics_over_time["YearQ"] = topics_over_time["Timestamp"].dt.to_period("Q").astype(str)

    pivot = topics_over_time.pivot_table(
        index="Topic",
        columns="YearQ",
        values="Frequency",
        aggfunc="sum",
    ).fillna(0)

    row_labels = [
        _topic_keywords(topic_model, topic_id, n_words=4)
        for topic_id in pivot.index
    ]
    pivot.index = row_labels

    pivot_norm = pivot.div(pivot.max(axis=1) + 1e-9, axis=0)

    fig, ax = plt.subplots(figsize=(16, 8))
    image = ax.imshow(
        pivot_norm.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(image, ax=ax, label="Normalised frequency (per topic)")
    ax.set_title(
        "Layer 2: Topic Evolution Heatmap - Top 15 Topics by Quarter\n"
        "NASA ASRS 2018-2026",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    _save_figure(fig, save_dir, "layer2_topic_heatmap.png")


def plot_risk_distribution(
    asrs: pd.DataFrame,
    save_dir: str | Path = "outputs/figures",
) -> None:
    """Plot risk score histogram and component breakdown for top incidents."""
    flagged = asrs[asrs["quadrant"].isin(["RED", "ORANGE"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Layer 3 - Rule-Based Precursor Risk Scores\n"
        f"RED and ORANGE quadrant incidents ({len(flagged):,} incidents)",
        fontsize=13,
        fontweight="bold",
    )

    threshold = flagged["precursor_score"].quantile(0.90)
    axes[0].hist(
        flagged["precursor_score"],
        bins=40,
        color="#003366",
        alpha=0.8,
        edgecolor="white",
    )
    axes[0].axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"90th pct = {threshold:.3f} (high-risk threshold)",
    )
    axes[0].set_xlabel("Precursor risk score (0-1)", fontsize=11)
    axes[0].set_ylabel("Number of incidents", fontsize=11)
    axes[0].set_title("Overall risk score distribution")
    axes[0].legend(fontsize=9)

    top50 = flagged.nlargest(50, "precursor_score")
    component_cols = [
        column for column in asrs.columns
        if column.startswith("component_")
    ]
    component_means = top50[component_cols].mean()
    component_labels = [
        column.replace("component_", "").replace("_", "\n")
        for column in component_cols
    ]

    colors = ["#cc0000", "#cc6600", "#003366", "#006600", "#660066"]
    axes[1].barh(
        component_labels,
        component_means.values,
        color=colors[:len(component_cols)],
        alpha=0.8,
    )
    axes[1].set_xlabel("Mean term count (top 50 high-risk incidents)", fontsize=11)
    axes[1].set_title("Which factors drive high-risk scores?")

    plt.tight_layout()
    _save_figure(fig, save_dir, "precursor_risk_distribution.png")