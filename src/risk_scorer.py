import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PrecursorComponent:
    """Human-factor precursor component used in the rule-based scorer."""

    weight: float
    terms: tuple[str, ...]


PRECURSOR_COMPONENTS = {
    "fatigue": PrecursorComponent(
        weight=2.5,
        terms=(
            "fatigue",
            "fatigued",
            "tired",
            "exhausted",
            "rest period",
            "duty time",
            "not rested",
            "long day",
            "sleep",
            "overworked",
        ),
    ),
    "comm_breakdown": PrecursorComponent(
        weight=2.0,
        terms=(
            "miscommunication",
            "misunderstood",
            "wrong frequency",
            "unclear",
            "confused",
            "readback",
            "communication",
            "misheard",
            "not received",
            "no response",
        ),
    ),
    "near_miss": PrecursorComponent(
        weight=2.5,
        terms=(
            "nearly",
            "almost",
            "narrowly",
            "close call",
            "could have",
            "nmac",
            "near miss",
            "inches",
            "feet away",
            "last moment",
            "just missed",
        ),
    ),
    "procedure_deviation": PrecursorComponent(
        weight=1.5,
        terms=(
            "skipped",
            "omitted",
            "forgot",
            "failed to",
            "did not check",
            "missed",
            "overlooked",
            "non-standard",
            "deviation",
            "violation",
        ),
    ),
    "urgency": PrecursorComponent(
        weight=1.5,
        terms=(
            "emergency",
            "immediately",
            "critical",
            "serious",
            "severe",
            "dangerous",
            "unsafe",
            "mayday",
            "pan pan",
            "declare emergency",
        ),
    ),
}

COMPONENT_WEIGHTS = {
    component: config.weight
    for component, config in PRECURSOR_COMPONENTS.items()
}

PRECURSOR_TERMS = {
    component: list(config.terms)
    for component, config in PRECURSOR_COMPONENTS.items()
}

MAX_RAW_SCORE = sum(2 * weight for weight in COMPONENT_WEIGHTS.values())

NEGATION_PATTERNS = {
    "fatigue": (
        r"\bnot\s+(?:fatigued|tired|exhausted)\b",
        r"\bno\s+(?:fatigue|tiredness)\b",
        r"\bwas\s+not\s+(?:fatigued|tired|exhausted)\b",
        r"\bwere\s+not\s+(?:fatigued|tired|exhausted)\b",
    ),
    "urgency": (
        r"\bno\s+emergency\b",
        r"\bnot\s+an\s+emergency\b",
        r"\bdid\s+not\s+declare\s+emergency\b",
    ),
}


def _term_pattern(term: str) -> re.Pattern[str]:
    """
    Compile a case-insensitive term pattern.

    Word-ish phrases get word boundaries so `tired` does not match inside a
    longer token. Phrases with spaces are matched flexibly across whitespace.
    """
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


COMPILED_TERMS = {
    component: {
        term: _term_pattern(term)
        for term in config.terms
    }
    for component, config in PRECURSOR_COMPONENTS.items()
}

COMPILED_NEGATIONS = {
    component: tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns)
    for component, patterns in NEGATION_PATTERNS.items()
}


def _negated_components(text: str) -> set[str]:
    """Return components whose obvious negation patterns appear in text."""
    negated = set()
    for component, patterns in COMPILED_NEGATIONS.items():
        if any(pattern.search(text) for pattern in patterns):
            negated.add(component)
    return negated


def _is_single_word(term: str) -> bool:
    """Return True when term should use word-boundary regex matching."""
    return " " not in term.strip()


SINGLE_WORD_PATTERNS = {
    component: {
        term: re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)
        for term in config.terms
        if _is_single_word(term)
    }
    for component, config in PRECURSOR_COMPONENTS.items()
}

PHRASE_TERMS = {
    component: tuple(
        term.lower()
        for term in config.terms
        if not _is_single_word(term)
    )
    for component, config in PRECURSOR_COMPONENTS.items()
}


def _matched_terms(text: str) -> dict[str, list[str]]:
    """Return matched precursor terms by component."""
    negated = _negated_components(text)
    matches: dict[str, list[str]] = {}

    for component in PRECURSOR_COMPONENTS:
        if component in negated:
            matches[component] = []
            continue

        component_matches = []

        for term, pattern in SINGLE_WORD_PATTERNS[component].items():
            if pattern.search(text):
                component_matches.append(term)

        for term in PHRASE_TERMS[component]:
            if term in text:
                component_matches.append(term)

        matches[component] = component_matches

    return matches


def score_incident(narrative: Any) -> dict[str, Any]:
    """
    Compute transparent rule-based precursor risk score.

    Returns:
        - precursor_score
        - per-component term counts
        - per-component matched terms for auditability
    """
    text = str(narrative).lower()
    matches = _matched_terms(text)

    component_counts = {
        component: len(terms)
        for component, terms in matches.items()
    }

    raw_score = sum(
        min(component_counts[component], 2) * COMPONENT_WEIGHTS[component]
        for component in COMPONENT_WEIGHTS
    )
    score = min(raw_score / MAX_RAW_SCORE, 1.0)

    result: dict[str, Any] = {
        "precursor_score": round(score, 3),
    }

    for component, count in component_counts.items():
        result[f"component_{component}"] = count
        result[f"matched_{component}"] = "; ".join(matches[component])

    return result


def apply_risk_scorer(asrs: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based scorer to all incidents, add high_precursor_risk flag."""
    if "full_narrative" not in asrs.columns:
        raise ValueError("Missing required column: full_narrative")

    logger.info("Scoring %s incidents...", f"{len(asrs):,}")

    scores = asrs["full_narrative"].apply(score_incident)
    score_df = pd.DataFrame(scores.tolist())

    asrs = asrs.copy()
    for column in score_df.columns:
        asrs[column] = score_df[column].values

    threshold = asrs["precursor_score"].quantile(0.90)
    asrs["high_precursor_risk"] = (
        asrs["precursor_score"] >= threshold
    ).astype(int)

    logger.info("Scoring complete. 90th-pct threshold: %.3f", threshold)
    logger.info("High-risk incidents: %s", f"{asrs['high_precursor_risk'].sum():,}")

    return asrs


def export_high_risk_incidents(
    asrs: pd.DataFrame,
    out_path: str | Path = "outputs/data/layer3_high_risk_incidents.csv",
) -> pd.DataFrame:
    """Export top 100 RED/ORANGE incidents ranked by precursor_score."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "ACN",
        "date",
        "quadrant",
        "precursor_score",
        "if_score",
        "spc_flag",
        "Events | Anomaly",
    ]
    component_cols = [
        column for column in asrs.columns
        if column.startswith("component_")
    ]
    matched_cols = [
        column for column in asrs.columns
        if column.startswith("matched_")
    ]
    optional_cols = [
        "Aircraft 1 | Flight Phase",
        "topic_label",
        "full_narrative",
    ]

    keep = [
        column
        for column in base_cols + component_cols + matched_cols + optional_cols
        if column in asrs.columns
    ]

    top100 = (
        asrs[asrs["quadrant"].isin(["RED", "ORANGE"])]
        .nlargest(100, "precursor_score")[keep]
        .copy()
    )

    if "full_narrative" in top100.columns:
        top100["narrative_preview"] = top100["full_narrative"].str[:300]
        top100 = top100.drop(columns=["full_narrative"])

    top100.to_csv(out_path, index=False)
    logger.info("Exported %d high-risk incidents to %s", len(top100), out_path)

    return top100