"""
Layer 3: Rule-based precursor risk scorer.

Transparent, fully auditable — every score maps to a known human factors
category. Deliberately not a trained ML model. See CLAUDE.md for the
defence rationale.
"""
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from src.logger import get_logger

logger = get_logger(__name__)


PRECURSOR_TERMS = {
    'fatigue': [
        'fatigue', 'tired', 'exhausted', 'rest period',
        'duty time', 'not rested', 'long day', 'sleep',
        'fatigued', 'overworked',
    ],
    'comm_breakdown': [
        'miscommunication', 'misunderstood', 'wrong frequency',
        'unclear', 'confused', 'readback', 'communication',
        'misheard', 'not received', 'no response',
    ],
    'near_miss': [
        'nearly', 'almost', 'narrowly', 'close call',
        'could have', 'nmac', 'near miss', 'inches',
        'feet away', 'last moment', 'just missed',
    ],
    'procedure_deviation': [
        'skipped', 'omitted', 'forgot', 'failed to',
        'did not check', 'missed', 'overlooked',
        'non-standard', 'deviation', 'violation',
    ],
    'urgency': [
        'emergency', 'immediately', 'critical', 'serious',
        'severe', 'dangerous', 'unsafe', 'mayday',
        'pan pan', 'declare emergency',
    ],
}

COMPONENT_WEIGHTS = {
    'fatigue': 2.5,
    'near_miss': 2.5,
    'comm_breakdown': 2.0,
    'procedure_deviation': 1.5,
    'urgency': 1.5,
}

# Max possible raw score: each component capped at 2 hits
MAX_RAW_SCORE = sum(2 * w for w in COMPONENT_WEIGHTS.values())


def score_incident(narrative: str) -> Dict:
    """
    Compute transparent rule-based precursor risk score.
    Returns overall score (0-1) and per-component term counts.
    Every component is independently auditable.
    """
    text = str(narrative).lower()

    components = {
        cat: sum(1 for term in terms if term in text)
        for cat, terms in PRECURSOR_TERMS.items()
    }

    raw = sum(
        min(components[cat], 2) * COMPONENT_WEIGHTS[cat]
        for cat in COMPONENT_WEIGHTS
    )

    score = min(raw / MAX_RAW_SCORE, 1.0)

    return {
        'precursor_score': round(score, 3),
        **{f'component_{k}': v for k, v in components.items()},
    }


def apply_risk_scorer(asrs: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based scorer to all incidents, add high_precursor_risk flag."""
    logger.info("Scoring %s incidents...", f"{len(asrs):,}")
    scores = asrs['full_narrative'].apply(score_incident)
    score_df = pd.DataFrame(scores.tolist())

    asrs = asrs.copy()
    for col in score_df.columns:
        asrs[col] = score_df[col].values

    threshold = asrs['precursor_score'].quantile(0.90)
    asrs['high_precursor_risk'] = (asrs['precursor_score'] >= threshold).astype(int)

    logger.info("Scoring complete. 90th-pct threshold: %.3f", threshold)
    logger.info("High-risk incidents: %s", f"{asrs['high_precursor_risk'].sum():,}")
    return asrs


def plot_risk_distribution(asrs: pd.DataFrame, save_dir: str = 'outputs/figures') -> None:
    """Two-panel chart: score histogram + component breakdown for top 50."""
    os.makedirs(save_dir, exist_ok=True)

    flagged = asrs[asrs['quadrant'].isin(['RED', 'ORANGE'])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Layer 3 — Rule-Based Precursor Risk Scores\n'
        f'RED and ORANGE quadrant incidents  ({len(flagged):,} incidents)',
        fontsize=13, fontweight='bold',
    )

    # Left: overall distribution
    threshold = flagged['precursor_score'].quantile(0.90)
    axes[0].hist(flagged['precursor_score'], bins=40,
                 color='#003366', alpha=0.8, edgecolor='white')
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'90th pct = {threshold:.3f} (high-risk threshold)')
    axes[0].set_xlabel('Precursor risk score (0-1)', fontsize=11)
    axes[0].set_ylabel('Number of incidents', fontsize=11)
    axes[0].set_title('Overall risk score distribution')
    axes[0].legend(fontsize=9)

    # Right: mean component counts for top 50 highest-risk incidents
    top50 = flagged.nlargest(50, 'precursor_score')
    component_cols = [c for c in asrs.columns if c.startswith('component_')]
    component_means = top50[component_cols].mean()
    component_labels = [c.replace('component_', '').replace('_', '\n')
                        for c in component_cols]

    colors_list = ['#cc0000', '#cc6600', '#003366', '#006600', '#660066']
    axes[1].barh(component_labels, component_means.values,
                 color=colors_list[:len(component_cols)], alpha=0.8)
    axes[1].set_xlabel('Mean term count (top 50 high-risk incidents)', fontsize=11)
    axes[1].set_title('Which factors drive high-risk scores?')

    plt.tight_layout()
    out = os.path.join(save_dir, 'precursor_risk_distribution.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    logger.info("Saved risk distribution chart to %s", out)
    plt.close()


def export_high_risk_incidents(asrs: pd.DataFrame,
                                out_path: str = 'outputs/data/layer3_high_risk_incidents.csv'
                                ) -> pd.DataFrame:
    """Export top 100 RED/ORANGE incidents ranked by precursor_score."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Build column list defensively — not all columns guaranteed present
    base_cols = ['ACN', 'date', 'quadrant', 'precursor_score',
                 'if_score', 'spc_flag', 'Events | Anomaly']
    component_cols = [c for c in asrs.columns if c.startswith('component_')]
    optional_cols = ['Aircraft 1 | Flight Phase', 'full_narrative', 'topic_label']
    extra = [c for c in optional_cols if c in asrs.columns]
    keep = [c for c in base_cols + component_cols + extra if c in asrs.columns]

    top100 = (
        asrs[asrs['quadrant'].isin(['RED', 'ORANGE'])]
        .nlargest(100, 'precursor_score')[keep]
        .copy()
    )

    if 'full_narrative' in top100.columns:
        top100['narrative_preview'] = top100['full_narrative'].str[:300]
        top100 = top100.drop(columns=['full_narrative'])

    top100.to_csv(out_path, index=False)
    logger.info("Exported %d high-risk incidents to %s", len(top100), out_path)
    return top100
