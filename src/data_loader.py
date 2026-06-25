import pandas as pd
import numpy as np
import os
from src.logger import get_logger

logger = get_logger(__name__)


def load_asrs_file(filepath: str) -> pd.DataFrame:
    """
    Load one ASRS XLS export (actually TSV with two header rows).
    """
    df = pd.read_csv(
        filepath,
        sep='\t',
        encoding='utf-8',
        low_memory=False,
        header=[0, 1],
        on_bad_lines='skip'
    )
    # Flatten two-level column names — "Group | Field" format
    df.columns = [
        f"{str(a).strip()} | {str(b).strip()}"
        if str(a).strip() and not str(a).startswith('Unnamed')
        else str(b).strip()
        for a, b in df.columns
    ]
    return df


def load_and_merge_asrs(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load all ASRS batch files and merge into one clean DataFrame."""
    files = [f for f in os.listdir(data_dir)
             if f.endswith('.xls') or f.endswith('.xlsx')]

    if not files:
        raise FileNotFoundError(
            f"No ASRS files found in {data_dir}. "
            "Download from https://asrs.arc.nasa.gov/search/database.html"
        )

    dfs = []
    for fname in sorted(files):
        path = os.path.join(data_dir, fname)
        df = load_asrs_file(path)
        logger.info("Loaded %s: %s records, %d columns", fname, f"{len(df):,}", len(df.columns))
        dfs.append(df)

    asrs = pd.concat(dfs, ignore_index=True)

    # Parse date (format: YYYYMM as integer e.g. 202301)
    date_col = 'Time | Date'
    if date_col in asrs.columns:
        asrs['date'] = pd.to_datetime(
            asrs[date_col].astype(str).str[:6],
            format='%Y%m',
            errors='coerce'
        )
    else:
        # Fallback: look for any column with 'Date' in name
        date_candidates = [c for c in asrs.columns if 'date' in c.lower()]
        if date_candidates:
            logger.warning("'%s' not found, using '%s'", date_col, date_candidates[0])
            asrs['date'] = pd.to_datetime(
                asrs[date_candidates[0]].astype(str).str[:6],
                format='%Y%m',
                errors='coerce'
            )

    # Combine both reporter narratives
    narr1 = 'Report 1 | Narrative'
    narr2 = 'Report 2 | Narrative'
    asrs['full_narrative'] = (
        asrs.get(narr1, pd.Series([''] * len(asrs))).fillna('') + ' ' +
        asrs.get(narr2, pd.Series([''] * len(asrs))).fillna('')
    ).str.strip()

    asrs['narrative_word_count'] = asrs['full_narrative'].str.split().str.len()

    # Sort by date
    if 'date' in asrs.columns:
        asrs = asrs.sort_values('date').reset_index(drop=True)

    logger.info("Merged: %s total records", f"{len(asrs):,}")
    if 'date' in asrs.columns:
        logger.info("Date range: %s to %s", asrs['date'].min(), asrs['date'].max())
        year_counts = asrs['date'].dt.year.value_counts().sort_index()
        logger.info("Records per year:\n%s", year_counts.to_string())

    logger.info(
        "Narrative word count — median: %.0f, pct>100: %.1f%%",
        asrs['narrative_word_count'].median(),
        (asrs['narrative_word_count'] > 100).mean() * 100,
    )

    # Save as parquet for fast reloading
    # Cast all object columns to str to prevent pyarrow mixed-type inference errors
    out_path = "outputs/data/asrs_merged.parquet"
    os.makedirs("outputs/data", exist_ok=True)
    save_df = asrs.copy()
    for col in save_df.select_dtypes(include=['object']).columns:
        save_df[col] = save_df[col].astype(str)
    save_df.to_parquet(out_path, index=False)
    logger.info("Saved merged parquet to %s", out_path)

    return asrs
