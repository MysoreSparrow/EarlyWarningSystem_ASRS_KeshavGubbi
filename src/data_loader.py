from pathlib import Path

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

ASRS_FILE_SUFFIXES = {".xls", ".xlsx"}
DEFAULT_MERGED_OUTPUT_PATH = Path("outputs/data/asrs_merged.parquet")


def load_asrs_file(filepath: str | Path) -> pd.DataFrame:
    """Load one ASRS XLS export, which is actually TSV with two header rows."""
    filepath = Path(filepath)

    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        low_memory=False,
        header=[0, 1],
        on_bad_lines="skip",
    )

    df.columns = [
        f"{str(group).strip()} | {str(field).strip()}"
        if str(group).strip() and not str(group).startswith("Unnamed")
        else str(field).strip()
        for group, field in df.columns
    ]
    return df


def load_and_merge_asrs(
    data_dir: str | Path = "data/raw",
    save_path: str | Path | None = DEFAULT_MERGED_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Load all ASRS batch files and merge into one clean DataFrame.

    Parameters
    ----------
    data_dir:
        Directory containing ASRS `.xls`/`.xlsx` batch exports. These files are
        TSV text exports despite the Excel-like extension.
    save_path:
        Optional parquet output path. Pass None to skip writing the merged file.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"ASRS data directory not found: {data_dir}")

    files = sorted(
        path for path in data_dir.iterdir()
        if path.suffix.lower() in ASRS_FILE_SUFFIXES
    )

    if not files:
        raise FileNotFoundError(
            f"No ASRS files found in {data_dir}. "
            "Download from https://asrs.arc.nasa.gov/search/database.html"
        )

    frames = []
    for path in files:
        df = load_asrs_file(path)
        logger.info(
            "Loaded %s: %s records, %d columns",
            path.name,
            f"{len(df):,}",
            len(df.columns),
        )
        frames.append(df)

    asrs = pd.concat(frames, ignore_index=True)

    date_col = "Time | Date"
    if date_col in asrs.columns:
        asrs["date"] = pd.to_datetime(
            asrs[date_col].astype(str).str[:6],
            format="%Y%m",
            errors="coerce",
        )
    else:
        date_candidates = [column for column in asrs.columns if "date" in column.lower()]
        if not date_candidates:
            raise ValueError(
                f"'{date_col}' not found and no fallback date columns detected."
            )

        fallback_col = date_candidates[0]
        logger.warning("'%s' not found, using '%s'", date_col, fallback_col)
        asrs["date"] = pd.to_datetime(
            asrs[fallback_col].astype(str).str[:6],
            format="%Y%m",
            errors="coerce",
        )

    narrative_1_col = "Report 1 | Narrative"
    narrative_2_col = "Report 2 | Narrative"
    empty_text = pd.Series("", index=asrs.index)

    asrs["full_narrative"] = (
        asrs.get(narrative_1_col, empty_text).fillna("").astype(str)
        + " "
        + asrs.get(narrative_2_col, empty_text).fillna("").astype(str)
    ).str.strip()

    asrs["narrative_word_count"] = asrs["full_narrative"].str.split().str.len()

    asrs = asrs.sort_values("date").reset_index(drop=True)

    logger.info("Merged: %s total records", f"{len(asrs):,}")
    logger.info("Date range: %s to %s", asrs["date"].min(), asrs["date"].max())

    year_counts = asrs["date"].dt.year.value_counts().sort_index()
    logger.info("Records per year:\n%s", year_counts.to_string())

    logger.info(
        "Narrative word count - median: %.0f, pct>100: %.1f%%",
        asrs["narrative_word_count"].median(),
        (asrs["narrative_word_count"] > 100).mean() * 100,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_df = asrs.copy()
        for column in save_df.select_dtypes(include=["object"], exclude=["str"]).columns:
            save_df[column] = save_df[column].astype(str)

        save_df.to_parquet(save_path, index=False)
        logger.info("Saved merged parquet to %s", save_path)

    return asrs