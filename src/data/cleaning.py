"""Data cleaning and threshold filtering module for MovieLens pipeline (US1.3 - Task 2).

Implements:
- Reading `data.min_ratings` threshold from `params.yaml` (separate from `serving.min_ratings`).
- Filtering out users and movies with fewer ratings than the threshold (AC3).
- End-to-end integration with `validate_raw_data` to ensure checks run before filtering (AC1, AC2).
- Saving clean data to processed directory.
- Saving an execution report recording check results and ratings removed (AC4).
- Optional DVC tracking integration for the clean dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import yaml

from src.data.validation import (
    DataValidationError,
    _resolve_col,
    validate_raw_data,
)

DEFAULT_PARAMS_PATH = "params.yaml"


class DataConfigError(Exception):
    """Raised when params.yaml is missing or contains invalid data configuration."""


@dataclass(frozen=True)
class DataSettings:
    """Configuration for data validation and cleaning."""

    min_ratings: int = 10
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    report_path: str = "data/processed/cleaning_report.json"


def load_data_settings(path: str | Path | None = None) -> DataSettings:
    """Read data configuration from params.yaml (separate from serving).

    Args:
        path: Optional path to params.yaml. Defaults to PARAMS_PATH env var or 'params.yaml'.

    Returns:
        DataSettings dataclass instance.

    Raises:
        DataConfigError: If params.yaml is missing or data.min_ratings is invalid.
    """
    path = Path(path or os.getenv("PARAMS_PATH", DEFAULT_PARAMS_PATH))
    if not path.exists():
        raise DataConfigError(
            f"Configuration file not found: {path}. Run from the repository root."
        )

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_section = config.get("data")
    if not data_section or not isinstance(data_section, dict):
        raise DataConfigError(f"Missing 'data' section in {path}.")

    raw_min = os.getenv("DATA_MIN_RATINGS", data_section.get("min_ratings"))
    if raw_min is None:
        raise DataConfigError(f"Missing 'data.min_ratings' in {path}.")

    try:
        min_ratings = int(raw_min)
    except (TypeError, ValueError) as exc:
        raise DataConfigError(
            f"'data.min_ratings' must be an integer, got {raw_min!r}."
        ) from exc

    if min_ratings < 1:
        raise DataConfigError(
            f"'data.min_ratings' must be 1 or greater, got {min_ratings}."
        )

    raw_dir = str(data_section.get("raw_dir", "data/raw"))
    processed_dir = str(data_section.get("processed_dir", "data/processed"))
    report_path = str(data_section.get("report_path", f"{processed_dir}/cleaning_report.json"))

    return DataSettings(
        min_ratings=min_ratings,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        report_path=report_path,
    )


def filter_ratings_by_threshold(
    ratings_df: pd.DataFrame,
    min_ratings: int = 10,
    user_col: Optional[str] = None,
    movie_col: Optional[str] = None,
    iterative: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove users and movies below the minimum rating threshold (AC3).

    Args:
        ratings_df: Input ratings DataFrame.
        min_ratings: Minimum number of ratings required per user and per movie.
        user_col: Optional override for user ID column name.
        movie_col: Optional override for movie ID column name.
        iterative: Whether to iteratively filter until both user and movie counts
            converge (k-core filtering). Defaults to True.

    Returns:
        Tuple of (clean_ratings_df, stats_dict).
    """
    if ratings_df is None or ratings_df.empty:
        raise ValueError("Cannot filter an empty ratings DataFrame.")

    u_col = user_col or _resolve_col(ratings_df, ["userId", "user_id", "UserId", "user"])
    m_col = movie_col or _resolve_col(ratings_df, ["movieId", "movie_id", "MovieId", "movie"])

    initial_ratings = len(ratings_df)
    initial_users = int(ratings_df[u_col].nunique())
    initial_movies = int(ratings_df[m_col].nunique())

    if min_ratings <= 1:
        return ratings_df.copy(), {
            "initial_ratings": initial_ratings,
            "clean_ratings": initial_ratings,
            "ratings_removed": 0,
            "pct_ratings_removed": 0.0,
            "initial_users": initial_users,
            "clean_users": initial_users,
            "users_removed": 0,
            "pct_users_removed": 0.0,
            "initial_movies": initial_movies,
            "clean_movies": initial_movies,
            "movies_removed": 0,
            "pct_movies_removed": 0.0,
        }

    df = ratings_df.copy()

    if iterative:
        prev_len = -1
        while len(df) != prev_len:
            prev_len = len(df)
            # Filter movies with >= min_ratings
            movie_counts = df[m_col].value_counts()
            qual_movies = movie_counts[movie_counts >= min_ratings].index
            df = df[df[m_col].isin(qual_movies)]

            # Filter users with >= min_ratings
            user_counts = df[u_col].value_counts()
            qual_users = user_counts[user_counts >= min_ratings].index
            df = df[df[u_col].isin(qual_users)]
    else:
        # Single-pass filter
        movie_counts = df[m_col].value_counts()
        qual_movies = movie_counts[movie_counts >= min_ratings].index
        user_counts = df[u_col].value_counts()
        qual_users = user_counts[user_counts >= min_ratings].index
        df = df[df[m_col].isin(qual_movies) & df[u_col].isin(qual_users)]

    clean_ratings = len(df)
    clean_users = int(df[u_col].nunique()) if clean_ratings > 0 else 0
    clean_movies = int(df[m_col].nunique()) if clean_ratings > 0 else 0

    ratings_removed = initial_ratings - clean_ratings
    users_removed = initial_users - clean_users
    movies_removed = initial_movies - clean_movies

    stats = {
        "initial_ratings": initial_ratings,
        "clean_ratings": clean_ratings,
        "ratings_removed": ratings_removed,
        "pct_ratings_removed": round((ratings_removed / initial_ratings) * 100, 2)
        if initial_ratings > 0
        else 0.0,
        "initial_users": initial_users,
        "clean_users": clean_users,
        "users_removed": users_removed,
        "pct_users_removed": round((users_removed / initial_users) * 100, 2)
        if initial_users > 0
        else 0.0,
        "initial_movies": initial_movies,
        "clean_movies": clean_movies,
        "movies_removed": movies_removed,
        "pct_movies_removed": round((movies_removed / initial_movies) * 100, 2)
        if initial_movies > 0
        else 0.0,
    }

    return df.reset_index(drop=True), stats


def standardize_dtypes_and_columns(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize column names and types following EDA Section 10 conclusions.

    - Renames identifiers to snake_case: user_id, movie_id.
    - Casts user_id and movie_id to int32 (halving RAM compared to int64).
    - Casts rating to float32.
    - Casts timestamp to int64 if present.
    """
    r_df = ratings_df.copy()
    m_df = movies_df.copy()

    # Standardize ratings columns
    u_col = _resolve_col(r_df, ["userId", "user_id", "UserId", "user"])
    m_col = _resolve_col(r_df, ["movieId", "movie_id", "MovieId", "movie"])
    rate_col = _resolve_col(r_df, ["rating", "Rating", "score"])

    rename_r = {u_col: "user_id", m_col: "movie_id", rate_col: "rating"}
    r_df = r_df.rename(columns=rename_r)

    r_df["user_id"] = r_df["user_id"].astype("int32")
    r_df["movie_id"] = r_df["movie_id"].astype("int32")
    r_df["rating"] = r_df["rating"].astype("float32")
    if "timestamp" in r_df.columns:
        r_df["timestamp"] = r_df["timestamp"].astype("int64")

    # Standardize movies columns
    m_col_m = _resolve_col(m_df, ["movieId", "movie_id", "MovieId", "movie", "id"])
    m_df = m_df.rename(columns={m_col_m: "movie_id"})
    m_df["movie_id"] = m_df["movie_id"].astype("int32")

    return r_df, m_df


def filter_movies_catalog(
    movies_df: pd.DataFrame,
    clean_ratings_df: pd.DataFrame,
    movie_col: Optional[str] = None,
) -> pd.DataFrame:
    """Filter movies catalog to keep only movies retained in clean ratings."""
    m_col_ratings = movie_col or _resolve_col(
        clean_ratings_df, ["movieId", "movie_id", "MovieId", "movie"]
    )
    m_col_movies = _resolve_col(movies_df, ["movieId", "movie_id", "MovieId", "movie", "id"])

    retained_movie_ids = set(clean_ratings_df[m_col_ratings].unique())
    clean_movies = movies_df[movies_df[m_col_movies].isin(retained_movie_ids)].copy()
    return clean_movies.reset_index(drop=True)


def clean_and_save_data(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    min_ratings: Optional[int] = None,
    params_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute validation and threshold cleaning, then save clean data and report.

    1. AC1 & AC2: Runs `validate_raw_data`. Halts with DataValidationError if data is invalid.
    2. AC3: Applies threshold `data.min_ratings` from `params.yaml` (default k=10 from EDA).
    3. EDA Recommendations: Standardizes column names (user_id, movie_id) and optimizes dtypes (int32, float32).
    4. Saves clean data as Parquet (primary) and CSV to `output_dir`.
    5. AC4: Saves short execution report to `report_path`.

    Args:
        ratings_df: Raw ratings table.
        movies_df: Raw movies metadata catalog.
        users_df: Optional raw users metadata table.
        min_ratings: Optional threshold override. Defaults to value in params.yaml.
        params_path: Optional path to params.yaml.
        output_dir: Directory where clean data will be written.
        report_path: File path where cleaning report JSON will be written.

    Returns:
        Structured summary report dictionary.
    """
    # 1. Validation step (AC1 & AC2): Stops immediately if quality checks fail
    validation_report = validate_raw_data(ratings_df, movies_df, users_df=users_df)

    # 2. Threshold determination
    if min_ratings is None:
        settings = load_data_settings(params_path)
        threshold = settings.min_ratings
        target_output_dir = Path(output_dir or settings.processed_dir)
        target_report_path = Path(report_path or settings.report_path)
    else:
        threshold = int(min_ratings)
        target_output_dir = Path(output_dir or "data/processed")
        target_report_path = Path(report_path or (target_output_dir / "cleaning_report.json"))

    # 3. Apply threshold filtering (AC3)
    clean_ratings, filter_stats = filter_ratings_by_threshold(
        ratings_df, min_ratings=threshold, iterative=True
    )
    clean_movies = filter_movies_catalog(movies_df, clean_ratings)

    # 4. Standardize dtypes & column names (EDA Section 10 recommendation 2)
    clean_ratings, clean_movies = standardize_dtypes_and_columns(clean_ratings, clean_movies)

    # 5. Save clean data (Parquet + CSV)
    target_output_dir.mkdir(parents=True, exist_ok=True)
    ratings_parquet = target_output_dir / "ratings.parquet"
    movies_parquet = target_output_dir / "movies.parquet"
    ratings_csv = target_output_dir / "ratings.csv"
    movies_csv = target_output_dir / "movies.csv"

    clean_ratings.to_parquet(ratings_parquet, index=False)
    clean_movies.to_parquet(movies_parquet, index=False)
    clean_ratings.to_csv(ratings_csv, index=False)
    clean_movies.to_csv(movies_csv, index=False)

    # 6. Build and save short execution report (AC4)
    target_report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "threshold_k": threshold,
        "schema": {
            "ratings_columns": list(clean_ratings.columns),
            "ratings_dtypes": {col: str(dtype) for col, dtype in clean_ratings.dtypes.items()},
            "movies_columns": list(clean_movies.columns),
            "movies_dtypes": {col: str(dtype) for col, dtype in clean_movies.dtypes.items()},
        },
        "validation": {
            "status": validation_report["status"],
            "checks_passed": validation_report["checks_passed"],
            "problems_found": {
                "missing_values": 0,
                "ratings_outside_range": 0,
                "duplicate_ratings": 0,
                "orphan_ids": 0,
            },
        },
        "filtering_summary": {
            "initial_ratings": filter_stats["initial_ratings"],
            "clean_ratings": filter_stats["clean_ratings"],
            "ratings_removed": filter_stats["ratings_removed"],
            "pct_ratings_removed": filter_stats["pct_ratings_removed"],
            "initial_movies": filter_stats["initial_movies"],
            "clean_movies": filter_stats["clean_movies"],
            "movies_removed": filter_stats["movies_removed"],
            "pct_movies_removed": filter_stats["pct_movies_removed"],
            "initial_users": filter_stats["initial_users"],
            "clean_users": filter_stats["clean_users"],
            "users_removed": filter_stats["users_removed"],
            "pct_users_removed": filter_stats["pct_users_removed"],
        },
        "saved_files": {
            "ratings_parquet": str(ratings_parquet),
            "movies_parquet": str(movies_parquet),
            "ratings_csv": str(ratings_csv),
            "movies_csv": str(movies_csv),
        },
    }

    with open(target_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def track_clean_data_with_dvc(processed_dir: str | Path = "data/processed") -> bool:
    """Track processed clean data directory with DVC.

    Runs 'dvc add <processed_dir>' to generate .dvc pointer files.
    """
    import shutil

    dir_path = Path(processed_dir)
    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist. Nothing to track with DVC.", file=sys.stderr)
        return False

    dvc_bin = shutil.which("dvc")
    if not dvc_bin:
        env_dvc = Path(sys.executable).parent / "dvc"
        if env_dvc.exists():
            dvc_bin = str(env_dvc)
        else:
            dvc_bin = "dvc"

    cmd = [dvc_bin, "add", str(dir_path)]
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"DVC tracking updated:\n{res.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: 'dvc add' failed:\n{e.stderr.strip()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Warning: 'dvc' command not found in PATH.", file=sys.stderr)
        return False


def main() -> None:
    """CLI runner for data validation and cleaning."""
    parser = argparse.ArgumentParser(
        description="Validate and filter MovieLens data using EDA threshold (US1.3)."
    )
    parser.add_argument(
        "--params-path",
        type=str,
        default="params.yaml",
        help="Path to params.yaml (default: params.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to raw data directory (default: from params.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to output processed directory (default: from params.yaml)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Override minimum rating threshold (default: read from params.yaml)",
    )
    parser.add_argument(
        "--dvc",
        action="store_true",
        help="Automatically track saved clean data with DVC (dvc add)",
    )
    args = parser.parse_args()

    # Load configuration
    settings = load_data_settings(args.params_path)
    data_dir = Path(args.data_dir or settings.raw_dir)
    output_dir = Path(args.output_dir or settings.processed_dir)
    threshold = args.min_ratings or settings.min_ratings

    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    # Fallback to data/raw_small if data/raw does not exist
    if not ratings_path.exists() or not movies_path.exists():
        fallback_dir = Path("data/raw_small")
        if fallback_dir.exists() and (fallback_dir / "ratings.csv").exists():
            data_dir = fallback_dir
            ratings_path = data_dir / "ratings.csv"
            movies_path = data_dir / "movies.csv"
            print(f"Notice: '{settings.raw_dir}' not found. Using fallback '{fallback_dir}'.")

    if not ratings_path.exists() or not movies_path.exists():
        print(f"Error: Missing ratings.csv or movies.csv in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ratings from {ratings_path}...")
    ratings_df = pd.read_csv(ratings_path)
    print(f"Loading movies from {movies_path}...")
    movies_df = pd.read_csv(movies_path)

    try:
        report = clean_and_save_data(
            ratings_df=ratings_df,
            movies_df=movies_df,
            min_ratings=threshold,
            params_path=args.params_path,
            output_dir=output_dir,
        )
    except DataValidationError as e:
        print(f"\n[PIPELINE HALTED] Data quality check failed:\n{e}", file=sys.stderr)
        sys.exit(1)

    print("\n================ DATA CLEANING REPORT (US1.3) ================")
    print(f"Validation Status    : {report['validation']['status']}")
    print(f"Threshold applied (k): {report['threshold_k']}")
    f_sum = report["filtering_summary"]
    print(f"Ratings: {f_sum['initial_ratings']:,} -> {f_sum['clean_ratings']:,} (-{f_sum['ratings_removed']:,}, -{f_sum['pct_ratings_removed']}%)")
    print(f"Movies : {f_sum['initial_movies']:,} -> {f_sum['clean_movies']:,} (-{f_sum['movies_removed']:,}, -{f_sum['pct_movies_removed']}%)")
    print(f"Users  : {f_sum['initial_users']:,} -> {f_sum['clean_users']:,} (-{f_sum['users_removed']:,}, -{f_sum['pct_users_removed']}%)")
    print(f"Clean files saved to : {output_dir}")
    print(f"Report saved to      : {settings.report_path}")
    print("==============================================================")

    if args.dvc:
        print("\nTracking clean data with DVC...")
        track_clean_data_with_dvc(output_dir)


if __name__ == "__main__":
    main()
