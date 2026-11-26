"""Data validation module for MovieLens recommender pipeline.

Implements automated quality checks at the entry point of the data pipeline:
- Missing values detection per column in users, movies, and ratings.
- Rating range validation (0.5 - 5.0, configurable).
- Duplicate user-movie rating detection.
- Referential integrity checks (orphan movie and user IDs).

If any check fails, a DataValidationError is raised with a detailed message
explaining the problem, stopping the pipeline immediately
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import pandas as pd


class DataValidationError(ValueError):
    """Raised when one or more data quality checks fail."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


def _resolve_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required column. Expected one of: {candidates}. Available: {list(df.columns)}")
    return None


def check_missing_values(df: pd.DataFrame, table_name: str, cols: Optional[List[str]] = None) -> List[str]:
    """Check for missing / null values across columns of a DataFrame."""
    if df is None or df.empty:
        return [f"Table '{table_name}' is empty or None."]

    target_cols = cols if cols is not None else list(df.columns)
    issues: List[str] = []
    for col in target_cols:
        if col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                issues.append(f"Table '{table_name}': column '{col}' contains {null_count:,} missing value(s).")
    return issues


def check_rating_range(
    ratings_df: pd.DataFrame,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
    rating_col: Optional[str] = None,
) -> List[str]:
    """Verify that all ratings fall within [min_rating, max_rating]."""
    if ratings_df is None or ratings_df.empty:
        return ["Ratings table is empty or None."]

    col = rating_col or _resolve_col(ratings_df, ["rating", "Rating", "score"])
    ratings_series = pd.to_numeric(ratings_df[col], errors="coerce")

    nan_converted = int(ratings_series.isna().sum() - ratings_df[col].isna().sum())
    if nan_converted > 0:
        return [f"Table 'ratings': column '{col}' contains {nan_converted} non-numeric rating value(s)."]

    valid_ratings = ratings_series.dropna()
    out_of_bounds = valid_ratings[(valid_ratings < min_rating) | (valid_ratings > max_rating)]
    if len(out_of_bounds) > 0:
        found_min = float(valid_ratings.min())
        found_max = float(valid_ratings.max())
        return [
            f"Table 'ratings': found {len(out_of_bounds):,} rating(s) outside [{min_rating}, {max_rating}]. "
            f"Min observed: {found_min}, Max observed: {found_max}."
        ]
    return []


def check_duplicate_ratings(
    ratings_df: pd.DataFrame,
    user_col: Optional[str] = None,
    movie_col: Optional[str] = None,
) -> List[str]:
    """Check for repeated user-movie ratings."""
    if ratings_df is None or ratings_df.empty:
        return ["Ratings table is empty or None."]

    u_col = user_col or _resolve_col(ratings_df, ["userId", "user_id", "UserId", "user"])
    m_col = movie_col or _resolve_col(ratings_df, ["movieId", "movie_id", "MovieId", "movie"])

    dup_count = int(ratings_df.duplicated(subset=[u_col, m_col]).sum())
    if dup_count > 0:
        return [f"Table 'ratings': found {dup_count:,} duplicate rating(s) for the same ({u_col}, {m_col}) pair."]
    return []


def check_orphan_ids(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    movie_col: Optional[str] = None,
    user_col: Optional[str] = None,
) -> List[str]:
    """Verify referential integrity between ratings and metadata catalogs."""
    issues: List[str] = []
    if ratings_df is None or ratings_df.empty:
        return ["Ratings table is empty or None."]
    if movies_df is None or movies_df.empty:
        return ["Movies catalog is empty or None."]

    r_movie_col = movie_col or _resolve_col(ratings_df, ["movieId", "movie_id", "MovieId", "movie"])
    m_movie_col = _resolve_col(movies_df, ["movieId", "movie_id", "MovieId", "movie", "id"])

    valid_movie_ids = set(movies_df[m_movie_col].dropna().unique())
    orphan_movies = set(ratings_df[r_movie_col].dropna().unique()) - valid_movie_ids

    if orphan_movies:
        sample_orphans = sorted(list(orphan_movies))[:5]
        sample_str = ", ".join(map(str, sample_orphans))
        if len(orphan_movies) > 5:
            sample_str += ", ..."
        issues.append(
            f"Referential integrity: found {len(orphan_movies):,} orphan movie ID(s) in ratings "
            f"not present in movies catalog (e.g. [{sample_str}])."
        )

    if users_df is not None and not users_df.empty:
        r_user_col = user_col or _resolve_col(ratings_df, ["userId", "user_id", "UserId", "user"])
        u_user_col = _resolve_col(users_df, ["userId", "user_id", "UserId", "user", "id"])
        valid_user_ids = set(users_df[u_user_col].dropna().unique())
        orphan_users = set(ratings_df[r_user_col].dropna().unique()) - valid_user_ids

        if orphan_users:
            sample_u = sorted(list(orphan_users))[:5]
            sample_u_str = ", ".join(map(str, sample_u))
            if len(orphan_users) > 5:
                sample_u_str += ", ..."
            issues.append(
                f"Referential integrity: found {len(orphan_users):,} orphan user ID(s) in ratings "
                f"not present in users catalog (e.g. [{sample_u_str}])."
            )

    return issues


def validate_raw_data(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: Optional[pd.DataFrame] = None,
    min_rating: float = 0.5,
    max_rating: float = 5.0,
) -> Dict[str, Any]:
    """Execute all basic data validation checks at the start of the pipeline.

    Args:
        ratings_df: Ratings table (requires user, movie, and rating columns).
        movies_df: Movies catalog table (requires movie column).
        users_df: Optional users metadata table.
        min_rating: Minimum allowed rating value (default: 0.5).
        max_rating: Maximum allowed rating value (default: 5.0).

    Returns:
        Summary report dictionary with check results if all checks pass.

    Raises:
        DataValidationError: If any quality check fails, stopping the pipeline (AC2).
    """
    errors: List[str] = []

    # 1. Missing values per column in users, movies, and ratings
    errors.extend(check_missing_values(ratings_df, "ratings"))
    errors.extend(check_missing_values(movies_df, "movies"))
    if users_df is not None:
        errors.extend(check_missing_values(users_df, "users"))

    # 2. Rating range check (0.5 to 5.0)
    errors.extend(check_rating_range(ratings_df, min_rating=min_rating, max_rating=max_rating))

    # 3. Repeated user-movie ratings check
    errors.extend(check_duplicate_ratings(ratings_df))

    # 4. Referential integrity (orphan IDs)
    errors.extend(check_orphan_ids(ratings_df, movies_df, users_df=users_df))

    # Fail fast if any issues were detected
    if errors:
        error_msg = (
            f"Data pipeline halted! Found {len(errors)} data quality issue(s):\n"
            + "\n".join(f"  - {err}" for err in errors)
        )
        raise DataValidationError(error_msg, errors=errors)

    # Return structured report for logging and downstream tasks
    u_col = _resolve_col(ratings_df, ["userId", "user_id", "UserId", "user"])
    m_col = _resolve_col(ratings_df, ["movieId", "movie_id", "MovieId", "movie"])
    r_col = _resolve_col(ratings_df, ["rating", "Rating", "score"])

    return {
        "status": "PASSED",
        "total_ratings": len(ratings_df),
        "total_movies": len(movies_df),
        "total_users": int(ratings_df[u_col].nunique()),
        "rating_range": {
            "min_allowed": min_rating,
            "max_allowed": max_rating,
            "min_observed": float(ratings_df[r_col].min()),
            "max_observed": float(ratings_df[r_col].max()),
        },
        "checks_passed": [
            "missing_values",
            "rating_range",
            "duplicate_ratings",
            "orphan_ids",
        ],
    }


def main() -> None:
    """CLI runner for data validation."""
    parser = argparse.ArgumentParser(description="Validate raw MovieLens dataset quality.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw data directory containing ratings.csv and movies.csv (default: data/raw)",
    )
    parser.add_argument("--min-rating", type=float, default=0.5, help="Minimum valid rating (default: 0.5)")
    parser.add_argument("--max-rating", type=float, default=5.0, help="Maximum valid rating (default: 5.0)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        fallback_dir = Path("data/raw_small")
        if fallback_dir.exists() and (fallback_dir / "ratings.csv").exists():
            data_dir = fallback_dir
            ratings_path = data_dir / "ratings.csv"
            movies_path = data_dir / "movies.csv"

    print(f"Loading ratings from {ratings_path}...")
    ratings_df = pd.read_csv(ratings_path)
    print(f"Loading movies from {movies_path}...")
    movies_df = pd.read_csv(movies_path)

    try:
        report = validate_raw_data(
            ratings_df,
            movies_df,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
        )
        print("Data validation successful! Checks passed:")
        for check in report["checks_passed"]:
            print(f"  ✓ {check}")
        print(
            f"Summary: {report['total_ratings']:,} ratings, "
            f"{report['total_movies']:,} movies, "
            f"{report['total_users']:,} users."
        )
    except DataValidationError as e:
        print(f"Validation failed:\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
