"""Batch recommendation job execution with structured logging and metadata.

This module implements the end-to-end execution of batch recommendations,
fulfilling US 2.1 Acceptance Criteria:
- AC1: Job completes successfully and produces recommendations.
- AC5: Output format is valid Parquet for downstream consumption.
- AC6: Exposes execution status (success/failure) and logs execution metadata.
"""
import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional, Union
import pandas as pd

from src.models.baseline import (
    compute_movie_stats,
    generate_batch_recommendations,
    export_recommendations_to_parquet,
)

logger = logging.getLogger("recsys.batch")


def setup_logger(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure structured logger for the batch execution pipeline.

    Args:
        level: Logging level (e.g. logging.INFO).
        log_file: Optional file path where logs should be appended.

    Returns:
        Configured logging.Logger instance.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def run_batch_job(
    ratings_df: pd.DataFrame,
    movies_df: Optional[pd.DataFrame] = None,
    user_ids: Optional[Iterable[int]] = None,
    output_path: Union[str, Path] = "data/processed/recommendations.parquet",
    top_n: int = 5,
    min_ratings: int = 50,
    exclude_rated: bool = True,
    log_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end recommendation batch job with logging and status reporting.

    Fulfills AC1 (completion), AC5 (Parquet format), and AC6 (status & metadata logging).

    Args:
        ratings_df: Historical ratings DataFrame containing user_id, movie_id, rating.
        movies_df: Optional movie metadata containing movie_id, title, genres.
        user_ids: Optional iterable of specific user IDs. If None, all unique users in ratings_df.
        output_path: Path where recommendations Parquet file will be written.
        top_n: Number of top recommendations per user (default 5).
        min_ratings: Minimum ratings required for candidate movies.
        m_prior: Minimum vote weight threshold for Bayesian average.
        c_weight: Global mean rating prior.
        exclude_rated: Whether to filter out movies users already rated.
        log_file: Optional path to save log output file.

    Returns:
        Dictionary with execution metadata and status ('SUCCESS' or 'FAILED').
    """
    setup_logger(log_file=log_file)

    start_wall_time = datetime.now(timezone.utc).isoformat()
    start_perf = time.perf_counter()

    metadata: Dict[str, Any] = {
        "status": "RUNNING",
        "start_time": start_wall_time,
        "end_time": None,
        "duration_seconds": None,
        "top_n": top_n,
        "min_ratings": min_ratings,
        "users_count": 0,
        "recommendations_count": 0,
        "output_path": str(output_path),
        "output_size_bytes": None,
        "error": None,
    }

    logger.info("Starting recommendation batch job (US 2.1)...")
    logger.info("Parameters: top_n=%d, min_ratings=%d, exclude_rated=%s", top_n, min_ratings, exclude_rated)

    try:
        if ratings_df is None or ratings_df.empty:
            raise ValueError("ratings_df cannot be empty or None")

        # Determine target user list
        if user_ids is None:
            target_users = sorted(ratings_df["user_id"].unique())
        else:
            target_users = list(user_ids)

        metadata["users_count"] = len(target_users)
        logger.info("Identified %d target users for recommendations", len(target_users))

        # Step 1: Compute movie statistics and Bayesian scores
        logger.info("Computing baseline scores and statistics per movie...")
        ranked_movies = compute_movie_stats(
            ratings_df=ratings_df,
            min_ratings=min_ratings,
            movies_df=movies_df,
        )
        logger.info("Qualified candidate movies: %d", len(ranked_movies))

        # Step 2: Generate top-N batch recommendations
        logger.info("Generating top-%d recommendations for %d users...", top_n, len(target_users))
        batch_recs = generate_batch_recommendations(
            user_ids=target_users,
            ranked_movies_df=ranked_movies,
            ratings_df=ratings_df,
            n=top_n,
            exclude_rated=exclude_rated,
        )
        metadata["recommendations_count"] = len(batch_recs)
        logger.info("Total recommendations generated: %d rows", len(batch_recs))

        # Step 3: Export to Parquet format (AC5)
        logger.info("Exporting batch recommendations to Parquet: %s", output_path)
        saved_file = export_recommendations_to_parquet(batch_recs, output_path=output_path)
        metadata["output_size_bytes"] = saved_file.stat().st_size
        logger.info("Successfully exported Parquet file (%d bytes)", metadata["output_size_bytes"])

        # Record success status (AC6)
        end_perf = time.perf_counter()
        metadata["status"] = "SUCCESS"
        metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        metadata["duration_seconds"] = round(end_perf - start_perf, 4)

        logger.info(
            "Batch job finished with status SUCCESS in %.3f seconds. %d recommendations written to %s",
            metadata["duration_seconds"],
            metadata["recommendations_count"],
            output_path,
        )
        return metadata

    except Exception as exc:
        end_perf = time.perf_counter()
        metadata["status"] = "FAILED"
        metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        metadata["duration_seconds"] = round(end_perf - start_perf, 4)
        metadata["error"] = str(exc)

        logger.error(
            "Batch job failed with status FAILED after %.3f seconds: %s",
            metadata["duration_seconds"],
            exc,
            exc_info=True,
        )
        raise exc


def main() -> None:
    """CLI entrypoint for executing batch recommendations."""
    parser = argparse.ArgumentParser(description="MovieLens Top-N Batch Recommendation Job (US 2.1)")
    parser.add_argument("--ratings-path", type=str, required=True, help="Path to ratings data (CSV/DAT/Parquet)")
    parser.add_argument("--movies-path", type=str, default=None, help="Optional path to movies metadata")
    parser.add_argument("--output-path", type=str, default="data/processed/recommendations.parquet", help="Output path")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations per user (default 5)")
    parser.add_argument("--min-ratings", type=int, default=50, help="Minimum ratings required (default 50)")
    parser.add_argument("--log-file", type=str, default=None, help="Optional log file path")

    args = parser.parse_args()
    setup_logger(log_file=args.log_file)

    logger.info("Loading ratings from %s", args.ratings_path)
    ratings_path = Path(args.ratings_path)
    if ratings_path.suffix == ".parquet":
        ratings_df = pd.read_parquet(ratings_path)
    elif "::" in open(ratings_path, "r", encoding="latin-1").readline():
        ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )
    else:
        ratings_df = pd.read_csv(ratings_path)

    movies_df = None
    if args.movies_path:
        movies_path = Path(args.movies_path)
        if movies_path.suffix == ".parquet":
            movies_df = pd.read_parquet(movies_path)
        elif "::" in open(movies_path, "r", encoding="latin-1").readline():
            movies_df = pd.read_csv(
                movies_path,
                sep="::",
                engine="python",
                names=["movie_id", "title", "genres"],
                encoding="latin-1",
            )
        else:
            movies_df = pd.read_csv(movies_path)

    run_batch_job(
        ratings_df=ratings_df,
        movies_df=movies_df,
        output_path=args.output_path,
        top_n=args.top_n,
        min_ratings=args.min_ratings,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
