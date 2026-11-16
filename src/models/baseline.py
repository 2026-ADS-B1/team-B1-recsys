"""Baseline recommendation algorithms for MovieLens.

This module implements popularity and rating-based baselines, computing
aggregated statistics and weighted ranking scores (IMDb / Bayesian average formula).
"""
from typing import Optional
import pandas as pd


def compute_movie_stats(
    ratings_df: pd.DataFrame,
    min_ratings: int = 50,
    movies_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute aggregated rating statistics and ranking scores per movie.

    Args:
        ratings_df: DataFrame containing at least 'movie_id' and 'rating'.
        min_ratings: Minimum number of ratings required for a movie to be included.
        movies_df: Optional DataFrame with 'movie_id', 'title', and 'genres'.

    Returns:
        DataFrame sorted in descending order of score, with columns:
        ['movie_id', 'title', 'genres', 'rating_count', 'mean_rating', 'score']
        (or without 'title' and 'genres' if movies_df is not supplied).

    Raises:
        ValueError: If ratings_df is empty or missing required columns.
    """
    if ratings_df is None or ratings_df.empty:
        raise ValueError("ratings_df cannot be empty or None")

    required_cols = {"movie_id", "rating"}
    missing = required_cols - set(ratings_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in ratings_df: {missing}")

    # Compute global mean rating (C)
    global_mean = ratings_df["rating"].mean()

    # Aggregate by movie
    stats = (
        ratings_df.groupby("movie_id")["rating"]
        .agg(rating_count="count", mean_rating="mean")
        .reset_index()
    )

    # Filter movies with at least min_ratings votes
    qualified = stats[stats["rating_count"] >= min_ratings].copy()

    if qualified.empty:
        # If no movie meets the threshold, return empty DataFrame with expected structure
        empty_cols = ["movie_id", "rating_count", "mean_rating", "score"]
        if movies_df is not None:
            empty_cols = ["movie_id", "title", "genres"] + empty_cols[1:]
        return pd.DataFrame(columns=empty_cols)

    # Weighted rating (IMDb / Bayesian formula):
    # score = (v / (v + m)) * R + (m / (v + m)) * C
    # where v = rating_count, m = min_ratings, R = mean_rating, C = global_mean
    v = qualified["rating_count"]
    m = min_ratings
    r = qualified["mean_rating"]
    c = global_mean

    qualified["score"] = ((v / (v + m)) * r) + ((m / (v + m)) * c)
    qualified["score"] = qualified["score"].round(4)
    qualified["mean_rating"] = qualified["mean_rating"].round(4)

    # If movies metadata is provided, merge title and genres
    if movies_df is not None and not movies_df.empty:
        cols_to_merge = ["movie_id"]
        for col in ["title", "genres"]:
            if col in movies_df.columns:
                cols_to_merge.append(col)

        qualified = qualified.merge(movies_df[cols_to_merge], on="movie_id", how="left")

        # Organize column order
        ordered_cols = ["movie_id"]
        for col in ["title", "genres"]:
            if col in qualified.columns:
                ordered_cols.append(col)
        ordered_cols += ["rating_count", "mean_rating", "score"]
        qualified = qualified[ordered_cols]

    # AC4: Recommendations must appear in descending relevance order
    qualified = qualified.sort_values(by="score", ascending=False).reset_index(drop=True)

    return qualified

