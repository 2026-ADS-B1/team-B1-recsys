"""Baseline recommendation algorithms for MovieLens.

This module implements popularity and rating-based baselines, computing
aggregated statistics, weighted ranking scores, and batch top-N recommendations.
"""
from typing import Iterable, List, Optional, Union
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

        ordered_cols = ["movie_id"]
        for col in ["title", "genres"]:
            if col in qualified.columns:
                ordered_cols.append(col)
        ordered_cols += ["rating_count", "mean_rating", "score"]
        qualified = qualified[ordered_cols]

    # AC4: Recommendations must appear in descending relevance order
    qualified = qualified.sort_values(by="score", ascending=False).reset_index(drop=True)

    return qualified


def get_user_recommendations(
    user_id: int,
    ranked_movies_df: pd.DataFrame,
    ratings_df: Optional[pd.DataFrame] = None,
    n: int = 5,
    exclude_rated: bool = True,
) -> pd.DataFrame:
    """Generate top-N recommendations for a specific user.

    Fulfills US2.1 acceptance criteria:
    - AC2: Returns at most N recommendations (exactly N if >= N eligible items exist).
    - AC3: No artificial or duplicate items generated. If fewer than N eligible items
           exist, returns all available without duplicates.
    - AC4: Results appear in descending order of relevance score.

    Args:
        user_id: Target user identifier.
        ranked_movies_df: Pre-ranked movie DataFrame (from compute_movie_stats).
        ratings_df: Optional ratings history DataFrame used to exclude movies already seen.
        n: Maximum number of recommendations to return (default 5).
        exclude_rated: Whether to filter out movies the user has already rated.

    Returns:
        DataFrame containing up to N recommendations with a 1-based 'rank' column.
    """
    if ranked_movies_df is None or ranked_movies_df.empty:
        cols = ["rank"] + list(ranked_movies_df.columns) if ranked_movies_df is not None else ["rank", "movie_id", "score"]
        return pd.DataFrame(columns=cols)

    candidates = ranked_movies_df.copy()

    # Exclude movies the user has already rated (if ratings_df is provided and user has history)
    if exclude_rated and ratings_df is not None and not ratings_df.empty and "user_id" in ratings_df.columns:
        user_rated = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
        candidates = candidates[~candidates["movie_id"].isin(user_rated)]

    # AC3: Remove any possible duplicate movie_ids (preserve highest score first)
    candidates = candidates.drop_duplicates(subset=["movie_id"], keep="first")

    # AC4: Ensure descending score order
    if "score" in candidates.columns:
        candidates = candidates.sort_values(by="score", ascending=False)

    # AC2 & AC3: Take at most N recommendations
    recs = candidates.head(n).copy().reset_index(drop=True)

    # Add 1-based rank column
    recs.insert(0, "rank", range(1, len(recs) + 1))

    return recs


def generate_batch_recommendations(
    user_ids: Iterable[int],
    ranked_movies_df: pd.DataFrame,
    ratings_df: Optional[pd.DataFrame] = None,
    n: int = 5,
    exclude_rated: bool = True,
) -> pd.DataFrame:
    """Generate batch top-N recommendations for a collection of users.

    Args:
        user_ids: Iterable of user identifiers to generate recommendations for.
        ranked_movies_df: Pre-ranked movie DataFrame (from compute_movie_stats).
        ratings_df: Optional ratings history DataFrame to exclude already rated movies.
        n: Number of recommendations per user.
        exclude_rated: Whether to filter out already rated movies.

    Returns:
        Combined DataFrame with 'user_id' as the first column, followed by 'rank'
        and recommendation details.
    """
    all_recs: List[pd.DataFrame] = []

    for uid in user_ids:
        user_recs = get_user_recommendations(
            user_id=uid,
            ranked_movies_df=ranked_movies_df,
            ratings_df=ratings_df,
            n=n,
            exclude_rated=exclude_rated,
        )
        if not user_recs.empty:
            user_recs.insert(0, "user_id", uid)
            all_recs.append(user_recs)

    if not all_recs:
        base_cols = ["user_id", "rank"] + [c for c in ranked_movies_df.columns if c not in ("user_id", "rank")]
        return pd.DataFrame(columns=base_cols)

    return pd.concat(all_recs, ignore_index=True)
