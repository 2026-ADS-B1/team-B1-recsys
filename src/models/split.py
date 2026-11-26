"""Reproducible train/test split for recommender systems ratings data.

Fulfills US 2.2 Acceptance Criteria:
- AC4: Fair comparison across all models on identical train/test splits.
- AC6: Split parameters configurable via params.yaml without modifying code.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml


def load_split_params(params_path: Union[str, Path] = "params.yaml") -> Dict[str, Any]:
    """Load split configuration from params.yaml.

    Args:
        params_path: Path to the YAML configuration file.

    Returns:
        Dictionary with split parameters ('test_size', 'random_state', etc.).
    """
    path = Path(params_path)
    if not path.exists():
        return {
            "test_size": 0.2,
            "random_state": 42,
            "stratified_by_user": True,
            "min_user_ratings": 5,
        }

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    split_cfg = data.get("models", {}).get("split", {})
    return {
        "test_size": float(split_cfg.get("test_size", 0.2)),
        "random_state": int(split_cfg.get("random_state", 42)),
        "stratified_by_user": bool(split_cfg.get("stratified_by_user", True)),
        "min_user_ratings": int(split_cfg.get("min_user_ratings", 5)),
    }


def train_test_split_ratings(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratified_by_user: bool = True,
    min_user_ratings: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into reproducible training and testing sets.

    When `stratified_by_user` is True, the function ensures each user with at least
    `min_user_ratings` has a proportional hold-out sample in the test set.
    Users with fewer than `min_user_ratings` remain exclusively in the training set
    to avoid cold-start issues during evaluation.

    Args:
        ratings_df: DataFrame with at least ['user_id', 'movie_id', 'rating'].
        test_size: Fraction of ratings to include in the test set (0.0 < test_size < 1.0).
        random_state: Random seed for reproducibility.
        stratified_by_user: Whether to sample test interactions per user.
        min_user_ratings: Minimum ratings required for a user to be sampled into test.

    Returns:
        Tuple of (train_df, test_df) DataFrames.

    Raises:
        ValueError: If ratings_df is empty or missing required columns, or invalid test_size.
    """
    if ratings_df is None or ratings_df.empty:
        raise ValueError("ratings_df cannot be empty or None")

    required_cols = {"user_id", "movie_id", "rating"}
    missing = required_cols - set(ratings_df.columns)
    if missing:
        raise ValueError(f"ratings_df missing required columns: {sorted(missing)}")

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    rng = np.random.RandomState(random_state)

    if not stratified_by_user:
        # Standard random split across all interactions
        n_test = int(round(len(ratings_df) * test_size))
        shuffled_idx = rng.permutation(ratings_df.index)
        test_idx = shuffled_idx[:n_test]
        train_idx = shuffled_idx[n_test:]

        train_df = ratings_df.loc[train_idx].copy().reset_index(drop=True)
        test_df = ratings_df.loc[test_idx].copy().reset_index(drop=True)
        return train_df, test_df

    # Stratified per-user split
    test_indices = []

    # Group index by user
    user_groups = ratings_df.groupby("user_id").indices

    for user_id, idx_array in user_groups.items():
        n_user_ratings = len(idx_array)
        if n_user_ratings >= min_user_ratings:
            n_user_test = max(1, int(round(n_user_ratings * test_size)))
            # Sample test indices for this user
            shuffled = rng.permutation(idx_array)
            test_indices.extend(shuffled[:n_user_test])

    test_idx_set = set(test_indices)
    all_indices = ratings_df.index.values

    train_mask = ~np.isin(all_indices, list(test_idx_set))

    train_df = ratings_df[train_mask].copy().reset_index(drop=True)
    test_df = ratings_df[~train_mask].copy().reset_index(drop=True)

    return train_df, test_df
