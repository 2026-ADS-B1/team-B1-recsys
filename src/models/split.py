"""Reproducible train/test/validation split for recommender systems ratings data.

Fulfills US 2.2 Acceptance Criteria:
- AC4: Fair comparison across all models on identical splits.
- AC6: Split parameters configurable via params.yaml without modifying code.
- Temporal hold-out: Chronological split by timestamp per user to prevent data leakage.
- 3-Way Split: Train (learning), Validation (early stopping / tuning), Test (unbiased benchmark).
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
        Dictionary with split parameters ('val_size', 'test_size', 'random_state', 'temporal', etc.).
    """
    path = Path(params_path)
    if not path.exists():
        return {
            "val_size": 0.1,
            "test_size": 0.1,
            "random_state": 42,
            "stratified_by_user": True,
            "min_user_ratings": 5,
            "temporal": True,
        }

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    split_cfg = data.get("models", {}).get("split", {})
    return {
        "val_size": float(split_cfg.get("val_size", 0.1)),
        "test_size": float(split_cfg.get("test_size", 0.1)),
        "random_state": int(split_cfg.get("random_state", 42)),
        "stratified_by_user": bool(split_cfg.get("stratified_by_user", True)),
        "min_user_ratings": int(split_cfg.get("min_user_ratings", 5)),
        "temporal": bool(split_cfg.get("temporal", True)),
    }


def train_test_split_ratings(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratified_by_user: bool = True,
    min_user_ratings: int = 5,
    temporal: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into reproducible training and testing sets (2-way split)."""
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
        n_test = int(round(len(ratings_df) * test_size))
        if temporal and "timestamp" in ratings_df.columns:
            sorted_indices = ratings_df.sort_values(by="timestamp").index.to_numpy()
            test_idx = sorted_indices[-n_test:]
            train_idx = sorted_indices[:-n_test]
        else:
            shuffled_idx = rng.permutation(ratings_df.index)
            test_idx = shuffled_idx[:n_test]
            train_idx = shuffled_idx[n_test:]

        return (
            ratings_df.loc[train_idx].copy().reset_index(drop=True),
            ratings_df.loc[test_idx].copy().reset_index(drop=True),
        )

    test_indices = []
    user_groups = ratings_df.groupby("user_id").indices

    for user_id, idx_array in user_groups.items():
        n_user_ratings = len(idx_array)
        if n_user_ratings >= min_user_ratings:
            n_user_test = max(1, int(round(n_user_ratings * test_size)))
            if temporal and "timestamp" in ratings_df.columns:
                user_timestamps = ratings_df.loc[idx_array, "timestamp"].to_numpy()
                sorted_idx = idx_array[np.argsort(user_timestamps)]
                test_indices.extend(sorted_idx[-n_user_test:])
            else:
                shuffled = rng.permutation(idx_array)
                test_indices.extend(shuffled[:n_user_test])

    test_idx_set = set(test_indices)
    all_indices = ratings_df.index.values

    train_mask = ~np.isin(all_indices, list(test_idx_set))
    return (
        ratings_df[train_mask].copy().reset_index(drop=True),
        ratings_df[~train_mask].copy().reset_index(drop=True),
    )


def train_val_test_split_ratings(
    ratings_df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    stratified_by_user: bool = True,
    min_user_ratings: int = 5,
    temporal: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings into Train, Validation, and Test sets (3-way split).

    - Train: used for SGD parameter optimization (past interactions).
    - Validation: used for monitoring generalization and Early Stopping (recent interactions).
    - Test: unbiased evaluation holdout (latest interactions).

    Args:
        ratings_df: DataFrame with ['user_id', 'movie_id', 'rating', 'timestamp'].
        val_size: Fraction of interactions for validation (0.0 < val_size < 1.0).
        test_size: Fraction of interactions for testing (0.0 < test_size < 1.0).
        random_state: Seed for reproducibility.
        stratified_by_user: Whether to hold out interactions per user.
        min_user_ratings: Minimum interactions required for a user to be sampled into val/test.
        temporal: Whether to split interactions chronologically by timestamp.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if ratings_df is None or ratings_df.empty:
        raise ValueError("ratings_df cannot be empty or None")

    required_cols = {"user_id", "movie_id", "rating"}
    missing = required_cols - set(ratings_df.columns)
    if missing:
        raise ValueError(f"ratings_df missing required columns: {sorted(missing)}")

    if not (0.0 < val_size < 1.0 and 0.0 < test_size < 1.0 and (val_size + test_size) < 1.0):
        raise ValueError(
            f"val_size ({val_size}) and test_size ({test_size}) must be > 0 and sum to < 1.0"
        )

    rng = np.random.RandomState(random_state)
    val_indices = []
    test_indices = []

    if not stratified_by_user:
        n_test = int(round(len(ratings_df) * test_size))
        n_val = int(round(len(ratings_df) * val_size))
        if temporal and "timestamp" in ratings_df.columns:
            sorted_indices = ratings_df.sort_values(by="timestamp").index.to_numpy()
            test_idx = sorted_indices[-n_test:]
            val_idx = sorted_indices[-(n_test + n_val) : -n_test]
            train_idx = sorted_indices[: -(n_test + n_val)]
        else:
            shuffled = rng.permutation(ratings_df.index)
            test_idx = shuffled[:n_test]
            val_idx = shuffled[n_test : n_test + n_val]
            train_idx = shuffled[n_test + n_val :]

        return (
            ratings_df.loc[train_idx].copy().reset_index(drop=True),
            ratings_df.loc[val_idx].copy().reset_index(drop=True),
            ratings_df.loc[test_idx].copy().reset_index(drop=True),
        )

    user_groups = ratings_df.groupby("user_id").indices

    for user_id, idx_array in user_groups.items():
        n_u = len(idx_array)
        if n_u >= min_user_ratings:
            n_u_test = max(1, int(round(n_u * test_size)))
            n_u_val = max(1, int(round(n_u * val_size)))

            # Ensure at least 1 interaction remains in train
            if (n_u_test + n_u_val) >= n_u:
                n_u_test = max(1, min(n_u_test, n_u - 2))
                n_u_val = max(1, min(n_u_val, n_u - n_u_test - 1))

            if temporal and "timestamp" in ratings_df.columns:
                user_ts = ratings_df.loc[idx_array, "timestamp"].to_numpy()
                sorted_idx = idx_array[np.argsort(user_ts)]
                test_indices.extend(sorted_idx[-n_u_test:])
                val_indices.extend(sorted_idx[-(n_u_test + n_u_val) : -n_u_test])
            else:
                shuffled = rng.permutation(idx_array)
                test_indices.extend(shuffled[:n_u_test])
                val_indices.extend(shuffled[n_u_test : n_u_test + n_u_val])

    val_set = set(val_indices)
    test_set = set(test_indices)
    all_indices = ratings_df.index.values

    train_mask = ~(np.isin(all_indices, list(val_set)) | np.isin(all_indices, list(test_set)))
    val_mask = np.isin(all_indices, list(val_set))
    test_mask = np.isin(all_indices, list(test_set))

    return (
        ratings_df[train_mask].copy().reset_index(drop=True),
        ratings_df[val_mask].copy().reset_index(drop=True),
        ratings_df[test_mask].copy().reset_index(drop=True),
    )
